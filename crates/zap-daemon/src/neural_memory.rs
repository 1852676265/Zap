//! 多智能体隔离记忆系统（SQLite WAL + FTS5 全文检索 + 后端提炼引擎）。
//!
//! 设计目标：
//! 1. 并发安全：全部状态落盘至单一 SQLite（WAL 模式），彻底消灭前端整文件覆盖丢失与多页签竞态。
//! 2. 极致省 Token：人设与 L1 核心档案通过会话冻结前缀注入（整场会话只出现一次、命中 Prompt Cache）；
//!    每轮动态检索的 L1 待办 / L2 索引 / L3 原文通过 `<system-reminder>` 按需尾注，界面展示自动剥离。
//! 3. 后台防垃圾提炼：回合结束直接在 `tokio::spawn` 中调底层 `LlmProvider` 异步提炼（不建临时会话），
//!    并内置寒暄/短确认过滤门禁，杜绝 `"可以"` 类垃圾神经元。
//! 4. 平滑无损迁移：首次初始化自动扫描并导入旧版插件目录与沙箱 JSON 数据，自动清理已知历史脏数据。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::{Datelike, Duration as ChronoDuration, Local, NaiveDate};
use futures::StreamExt;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use zap_capabilities::reminder::{strip_injected_reminders_for_display, system_reminder};
use zap_config::config::Config;
use zap_kernel::message::Message;
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;

// ---------------------------------------------------------------------------
// 常量与数据结构
// ---------------------------------------------------------------------------

const MAX_NON_PINNED_PROFILES: usize = 40;
const MAX_L3_EVIDENCE_CHARS: usize = 4500;
const DISTILL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(45);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralAgent {
    pub id: String,
    pub icon: String,
    pub name: String,
    pub label: String,
    pub color: String,
    pub hint: String,
    #[serde(default)]
    pub system_prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralConfig {
    #[serde(default = "default_active_agent")]
    pub active_agent: String,
    #[serde(default = "default_distill_mode")]
    pub distill_mode: String,
    #[serde(default)]
    pub distill_model: String,
    #[serde(default = "default_true")]
    pub timeline_auto_capture: bool,
    #[serde(default = "default_true")]
    pub show_floating_capsule: bool,
    #[serde(default)]
    pub memory_data_dir: String,
}

fn default_active_agent() -> String {
    "none".to_string()
}
fn default_distill_mode() -> String {
    "llm_auto".to_string()
}
fn default_true() -> bool {
    true
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            active_agent: default_active_agent(),
            distill_mode: default_distill_mode(),
            distill_model: String::new(),
            timeline_auto_capture: true,
            show_floating_capsule: true,
            memory_data_dir: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileFact {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub source_turn_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralTask {
    pub id: String,
    pub agent: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub working_dir: String,
    pub title: String,
    #[serde(default)]
    pub progress_note: String,
    #[serde(default)]
    pub remaining_note: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub source_turn_ids: Vec<String>,
    #[serde(default = "default_task_status")]
    pub status: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub done_at: String,
}

fn default_task_status() -> String {
    "in_progress".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeEvent {
    #[serde(default)]
    pub ts: String,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralNode {
    pub id: String,
    pub name: String,
    pub cat: String,
    #[serde(default)]
    pub project: String,
    #[serde(default = "default_node_status")]
    pub status: String,
    #[serde(default = "default_mastery")]
    pub mastery: i64,
    #[serde(default)]
    pub tier: i64,
    #[serde(default = "default_windows")]
    pub windows: Vec<String>,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub events: Vec<NodeEvent>,
    #[serde(default)]
    pub updated_at: String,
}

fn default_node_status() -> String {
    "active".to_string()
}
fn default_mastery() -> i64 {
    75
}
fn default_windows() -> Vec<String> {
    vec!["all".into(), "today".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralLink {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineEvent {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub ts: String,
    #[serde(default)]
    pub agent: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub turn_id: String,
    #[serde(default)]
    pub raw_file: String,
    #[serde(default)]
    pub user_goal: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub ai_outcome: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub touched_files: Vec<String>,
    #[serde(default)]
    pub distilled_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawTurn {
    pub turn_id: String,
    pub ts: String,
    pub agent: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default)]
    pub session_id: String,
    pub user_text: String,
    pub ai_text: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
}

// ---------------------------------------------------------------------------
// 默认智能体预设（保持空，仅从已有数据迁移或由用户手动创建）
// ---------------------------------------------------------------------------

fn default_agents() -> Vec<NeuralAgent> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// 存储单例与路径定位
// ---------------------------------------------------------------------------

struct StoreInner {
    data_dir: PathBuf,
    conn: Connection,
}

static STORE: OnceLock<Mutex<StoreInner>> = OnceLock::new();

#[derive(Serialize, Deserialize, Default)]
struct LocationPointer {
    #[serde(default)]
    data_dir: String,
}

fn pointer_file_path() -> PathBuf {
    zap_config::Config::config_dir()
        .join("neural-memory")
        .join("location.json")
}

fn default_data_dir() -> PathBuf {
    zap_config::Config::config_dir().join("neural-memory")
}

fn resolve_configured_data_dir() -> PathBuf {
    let ptr = pointer_file_path();
    if let Ok(raw) = std::fs::read_to_string(&ptr) {
        if let Ok(parsed) = serde_json::from_str::<LocationPointer>(&raw) {
            let trimmed = parsed.data_dir.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
    }
    default_data_dir()
}

fn save_data_dir_pointer(dir: &Path) -> Result<(), String> {
    let ptr = pointer_file_path();
    if let Some(parent) = ptr.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let payload = serde_json::to_string_pretty(&LocationPointer {
        data_dir: dir.to_string_lossy().into_owned(),
    })
    .map_err(|e| e.to_string())?;
    std::fs::write(&ptr, payload).map_err(|e| e.to_string())
}

fn open_connection(data_dir: &Path) -> Result<Connection, String> {
    std::fs::create_dir_all(data_dir).map_err(|e| format!("创建记忆目录失败: {e}"))?;
    let db_path = data_dir.join("neural_memory.db");
    let conn = Connection::open(&db_path).map_err(|e| format!("打开 SQLite 失败: {e}"))?;
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA busy_timeout = 5000;
        PRAGMA foreign_keys = OFF;

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            icon TEXT NOT NULL,
            name TEXT NOT NULL,
            label TEXT NOT NULL,
            color TEXT NOT NULL,
            hint TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS session_agents (
            session_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            text TEXT NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            source_turn_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_profiles_agent ON profiles(agent_id, pinned DESC, updated_at DESC);

        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            working_dir TEXT NOT NULL,
            title TEXT NOT NULL,
            progress_note TEXT NOT NULL,
            remaining_note TEXT NOT NULL,
            touched_files_json TEXT NOT NULL,
            session_id TEXT NOT NULL,
            source_turn_ids_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            done_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_status ON tasks(agent_id, status, updated_at DESC);

        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            project TEXT NOT NULL,
            status TEXT NOT NULL,
            mastery INTEGER NOT NULL,
            tier INTEGER NOT NULL,
            windows_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            events_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_nodes_agent ON nodes(agent_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS links (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            PRIMARY KEY (source, target)
        );

        CREATE TABLE IF NOT EXISTS timeline (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            working_dir TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            raw_file TEXT NOT NULL,
            user_goal TEXT NOT NULL,
            summary TEXT NOT NULL,
            ai_outcome TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            touched_files_json TEXT NOT NULL,
            distilled_by TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_timeline_agent_ts ON timeline(agent_id, ts DESC);

        CREATE TABLE IF NOT EXISTS raw_turns (
            turn_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            working_dir TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_text TEXT NOT NULL,
            ai_text TEXT NOT NULL,
            touched_files_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_raw_turns_agent_ts ON raw_turns(agent_id, ts DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            node_id UNINDEXED,
            agent_id UNINDEXED,
            search_text
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS timeline_fts USING fts5(
            event_id UNINDEXED,
            agent_id UNINDEXED,
            search_text
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS raw_turns_fts USING fts5(
            turn_id UNINDEXED,
            agent_id UNINDEXED,
            search_text
        );
        ",
    )
    .map_err(|e| format!("初始化 SQLite 表结构失败: {e}"))?;
    Ok(conn)
}

fn with_store<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut StoreInner) -> Result<R, String>,
{
    let lock = STORE.get_or_init(|| {
        let dir = resolve_configured_data_dir();
        let mut conn = match open_connection(&dir) {
            Ok(c) => c,
            Err(err) => {
                tracing::error!(error = %err, "neural_memory: open_connection failed, using in-memory fallback");
                Connection::open_in_memory().expect("in-memory sqlite")
            }
        };
        bootstrap_initial_data(&mut conn, &dir);
        Mutex::new(StoreInner {
            data_dir: dir,
            conn,
        })
    });
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

// ---------------------------------------------------------------------------
// 文本切词、防垃圾过滤与 FTS 索引辅助
// ---------------------------------------------------------------------------

fn now_ts_minute() -> String {
    Local::now().format("%Y-%m-%d %H:%M").to_string()
}

fn now_iso() -> String {
    Local::now().to_rfc3339()
}

fn new_id(prefix: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let rand_tail = &uuid::Uuid::new_v4().simple().to_string()[..6];
    format!("{prefix}-{ts:x}-{rand_tail}")
}

fn normalize_project_name(working_dir: &Path) -> String {
    let s = working_dir.to_string_lossy();
    let trimmed = s.trim().trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        return String::new();
    }
    Path::new(trimmed)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// 生成供 FTS5 检索的混合文本：保留英文/标识符原词，同时对中文片段补充 2-gram 空格分词。
fn build_fts_search_text(parts: &[&str]) -> String {
    let joined = parts.join(" ");
    let mut tokens: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // 原文词元
    for word in joined
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.')
        .filter(|w| w.chars().count() >= 2)
    {
        let low = word.to_lowercase();
        if seen.insert(low.clone()) {
            tokens.push(low);
        }
    }

    // 针对连续中文片段切出 2-gram 与 3-gram，供 FTS5 BM25 精准召回
    let mut cjk_run = Vec::new();
    let flush_cjk = |run: &mut Vec<char>, tokens: &mut Vec<String>, seen: &mut HashSet<String>| {
        if run.len() >= 2 {
            for win in run.windows(2) {
                let g: String = win.iter().collect();
                if seen.insert(g.clone()) {
                    tokens.push(g);
                }
            }
            if run.len() >= 3 {
                for win in run.windows(3) {
                    let g: String = win.iter().collect();
                    if seen.insert(g.clone()) {
                        tokens.push(g);
                    }
                }
            }
        }
        run.clear();
    };
    for ch in joined.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            cjk_run.push(ch);
        } else {
            flush_cjk(&mut cjk_run, &mut tokens, &mut seen);
        }
    }
    flush_cjk(&mut cjk_run, &mut tokens, &mut seen);

    if tokens.is_empty() {
        joined
    } else {
        format!("{joined} {}", tokens.join(" "))
    }
}

fn build_fts_or_query(query: &str) -> Option<String> {
    let mut terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for word in query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.chars().count() >= 2)
    {
        let clean = word.to_lowercase();
        if clean.len() <= 32 && seen.insert(clean.clone()) {
            terms.push(format!("\"{}\"", clean.replace('"', "")));
        }
    }

    let mut cjk_run = Vec::new();
    let flush_cjk = |run: &mut Vec<char>, terms: &mut Vec<String>, seen: &mut HashSet<String>| {
        if run.len() >= 2 {
            for win in run.windows(2) {
                let g: String = win.iter().collect();
                if seen.insert(g.clone()) {
                    terms.push(format!("\"{g}\""));
                }
            }
        }
        run.clear();
    };
    for ch in query.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            cjk_run.push(ch);
        } else {
            flush_cjk(&mut cjk_run, &mut terms, &mut seen);
        }
    }
    flush_cjk(&mut cjk_run, &mut terms, &mut seen);

    if terms.is_empty() {
        None
    } else {
        terms.truncate(24);
        Some(terms.join(" OR "))
    }
}

/// 剥离前端/后端注入的所有系统提醒块，还原用户的干净原始提问。
pub fn clean_user_prompt(raw: &str) -> String {
    strip_injected_reminders_for_display(raw).trim().to_string()
}

/// 防垃圾守门：过滤纯确认语/寒暄/无实质内容对话，防止污染 L1/L2 脑图与时间线。
fn is_trivial_turn(user_text: &str, ai_text: &str, touched_files: &[String]) -> bool {
    if !touched_files.is_empty() {
        return false;
    }
    let trimmed = user_text
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || "，。！？、~～… ".contains(c));
    if trimmed.is_empty() {
        return true;
    }
    let low = trimmed.to_lowercase();
    const NOISE_SET: &[&str] = &[
        "可以",
        "好的",
        "好",
        "好吧",
        "行",
        "嗯",
        "嗯嗯",
        "收到",
        "明白",
        "了解",
        "知道了",
        "谢谢",
        "多谢",
        "感谢",
        "辛苦了",
        "继续",
        "开始",
        "开始吧",
        "没问题",
        "可以的",
        "行吧",
        "ok",
        "okay",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "thanks",
        "thank you",
        "thx",
        "go",
        "go on",
        "continue",
        "done",
        "nice",
        "good",
        "great",
        "1",
        "2",
        "3",
    ];
    if NOISE_SET.contains(&low.as_str()) {
        return true;
    }
    let user_chars = trimmed.chars().count();
    let ai_chars = ai_text.trim().chars().count();
    if user_chars <= 4 && ai_chars <= 140 {
        return true;
    }
    false
}

fn is_noise_node_name(name: &str) -> bool {
    let trimmed = name
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || "，。！？、~～… ".contains(c));
    if trimmed.is_empty() || trimmed.chars().count() <= 1 {
        return true;
    }
    const NOISE_NAMES: &[&str] = &[
        "可以", "好的", "好", "嗯", "行", "谢谢", "收到", "继续", "ok", "yes", "no",
    ];
    NOISE_NAMES.contains(&trimmed.to_lowercase().as_str())
}

// ---------------------------------------------------------------------------
// 自动迁移旧版数据（含历史垃圾清理）
// ---------------------------------------------------------------------------

fn find_legacy_data_dirs() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    // 1. 开发/常用外置插件路径
    candidates.push(PathBuf::from("D:/插件/neural-agents/data"));
    // 2. 已安装插件目录下的 data/
    let plugins_root = zap_config::Config::config_dir().join("plugins");
    if let Ok(entries) = std::fs::read_dir(&plugins_root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.contains("neural") {
                    candidates.push(p.join("data"));
                }
                if let Ok(sub) = std::fs::read_dir(&p) {
                    for s in sub.flatten() {
                        if s.path().is_dir()
                            && s.file_name()
                                .to_string_lossy()
                                .to_lowercase()
                                .contains("neural")
                        {
                            candidates.push(s.path().join("data"));
                        }
                    }
                }
            }
        }
    }
    candidates
}

fn find_legacy_storage_json() -> Option<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let p = PathBuf::from(home)
        .join(".zap")
        .join("plugin-storage")
        .join("_zap_neural-agents")
        .join("storage.json");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

fn bootstrap_initial_data(conn: &mut Connection, _data_dir: &Path) {
    let already_migrated: bool = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'migration_done'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .ok()
        .flatten()
        .is_some();
    if already_migrated {
        return;
    }

    // 导入默认智能体（如果后续有 agents.json 会覆盖）
    let defaults = default_agents();
    for (idx, a) in defaults.iter().enumerate() {
        let _ = upsert_agent_conn(conn, a, idx as i64);
    }

    // 检查旧版插件目录与 sandbox storage.json
    let mut imported_any = false;
    for dir in find_legacy_data_dirs() {
        if dir.is_dir() && dir.join("agents.json").is_file() {
            if let Err(err) = import_legacy_dir_conn(conn, &dir) {
                tracing::warn!(dir = %dir.display(), error = %err, "neural_memory: legacy import error");
            } else {
                imported_any = true;
                tracing::info!(dir = %dir.display(), "neural_memory: imported legacy plugin data");
                break;
            }
        }
    }

    if let Some(storage_path) = find_legacy_storage_json() {
        if let Ok(raw) = std::fs::read_to_string(&storage_path) {
            if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                if let Some(obj) = val.as_object() {
                    let cfg = NeuralConfig {
                        active_agent: obj
                            .get("activeAgent")
                            .and_then(|v| v.as_str())
                            .unwrap_or("none")
                            .to_string(),
                        distill_mode: obj
                            .get("distillMode")
                            .and_then(|v| v.as_str())
                            .unwrap_or("llm_auto")
                            .to_string(),
                        distill_model: obj
                            .get("distillModel")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        timeline_auto_capture: obj
                            .get("timelineAutoCapture")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                        show_floating_capsule: obj
                            .get("showFloatingCapsule")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                        memory_data_dir: obj
                            .get("memoryDataDir")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    };
                    let _ = save_config_conn(conn, &cfg);
                    if let Some(smap) = obj.get("sessionAgentMap").and_then(|v| v.as_object()) {
                        for (sid, aid) in smap {
                            if let Some(a) = aid.as_str() {
                                let _ = set_session_agent_conn(conn, sid, a);
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('migration_done', ?1)",
        params![if imported_any { "imported" } else { "clean" }],
    );
}

fn import_legacy_dir_conn(conn: &mut Connection, dir: &Path) -> Result<Value, String> {
    let mut stats = json!({
        "agents": 0,
        "nodes": 0,
        "links": 0,
        "timeline": 0,
        "tasks": 0,
        "profiles": 0,
        "rawTurns": 0,
        "cleanedNoise": 0
    });
    let mut cleaned_noise: usize = 0;

    // 1. agents.json
    if let Ok(raw) = std::fs::read_to_string(dir.join("agents.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<NeuralAgent>>(&raw) {
            if !list.is_empty() {
                conn.execute("DELETE FROM agents", []).ok();
                for (idx, a) in list.iter().enumerate() {
                    upsert_agent_conn(conn, a, idx as i64)?;
                }
                stats["agents"] = json!(list.len());
            }
        }
    }

    // 2. session-map.json
    if let Ok(raw) = std::fs::read_to_string(dir.join("session-map.json")) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&raw) {
            for (sid, aid) in map {
                set_session_agent_conn(conn, &sid, &aid)?;
            }
        }
    }

    // 3. graph.json（过滤掉类似 `"可以"` 的垃圾节点）
    let mut removed_node_ids: HashSet<String> = HashSet::new();
    if let Ok(raw) = std::fs::read_to_string(dir.join("graph.json")) {
        if let Ok(val) = serde_json::from_str::<Value>(&raw) {
            let mut node_count = 0usize;
            if let Some(nodes) = val.get("nodes").and_then(|v| v.as_array()) {
                for nv in nodes {
                    if let Ok(n) = serde_json::from_value::<NeuralNode>(nv.clone()) {
                        if is_noise_node_name(&n.name) {
                            removed_node_ids.insert(n.id);
                            cleaned_noise += 1;
                            continue;
                        }
                        upsert_node_conn(conn, &n)?;
                        node_count += 1;
                    }
                }
            }
            stats["nodes"] = json!(node_count);

            let mut link_count = 0usize;
            if let Some(links) = val.get("links").and_then(|v| v.as_array()) {
                for lv in links {
                    let src = lv.get("source").and_then(|v| v.as_str()).unwrap_or("");
                    let tgt = lv.get("target").and_then(|v| v.as_str()).unwrap_or("");
                    if !src.is_empty()
                        && !tgt.is_empty()
                        && !removed_node_ids.contains(src)
                        && !removed_node_ids.contains(tgt)
                    {
                        conn.execute(
                            "INSERT OR IGNORE INTO links (source, target) VALUES (?1, ?2)",
                            params![src, tgt],
                        )
                        .ok();
                        link_count += 1;
                    }
                }
            }
            stats["links"] = json!(link_count);
        }
    }

    // 4. profiles.json
    if let Ok(raw) = std::fs::read_to_string(dir.join("profiles.json")) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<ProfileFact>>>(&raw) {
            let mut count = 0usize;
            for (agent_id, list) in map {
                for f in list {
                    upsert_profile_conn(conn, &agent_id, &f)?;
                    count += 1;
                }
            }
            stats["profiles"] = json!(count);
        }
    }

    // 5. tasks.json
    if let Ok(raw) = std::fs::read_to_string(dir.join("tasks.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<NeuralTask>>(&raw) {
            for t in &list {
                upsert_task_conn(conn, t)?;
            }
            stats["tasks"] = json!(list.len());
        }
    }

    // 6. timeline.json（清理空摘要、`"agent": "hubs"` 和单字寒暄噪声）
    if let Ok(raw) = std::fs::read_to_string(dir.join("timeline.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<TimelineEvent>>(&raw) {
            let mut count = 0usize;
            for mut ev in list {
                let sum = ev.summary.trim();
                let goal = ev.user_goal.trim();
                if ev.agent.trim().is_empty()
                    || ev.agent == "hubs"
                    || (sum.is_empty() && goal.is_empty())
                    || (is_noise_node_name(sum) && goal.is_empty())
                {
                    cleaned_noise += 1;
                    continue;
                }
                if ev.id.is_empty() {
                    ev.id = new_id("tl");
                }
                insert_timeline_conn(conn, &ev)?;
                count += 1;
            }
            stats["timeline"] = json!(count);
        }
    }

    // 7. raw/*.json 与 archive/*.json
    let mut raw_count = 0usize;
    for sub in ["raw", "archive"] {
        let sub_dir = dir.join(sub);
        if let Ok(entries) = std::fs::read_dir(&sub_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(raw) = std::fs::read_to_string(&p) {
                        if let Ok(turns) = serde_json::from_str::<Vec<RawTurn>>(&raw) {
                            for t in turns {
                                upsert_raw_turn_conn(conn, &t)?;
                                raw_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    stats["rawTurns"] = json!(raw_count);
    stats["cleanedNoise"] = json!(cleaned_noise);
    Ok(stats)
}

// ---------------------------------------------------------------------------
// 底层 SQLite CRUD 封装
// ---------------------------------------------------------------------------

fn upsert_agent_conn(conn: &Connection, a: &NeuralAgent, sort_order: i64) -> Result<(), String> {
    let label = if a.label.trim().is_empty() {
        format!("{} {}", a.icon, a.name)
    } else {
        a.label.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO agents (id, icon, name, label, color, hint, system_prompt, sort_order)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![a.id, a.icon, a.name, label, a.color, a.hint, a.system_prompt, sort_order],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn load_agents_conn(conn: &Connection) -> Result<Vec<NeuralAgent>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, icon, name, label, color, hint, system_prompt
             FROM agents ORDER BY sort_order ASC, id ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(NeuralAgent {
                id: row.get(0)?,
                icon: row.get(1)?,
                name: row.get(2)?,
                label: row.get(3)?,
                color: row.get(4)?,
                hint: row.get(5)?,
                system_prompt: row.get(6)?,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

fn get_config_conn(conn: &Connection) -> NeuralConfig {
    let mut cfg = NeuralConfig::default();
    if let Ok(mut stmt) = conn.prepare("SELECT key, value FROM meta") {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        {
            for r in rows.flatten() {
                match r.0.as_str() {
                    "cfg_active_agent" => cfg.active_agent = r.1,
                    "cfg_distill_mode" => cfg.distill_mode = r.1,
                    "cfg_distill_model" => cfg.distill_model = r.1,
                    "cfg_timeline_auto_capture" => cfg.timeline_auto_capture = r.1 != "false",
                    "cfg_show_floating_capsule" => cfg.show_floating_capsule = r.1 != "false",
                    "cfg_memory_data_dir" => cfg.memory_data_dir = r.1,
                    _ => {}
                }
            }
        }
    }
    cfg
}

fn save_config_conn(conn: &Connection, cfg: &NeuralConfig) -> Result<(), String> {
    let pairs = [
        ("cfg_active_agent", cfg.active_agent.as_str()),
        ("cfg_distill_mode", cfg.distill_mode.as_str()),
        ("cfg_distill_model", cfg.distill_model.as_str()),
        (
            "cfg_timeline_auto_capture",
            if cfg.timeline_auto_capture {
                "true"
            } else {
                "false"
            },
        ),
        (
            "cfg_show_floating_capsule",
            if cfg.show_floating_capsule {
                "true"
            } else {
                "false"
            },
        ),
        ("cfg_memory_data_dir", cfg.memory_data_dir.as_str()),
    ];
    for (k, v) in pairs {
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![k, v],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn set_session_agent_conn(
    conn: &Connection,
    session_id: &str,
    agent_id: &str,
) -> Result<(), String> {
    if session_id.trim().is_empty() {
        return Ok(());
    }
    conn.execute(
        "INSERT OR REPLACE INTO session_agents (session_id, agent_id, updated_at) VALUES (?1, ?2, ?3)",
        params![session_id.trim(), agent_id.trim(), now_iso()],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn resolve_agent_for_session_conn(conn: &Connection, session_id: &str) -> Option<NeuralAgent> {
    let cfg = get_config_conn(conn);
    let mapped: Option<String> = if session_id.trim().is_empty() {
        None
    } else {
        conn.query_row(
            "SELECT agent_id FROM session_agents WHERE session_id = ?1",
            params![session_id.trim()],
            |r| r.get(0),
        )
        .optional()
        .ok()
        .flatten()
    };
    let agent_id = mapped.unwrap_or_else(|| cfg.active_agent.clone());
    if agent_id.trim().is_empty() || agent_id == "none" {
        return None;
    }
    load_agents_conn(conn)
        .ok()?
        .into_iter()
        .find(|a| a.id == agent_id)
}

fn upsert_profile_conn(conn: &Connection, agent_id: &str, f: &ProfileFact) -> Result<(), String> {
    let id = if f.id.trim().is_empty() {
        new_id("pf")
    } else {
        f.id.clone()
    };
    let updated_at = if f.updated_at.trim().is_empty() {
        now_ts_minute()
    } else {
        f.updated_at.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO profiles (id, agent_id, project, text, pinned, updated_at, source_turn_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            id,
            agent_id,
            f.project,
            f.text,
            if f.pinned { 1 } else { 0 },
            updated_at,
            f.source_turn_id
        ],
    )
    .map_err(|e| e.to_string())?;

    // 只淘汰非置顶的老事实（保留最近 MAX_NON_PINNED_PROFILES 条）
    conn.execute(
        "DELETE FROM profiles WHERE agent_id = ?1 AND pinned = 0 AND id NOT IN (
             SELECT id FROM profiles WHERE agent_id = ?1 AND pinned = 0
             ORDER BY updated_at DESC LIMIT ?2
         )",
        params![agent_id, MAX_NON_PINNED_PROFILES as i64],
    )
    .ok();
    Ok(())
}

fn load_profiles_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<HashMap<String, Vec<ProfileFact>>, String> {
    let mut map: HashMap<String, Vec<ProfileFact>> = HashMap::new();
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, project, text, pinned, updated_at, source_turn_id
         FROM profiles WHERE agent_id = ?1 ORDER BY pinned DESC, updated_at DESC"
    } else {
        "SELECT id, agent_id, project, text, pinned, updated_at, source_turn_id
         FROM profiles ORDER BY pinned DESC, updated_at DESC"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(String, ProfileFact)> {
        let aid: String = row.get(1)?;
        let pinned_int: i64 = row.get(4)?;
        Ok((
            aid,
            ProfileFact {
                id: row.get(0)?,
                project: row.get(2)?,
                text: row.get(3)?,
                pinned: pinned_int != 0,
                updated_at: row.get(5)?,
                source_turn_id: row.get(6)?,
            },
        ))
    };
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            let (a, f) = r.map_err(|e| e.to_string())?;
            map.entry(a).or_default().push(f);
        }
    } else {
        let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
        for r in rows {
            let (a, f) = r.map_err(|e| e.to_string())?;
            map.entry(a).or_default().push(f);
        }
    }
    Ok(map)
}

fn upsert_task_conn(conn: &Connection, t: &NeuralTask) -> Result<(), String> {
    let touched = serde_json::to_string(&t.touched_files).unwrap_or_else(|_| "[]".into());
    let turns = serde_json::to_string(&t.source_turn_ids).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "INSERT OR REPLACE INTO tasks (
            id, agent_id, project, working_dir, title, progress_note, remaining_note,
            touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            t.id,
            t.agent,
            t.project,
            t.working_dir,
            t.title,
            t.progress_note,
            t.remaining_note,
            touched,
            t.session_id,
            turns,
            t.status,
            t.created_at,
            t.updated_at,
            t.done_at
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn load_tasks_conn(conn: &Connection, agent_id: Option<&str>) -> Result<Vec<NeuralTask>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, project, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
         FROM tasks WHERE agent_id = ?1 ORDER BY updated_at DESC"
    } else {
        "SELECT id, agent_id, project, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
         FROM tasks ORDER BY updated_at DESC"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<NeuralTask> {
        let touched_str: String = row.get(7)?;
        let turns_str: String = row.get(9)?;
        Ok(NeuralTask {
            id: row.get(0)?,
            agent: row.get(1)?,
            project: row.get(2)?,
            working_dir: row.get(3)?,
            title: row.get(4)?,
            progress_note: row.get(5)?,
            remaining_note: row.get(6)?,
            touched_files: serde_json::from_str(&touched_str).unwrap_or_default(),
            session_id: row.get(8)?,
            source_turn_ids: serde_json::from_str(&turns_str).unwrap_or_default(),
            status: row.get(10)?,
            created_at: row.get(11)?,
            updated_at: row.get(12)?,
            done_at: row.get(13)?,
        })
    };
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

fn upsert_node_conn(conn: &Connection, n: &NeuralNode) -> Result<(), String> {
    let windows_json = serde_json::to_string(&n.windows).unwrap_or_else(|_| "[\"all\"]".into());
    let events_json = serde_json::to_string(&n.events).unwrap_or_else(|_| "[]".into());
    let updated_at = if n.updated_at.trim().is_empty() {
        now_ts_minute()
    } else {
        n.updated_at.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO nodes (id, agent_id, name, project, status, mastery, tier, windows_json, summary, events_json, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            n.id, n.cat, n.name, n.project, n.status, n.mastery, n.tier, windows_json, n.summary,
            events_json, updated_at
        ],
    )
    .map_err(|e| e.to_string())?;

    // 同步 FTS5 索引
    conn.execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![n.id])
        .ok();
    let event_texts: Vec<&str> = n.events.iter().map(|e| e.text.as_str()).collect();
    let ev_joined = event_texts.join(" ");
    let search_text = build_fts_search_text(&[&n.name, &n.project, &n.summary, &ev_joined]);
    conn.execute(
        "INSERT INTO nodes_fts (node_id, agent_id, search_text) VALUES (?1, ?2, ?3)",
        params![n.id, n.cat, search_text],
    )
    .ok();
    Ok(())
}

fn load_nodes_conn(conn: &Connection, agent_id: Option<&str>) -> Result<Vec<NeuralNode>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, name, project, status, mastery, tier, windows_json, summary, events_json, updated_at
         FROM nodes WHERE agent_id = ?1 ORDER BY updated_at DESC"
    } else {
        "SELECT id, agent_id, name, project, status, mastery, tier, windows_json, summary, events_json, updated_at
         FROM nodes ORDER BY updated_at DESC"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<NeuralNode> {
        let win_str: String = row.get(7)?;
        let ev_str: String = row.get(9)?;
        Ok(NeuralNode {
            id: row.get(0)?,
            cat: row.get(1)?,
            name: row.get(2)?,
            project: row.get(3)?,
            status: row.get(4)?,
            mastery: row.get(5)?,
            tier: row.get(6)?,
            windows: serde_json::from_str(&win_str).unwrap_or_else(|_| default_windows()),
            summary: row.get(8)?,
            events: serde_json::from_str(&ev_str).unwrap_or_default(),
            updated_at: row.get(10)?,
        })
    };
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

fn load_links_conn(conn: &Connection) -> Result<Vec<NeuralLink>, String> {
    let mut stmt = conn
        .prepare("SELECT source, target FROM links")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok(NeuralLink {
                source: r.get(0)?,
                target: r.get(1)?,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

fn insert_timeline_conn(conn: &Connection, ev: &TimelineEvent) -> Result<(), String> {
    let id = if ev.id.trim().is_empty() {
        new_id("tl")
    } else {
        ev.id.clone()
    };
    let kw_json = serde_json::to_string(&ev.keywords).unwrap_or_else(|_| "[]".into());
    let tf_json = serde_json::to_string(&ev.touched_files).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "INSERT OR REPLACE INTO timeline (
            id, ts, agent_id, project, working_dir, session_id, turn_id, raw_file,
            user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            id,
            ev.ts,
            ev.agent,
            ev.project,
            ev.working_dir,
            ev.session_id,
            ev.turn_id,
            ev.raw_file,
            ev.user_goal,
            ev.summary,
            ev.ai_outcome,
            kw_json,
            tf_json,
            ev.distilled_by
        ],
    )
    .map_err(|e| e.to_string())?;

    conn.execute("DELETE FROM timeline_fts WHERE event_id = ?1", params![id])
        .ok();
    let kw_joined = ev.keywords.join(" ");
    let tf_joined = ev.touched_files.join(" ");
    let search_text = build_fts_search_text(&[
        &ev.project,
        &ev.user_goal,
        &ev.summary,
        &ev.ai_outcome,
        &kw_joined,
        &tf_joined,
    ]);
    conn.execute(
        "INSERT INTO timeline_fts (event_id, agent_id, search_text) VALUES (?1, ?2, ?3)",
        params![id, ev.agent, search_text],
    )
    .ok();
    Ok(())
}

fn load_timeline_conn(
    conn: &Connection,
    agent_id: Option<&str>,
    limit: usize,
) -> Result<Vec<TimelineEvent>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, ts, agent_id, project, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
         FROM timeline WHERE agent_id = ?1 ORDER BY ts DESC LIMIT ?2"
    } else {
        "SELECT id, ts, agent_id, project, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
         FROM timeline ORDER BY ts DESC LIMIT ?1"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<TimelineEvent> {
        let kw_str: String = row.get(11)?;
        let tf_str: String = row.get(12)?;
        Ok(TimelineEvent {
            id: row.get(0)?,
            ts: row.get(1)?,
            agent: row.get(2)?,
            project: row.get(3)?,
            working_dir: row.get(4)?,
            session_id: row.get(5)?,
            turn_id: row.get(6)?,
            raw_file: row.get(7)?,
            user_goal: row.get(8)?,
            summary: row.get(9)?,
            ai_outcome: row.get(10)?,
            keywords: serde_json::from_str(&kw_str).unwrap_or_default(),
            touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
            distilled_by: row.get(13)?,
        })
    };
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid, limit as i64], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt
            .query_map(params![limit as i64], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

fn upsert_raw_turn_conn(conn: &Connection, t: &RawTurn) -> Result<(), String> {
    let tf_json = serde_json::to_string(&t.touched_files).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "INSERT OR REPLACE INTO raw_turns (turn_id, ts, agent_id, project, working_dir, session_id, user_text, ai_text, touched_files_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            t.turn_id,
            t.ts,
            t.agent,
            t.project,
            t.working_dir,
            t.session_id,
            t.user_text,
            t.ai_text,
            tf_json
        ],
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "DELETE FROM raw_turns_fts WHERE turn_id = ?1",
        params![t.turn_id],
    )
    .ok();
    let tf_joined = t.touched_files.join(" ");
    let search_text = build_fts_search_text(&[&t.project, &t.user_text, &t.ai_text, &tf_joined]);
    conn.execute(
        "INSERT INTO raw_turns_fts (turn_id, agent_id, search_text) VALUES (?1, ?2, ?3)",
        params![t.turn_id, t.agent, search_text],
    )
    .ok();
    Ok(())
}

fn load_raw_turn_conn(conn: &Connection, turn_id: &str) -> Result<Option<RawTurn>, String> {
    conn.query_row(
        "SELECT turn_id, ts, agent_id, project, working_dir, session_id, user_text, ai_text, touched_files_json
         FROM raw_turns WHERE turn_id = ?1",
        params![turn_id],
        |row| {
            let tf_str: String = row.get(8)?;
            Ok(RawTurn {
                turn_id: row.get(0)?,
                ts: row.get(1)?,
                agent: row.get(2)?,
                project: row.get(3)?,
                working_dir: row.get(4)?,
                session_id: row.get(5)?,
                user_text: row.get(6)?,
                ai_text: row.get(7)?,
                touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
            })
        },
    )
    .optional()
    .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// 时间范围解析 + FTS5 召回与上下文注入构造
// ---------------------------------------------------------------------------

struct DateRange {
    label: String,
    start: String,
    end: String,
}

fn parse_temporal_range(query: &str) -> Option<DateRange> {
    let today = Local::now().date_naive();
    let fmt_day = |d: NaiveDate| d.format("%Y-%m-%d").to_string();

    if query.contains("今天") || query.contains("今日") {
        let s = fmt_day(today);
        return Some(DateRange {
            label: "今天".into(),
            start: format!("{s} 00:00"),
            end: format!("{s} 23:59"),
        });
    }
    if query.contains("昨天") || query.contains("昨日") {
        let d = today - ChronoDuration::days(1);
        let s = fmt_day(d);
        return Some(DateRange {
            label: "昨天".into(),
            start: format!("{s} 00:00"),
            end: format!("{s} 23:59"),
        });
    }
    if query.contains("前天") {
        let d = today - ChronoDuration::days(2);
        let s = fmt_day(d);
        return Some(DateRange {
            label: "前天".into(),
            start: format!("{s} 00:00"),
            end: format!("{s} 23:59"),
        });
    }
    if query.contains("本周") || query.contains("这周") || query.contains("这个星期") {
        let offset = today.weekday().num_days_from_monday() as i64;
        let mon = today - ChronoDuration::days(offset);
        return Some(DateRange {
            label: "本周".into(),
            start: format!("{} 00:00", fmt_day(mon)),
            end: format!("{} 23:59", fmt_day(today)),
        });
    }
    if query.contains("上周") || query.contains("上星期") {
        let offset = today.weekday().num_days_from_monday() as i64;
        let this_mon = today - ChronoDuration::days(offset);
        let last_mon = this_mon - ChronoDuration::days(7);
        let last_sun = this_mon - ChronoDuration::days(1);
        return Some(DateRange {
            label: "上周".into(),
            start: format!("{} 00:00", fmt_day(last_mon)),
            end: format!("{} 23:59", fmt_day(last_sun)),
        });
    }
    if query.contains("本月") || query.contains("这个月") {
        let start = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?;
        return Some(DateRange {
            label: "本月".into(),
            start: format!("{} 00:00", fmt_day(start)),
            end: format!("{} 23:59", fmt_day(today)),
        });
    }
    if query.contains("上月") || query.contains("上个月") {
        let (y, m) = if today.month() == 1 {
            (today.year() - 1, 12)
        } else {
            (today.year(), today.month() - 1)
        };
        let start = NaiveDate::from_ymd_opt(y, m, 1)?;
        let next_month = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?;
        let end = next_month - ChronoDuration::days(1);
        return Some(DateRange {
            label: "上个月".into(),
            start: format!("{} 00:00", fmt_day(start)),
            end: format!("{} 23:59", fmt_day(end)),
        });
    }
    if query.contains("今年") {
        return Some(DateRange {
            label: "今年".into(),
            start: format!("{}-01-01 00:00", today.year()),
            end: format!("{}-12-31 23:59", today.year()),
        });
    }
    if query.contains("去年") {
        return Some(DateRange {
            label: "去年".into(),
            start: format!("{}-01-01 00:00", today.year() - 1),
            end: format!("{}-12-31 23:59", today.year() - 1),
        });
    }
    // 解析显式 `YYYY-MM-DD` 或 `YYYY/MM/DD`
    for token in query.split(|c: char| !c.is_ascii_digit() && c != '-' && c != '/') {
        if token.len() == 10 {
            let norm = token.replace('/', "-");
            if NaiveDate::parse_from_str(&norm, "%Y-%m-%d").is_ok() {
                return Some(DateRange {
                    label: norm.clone(),
                    start: format!("{norm} 00:00"),
                    end: format!("{norm} 23:59"),
                });
            }
        }
    }
    None
}

fn should_inject_l3_verbatim(query: &str) -> bool {
    const HINTS: &[&str] = &[
        "详细",
        "原文",
        "原话",
        "代码",
        "报错",
        "堆栈",
        "sql",
        "表名",
        "字段",
        "怎么写",
        "回顾",
        "展开",
        "具体",
        "细节",
        "当时",
        "之前说的",
        "上次",
        "怎么改",
    ];
    let low = query.to_lowercase();
    HINTS.iter().any(|h| low.contains(h))
}

fn build_static_system_block_conn(
    conn: &Connection,
    session_id: &str,
) -> Result<Option<String>, String> {
    let Some(agent) = resolve_agent_for_session_conn(conn, session_id) else {
        return Ok(None);
    };
    let profiles_map = load_profiles_conn(conn, Some(&agent.id))?;
    let facts = profiles_map.get(&agent.id).cloned().unwrap_or_default();

    let mut parts = Vec::new();
    parts.push(format!(
        "[Neural Agent Active: {} ({})]\n当前会话已启用独立记忆隔离域：只能引用本智能体记忆，严禁混用其他角色上下文。",
        agent.label, agent.id
    ));
    if !agent.system_prompt.trim().is_empty() {
        parts.push(format!(
            "[Agent System Prompt]\n{}",
            agent.system_prompt.trim()
        ));
    }
    if !facts.is_empty() {
        let lines: Vec<String> = facts
            .iter()
            .take(18)
            .map(|f| {
                let pin = if f.pinned { "[置顶] " } else { "" };
                let proj = if f.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", f.project)
                };
                format!("- {pin}{proj}{}", f.text)
            })
            .collect();
        parts.push(format!("[L1 Core Agent Profile]\n{}", lines.join("\n")));
    }
    Ok(Some(parts.join("\n\n")))
}

/// 静态会话前缀块：注入智能体人设 + L1 长期核心档案。
/// 会通过 `extra_system_append` 进入冻结 `<project_instructions>` 块，整场对话只放一次、命中 Prompt Cache。
pub fn build_static_system_block(session_id: &str) -> Option<String> {
    with_store(|store| build_static_system_block_conn(&store.conn, session_id))
        .ok()
        .flatten()
}

static INJECTED_STATIC_AGENTS: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

fn record_injected_static_agent(session_id: &str, agent_id: &str) {
    let map = INJECTED_STATIC_AGENTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
    guard.insert(session_id.to_string(), agent_id.to_string());
}

fn static_agent_matches(session_id: &str, agent_id: &str) -> bool {
    let map = INJECTED_STATIC_AGENTS.get_or_init(|| Mutex::new(HashMap::new()));
    let guard = map.lock().unwrap_or_else(|e| e.into_inner());
    guard.get(session_id).map(String::as_str) == Some(agent_id)
}

/// 将本会话对应智能体的静态人设与 L1 核心档案合入 `extra_system_append`。
/// 通过冻结会话前缀注入，整场对话只占一次上下文，命中 Prompt Cache。
pub fn apply_session_static_system_append(
    runtime: &mut zap_coding::CodingRuntimeConfig,
    session_id: Option<&str>,
) {
    let Some(sid) = session_id.map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };
    let agent_id =
        with_store(|store| Ok(resolve_agent_for_session_conn(&store.conn, sid).map(|a| a.id)))
            .ok()
            .flatten();
    if let Some(ref aid) = agent_id {
        record_injected_static_agent(sid, aid);
    } else {
        record_injected_static_agent(sid, "none");
    }
    let Some(block) = build_static_system_block(sid) else {
        return;
    };
    match runtime.extra_system_append.as_mut() {
        Some(existing) if !existing.trim().is_empty() => {
            existing.push_str("\n\n");
            existing.push_str(&block);
        }
        _ => {
            runtime.extra_system_append = Some(block);
        }
    }
}

/// 将动态检索的 `<system-reminder>` 追加到送往内核的 `runtime_text` 末尾；
/// 聊天展示与存盘会自动剥离 `<system-reminder>`，前端气泡保持干净。
pub fn append_dynamic_reminder_to_text(
    session_id: &str,
    base_user_text: &str,
    working_dir: &Path,
    mut runtime_text: String,
) -> String {
    if let Some(reminder) = build_dynamic_turn_reminder(session_id, base_user_text, working_dir) {
        if !runtime_text.trim().is_empty() {
            runtime_text.push_str("\n\n");
        }
        runtime_text.push_str(&reminder);
    }
    runtime_text
}
/// 动态单轮提示词尾注：按本轮问题从 SQLite 检索 L1 待办 + L2 节点/时间线 + 按需 L3 原文，
/// 封装在 `<system-reminder>` 中附着到送给内核的用户消息末尾（界面气泡会自动剥离，保持干净）。
pub fn build_dynamic_turn_reminder(
    session_id: &str,
    user_text: &str,
    working_dir: &Path,
) -> Option<String> {
    let clean_q = clean_user_prompt(user_text);
    if clean_q.is_empty() {
        return None;
    }
    with_store(|store| {
        let Some(agent) = resolve_agent_for_session_conn(&store.conn, session_id) else {
            return Ok(None);
        };
        let proj = normalize_project_name(working_dir);
        let mut sections: Vec<String> = Vec::new();

        // 若在持久 /live 会话中途切换了智能体，运行时冻结前缀仍是旧角色，则在动态区补充新角色规范
        if !static_agent_matches(session_id, &agent.id) {
            if let Ok(Some(static_blk)) = build_static_system_block_conn(&store.conn, session_id) {
                sections.push(format!("[Switched Neural Agent Override]\n{static_blk}"));
            }
        }

        // 1. L1 未完成任务（仅注入 open / stale）
        let all_tasks = load_tasks_conn(&store.conn, Some(&agent.id))?;
        let open_tasks: Vec<&NeuralTask> = all_tasks
            .iter()
            .filter(|t| t.status != "done")
            .take(8)
            .collect();
        if !open_tasks.is_empty() {
            let mut lines = Vec::new();
            for t in open_tasks {
                let tag = if t.status == "stale" {
                    "待确认/陈旧"
                } else {
                    "进行中"
                };
                let p = if t.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", t.project)
                };
                let rem = if t.remaining_note.is_empty() {
                    String::new()
                } else {
                    format!(" | 待办: {}", t.remaining_note)
                };
                lines.push(format!("- ({tag}, id={}) {p}{}{rem}", t.id, t.title));
            }
            sections.push(format!("[L1 Pinned Open Tasks]\n{}", lines.join("\n")));
        }

        // 2. L2 知识节点（FTS5 检索 + 最近活跃兜底）
        let fts_q = build_fts_or_query(&clean_q);
        let mut matched_nodes: Vec<NeuralNode> = Vec::new();
        let mut seen_node_ids: HashSet<String> = HashSet::new();

        if let Some(ref fq) = fts_q {
            if let Ok(mut stmt) = store.conn.prepare(
                "SELECT n.id, n.agent_id, n.name, n.project, n.status, n.mastery, n.tier,
                        n.windows_json, n.summary, n.events_json, n.updated_at
                 FROM nodes_fts
                 JOIN nodes n ON n.id = nodes_fts.node_id
                 WHERE n.agent_id = ?1 AND nodes_fts MATCH ?2
                 ORDER BY bm25(nodes_fts) ASC
                 LIMIT 8",
            ) {
                if let Ok(rows) = stmt.query_map(params![agent.id, fq], |row| {
                    let win_str: String = row.get(7)?;
                    let ev_str: String = row.get(9)?;
                    Ok(NeuralNode {
                        id: row.get(0)?,
                        cat: row.get(1)?,
                        name: row.get(2)?,
                        project: row.get(3)?,
                        status: row.get(4)?,
                        mastery: row.get(5)?,
                        tier: row.get(6)?,
                        windows: serde_json::from_str(&win_str).unwrap_or_else(|_| default_windows()),
                        summary: row.get(8)?,
                        events: serde_json::from_str(&ev_str).unwrap_or_default(),
                        updated_at: row.get(10)?,
                    })
                }) {
                    for n in rows.flatten() {
                        if seen_node_ids.insert(n.id.clone()) {
                            matched_nodes.push(n);
                        }
                    }
                }
            }
        }
        if matched_nodes.len() < 5 {
            let recent = load_nodes_conn(&store.conn, Some(&agent.id))?;
            // 当前项目或待复习节点优先
            let mut sorted = recent;
            sorted.sort_by_key(|n| {
                let proj_hit = if !proj.is_empty() && n.project == proj { 0 } else { 1 };
                let rev_hit = if n.status == "review" { 0 } else { 1 };
                (proj_hit, rev_hit)
            });
            for n in sorted {
                if matched_nodes.len() >= 8 {
                    break;
                }
                if seen_node_ids.insert(n.id.clone()) {
                    matched_nodes.push(n);
                }
            }
        }
        if !matched_nodes.is_empty() {
            let lines: Vec<String> = matched_nodes
                .iter()
                .map(|n| {
                    let p = if n.project.is_empty() {
                        String::new()
                    } else {
                        format!("[{}] ", n.project)
                    };
                    format!("- {p}{} (status={}, mastery={}%): {}", n.name, n.status, n.mastery, n.summary)
                })
                .collect();
            sections.push(format!("[L2 Knowledge Nodes]\n{}", lines.join("\n")));
        }

        // 3. L2 时间线索引（日期范围感知 OR FTS5 BM25 + 最近记录）
        let date_range = parse_temporal_range(&clean_q);
        let mut timeline_hits: Vec<TimelineEvent> = Vec::new();
        let mut seen_tl_ids: HashSet<String> = HashSet::new();

        if let Some(ref dr) = date_range {
            if let Ok(mut stmt) = store.conn.prepare(
                "SELECT id, ts, agent_id, project, working_dir, session_id, turn_id, raw_file,
                        user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
                 FROM timeline
                 WHERE agent_id = ?1 AND ts >= ?2 AND ts <= ?3
                 ORDER BY ts DESC LIMIT 10",
            ) {
                if let Ok(rows) = stmt.query_map(params![agent.id, dr.start, dr.end], |row| {
                    let kw_str: String = row.get(11)?;
                    let tf_str: String = row.get(12)?;
                    Ok(TimelineEvent {
                        id: row.get(0)?,
                        ts: row.get(1)?,
                        agent: row.get(2)?,
                        project: row.get(3)?,
                        working_dir: row.get(4)?,
                        session_id: row.get(5)?,
                        turn_id: row.get(6)?,
                        raw_file: row.get(7)?,
                        user_goal: row.get(8)?,
                        summary: row.get(9)?,
                        ai_outcome: row.get(10)?,
                        keywords: serde_json::from_str(&kw_str).unwrap_or_default(),
                        touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
                        distilled_by: row.get(13)?,
                    })
                }) {
                    for ev in rows.flatten() {
                        if seen_tl_ids.insert(ev.id.clone()) {
                            timeline_hits.push(ev);
                        }
                    }
                }
            }
        }
        if timeline_hits.len() < 8 {
            if let Some(ref fq) = fts_q {
                if let Ok(mut stmt) = store.conn.prepare(
                    "SELECT t.id, t.ts, t.agent_id, t.project, t.working_dir, t.session_id, t.turn_id, t.raw_file,
                            t.user_goal, t.summary, t.ai_outcome, t.keywords_json, t.touched_files_json, t.distilled_by
                     FROM timeline_fts
                     JOIN timeline t ON t.id = timeline_fts.event_id
                     WHERE t.agent_id = ?1 AND timeline_fts MATCH ?2
                     ORDER BY bm25(timeline_fts) ASC
                     LIMIT 6",
                ) {
                    if let Ok(rows) = stmt.query_map(params![agent.id, fq], |row| {
                        let kw_str: String = row.get(11)?;
                        let tf_str: String = row.get(12)?;
                        Ok(TimelineEvent {
                            id: row.get(0)?,
                            ts: row.get(1)?,
                            agent: row.get(2)?,
                            project: row.get(3)?,
                            working_dir: row.get(4)?,
                            session_id: row.get(5)?,
                            turn_id: row.get(6)?,
                            raw_file: row.get(7)?,
                            user_goal: row.get(8)?,
                            summary: row.get(9)?,
                            ai_outcome: row.get(10)?,
                            keywords: serde_json::from_str(&kw_str).unwrap_or_default(),
                            touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
                            distilled_by: row.get(13)?,
                        })
                    }) {
                        for ev in rows.flatten() {
                            if timeline_hits.len() >= 8 {
                                break;
                            }
                            if seen_tl_ids.insert(ev.id.clone()) {
                                timeline_hits.push(ev);
                            }
                        }
                    }
                }
            }
        }
        if timeline_hits.len() < 4 && date_range.is_none() {
            for ev in load_timeline_conn(&store.conn, Some(&agent.id), 4)? {
                if timeline_hits.len() >= 6 {
                    break;
                }
                if seen_tl_ids.insert(ev.id.clone()) {
                    timeline_hits.push(ev);
                }
            }
        }
        if !timeline_hits.is_empty() {
            let header = match &date_range {
                Some(dr) => format!("[L2 Period Index: {}]", dr.label),
                None => "[L2 Timeline Index]".to_string(),
            };
            let lines: Vec<String> = timeline_hits
                .iter()
                .map(|ev| {
                    let p = if ev.project.is_empty() {
                        String::new()
                    } else {
                        format!("[{}] ", ev.project)
                    };
                    let out = if ev.ai_outcome.is_empty() {
                        String::new()
                    } else {
                        format!(" -> {}", ev.ai_outcome)
                    };
                    format!("- [{}] {p}{}{out}", ev.ts, ev.summary)
                })
                .collect();
            sections.push(format!("{header}\n{}", lines.join("\n")));
        }

        // 4. L3 原文按需调取（限制在预算内）
        if should_inject_l3_verbatim(&clean_q) {
            let mut raw_hits: Vec<RawTurn> = Vec::new();
            let mut seen_turns: HashSet<String> = HashSet::new();

            if let Some(ref fq) = fts_q {
                if let Ok(mut stmt) = store.conn.prepare(
                    "SELECT r.turn_id, r.ts, r.agent_id, r.project, r.working_dir, r.session_id,
                            r.user_text, r.ai_text, r.touched_files_json
                     FROM raw_turns_fts
                     JOIN raw_turns r ON r.turn_id = raw_turns_fts.turn_id
                     WHERE r.agent_id = ?1 AND raw_turns_fts MATCH ?2
                     ORDER BY bm25(raw_turns_fts) ASC
                     LIMIT 3",
                ) {
                    if let Ok(rows) = stmt.query_map(params![agent.id, fq], |row| {
                        let tf_str: String = row.get(8)?;
                        Ok(RawTurn {
                            turn_id: row.get(0)?,
                            ts: row.get(1)?,
                            agent: row.get(2)?,
                            project: row.get(3)?,
                            working_dir: row.get(4)?,
                            session_id: row.get(5)?,
                            user_text: row.get(6)?,
                            ai_text: row.get(7)?,
                            touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
                        })
                    }) {
                        for r in rows.flatten() {
                            if seen_turns.insert(r.turn_id.clone()) {
                                raw_hits.push(r);
                            }
                        }
                    }
                }
            }
            // 若命中了时间范围但 FTS5 没凑够，用时间线指向的 turn_id 补齐
            if raw_hits.is_empty() {
                for ev in timeline_hits.iter().take(2) {
                    if !ev.turn_id.is_empty() && seen_turns.insert(ev.turn_id.clone()) {
                        if let Ok(Some(r)) = load_raw_turn_conn(&store.conn, &ev.turn_id) {
                            raw_hits.push(r);
                        }
                    }
                }
            }
            if !raw_hits.is_empty() {
                let mut total_chars = 0usize;
                let mut blocks = Vec::new();
                for r in raw_hits {
                    let u: String = r.user_text.chars().take(700).collect();
                    let a: String = r.ai_text.chars().take(1200).collect();
                    let blk = format!("--- [{}] [{}] ---\nUser: {u}\nAssistant: {a}", r.ts, r.project);
                    total_chars += blk.chars().count();
                    if total_chars > MAX_L3_EVIDENCE_CHARS && !blocks.is_empty() {
                        break;
                    }
                    blocks.push(blk);
                }
                if !blocks.is_empty() {
                    sections.push(format!("[L3 Verbatim Raw Evidence]\n{}", blocks.join("\n\n")));
                }
            }
        }

        if sections.is_empty() {
            return Ok(None);
        }
        Ok(Some(system_reminder(&sections.join("\n\n"))))
    })
    .ok()
    .flatten()
}

// ---------------------------------------------------------------------------
// 后台异步提炼（脱离浏览器生命周期，不建临时会话）
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DistillProfileUpdate {
    text: String,
    #[serde(default)]
    pin: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DistillProfileInput {
    Obj(DistillProfileUpdate),
    Str(String),
}

#[derive(Debug, Deserialize)]
struct DistillTaskAction {
    action: String,
    #[serde(default)]
    task_id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    progress_note: String,
    #[serde(default)]
    remaining_note: String,
}

#[derive(Debug, Deserialize)]
struct DistillNodeAction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    summary: String,
    #[serde(default = "default_mastery")]
    mastery: i64,
    #[serde(default = "default_node_status")]
    status: String,
}

#[derive(Debug, Deserialize)]
struct DistillTimeline {
    #[serde(default)]
    user_goal: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    ai_outcome: String,
    #[serde(default)]
    keywords: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DistillOutput {
    #[serde(default = "default_true")]
    should_record: bool,
    #[serde(default)]
    profile_updates: Vec<DistillProfileInput>,
    #[serde(default)]
    profile_removes: Vec<String>,
    #[serde(default)]
    task_actions: Vec<DistillTaskAction>,
    #[serde(default)]
    node_actions: Vec<DistillNodeAction>,
    #[serde(default)]
    timeline: Option<DistillTimeline>,
}

fn parse_distill_json(raw: &str) -> Option<DistillOutput> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<DistillOutput>(trimmed) {
        return Some(parsed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<DistillOutput>(&trimmed[start..=end]).ok()
}

/// 从最终会话快照提取本轮干净的 User / Assistant 文本以及触及的文件路径。
pub fn extract_last_turn_payload(messages: &[Message]) -> Option<(String, String, Vec<String>)> {
    // 找最后一条真实非合成 User 消息索引
    let last_user_idx = messages.iter().rposition(|m| {
        matches!(m.role, zap_kernel::message::Role::User)
            && !m.synthetic
            && !zap_capabilities::reminder::is_system_reminder(&m.text)
            && !zap_capabilities::reminder::is_strong_reminder(&m.text)
    })?;

    let user_text = clean_user_prompt(&messages[last_user_idx].text);
    if user_text.is_empty() {
        return None;
    }

    let mut ai_chunks = Vec::new();
    let mut touched = Vec::new();
    let mut seen_files = HashSet::new();

    for msg in &messages[last_user_idx + 1..] {
        if matches!(msg.role, zap_kernel::message::Role::Assistant) {
            let t = msg.text.trim();
            if !t.is_empty() {
                ai_chunks.push(t.to_string());
            }
            for call in &msg.tool_calls {
                if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(&call.arguments) {
                    for key in ["path", "file", "target_file", "filePath"] {
                        if let Some(p) = obj.get(key).and_then(|v| v.as_str()) {
                            let clean = p.trim().to_string();
                            if !clean.is_empty() && seen_files.insert(clean.clone()) {
                                touched.push(clean);
                            }
                        }
                    }
                }
            }
        }
    }

    let ai_text = ai_chunks.last().cloned().unwrap_or_default();
    if ai_text.is_empty() && touched.is_empty() {
        return None;
    }
    Some((user_text, ai_text, touched))
}

static RECENT_CAPTURES: OnceLock<Mutex<HashMap<String, std::time::Instant>>> = OnceLock::new();

/// 从回合结束的权威快照自动提炼并落盘记忆（带 10 秒防重复触发去重）。
pub fn maybe_capture_turn_from_snapshot(
    session_id: &str,
    working_dir: &Path,
    messages: &[Message],
    provider_name: &str,
) {
    let sid = session_id.trim();
    if sid.is_empty() {
        return;
    }
    let Some((user_text, ai_text, touched_files)) = extract_last_turn_payload(messages) else {
        return;
    };
    let dedup_key = format!(
        "{sid}::{}::{}",
        user_text.chars().take(64).collect::<String>(),
        ai_text.chars().take(64).collect::<String>()
    );
    {
        let map = RECENT_CAPTURES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        guard.retain(|_, ts| now.duration_since(*ts).as_secs() < 15);
        if guard.contains_key(&dedup_key) {
            return;
        }
        guard.insert(dedup_key, now);
    }
    let config = zap_config::ConfigStore::default_store()
        .read()
        .map(|s| s.config)
        .unwrap_or_else(|_| Config::load(&Config::default_path()).unwrap_or_default());
    spawn_turn_capture(
        sid.to_string(),
        working_dir.to_path_buf(),
        user_text,
        ai_text,
        touched_files,
        config,
        provider_name.to_string(),
    );
}

/// 在后台独立任务中记录 L3 原文并提炼 L1/L2，不阻塞聊天主流程。
pub fn spawn_turn_capture(
    session_id: String,
    working_dir: PathBuf,
    user_text: String,
    ai_text: String,
    touched_files: Vec<String>,
    config: Config,
    provider_name: String,
) {
    tokio::spawn(async move {
        let capture_info = with_store(|store| {
            let cfg = get_config_conn(&store.conn);
            if !cfg.timeline_auto_capture {
                return Ok(None);
            }
            let Some(agent) = resolve_agent_for_session_conn(&store.conn, &session_id) else {
                return Ok(None);
            };
            let turn_id = new_id("turn");
            let ts = now_ts_minute();
            let project = normalize_project_name(&working_dir);
            let wd_str = working_dir.to_string_lossy().into_owned();

            // 1. 先落盘 L3 完整原文（毫秒级，绝对不丢）
            let raw_turn = RawTurn {
                turn_id: turn_id.clone(),
                ts: ts.clone(),
                agent: agent.id.clone(),
                project: project.clone(),
                working_dir: wd_str.clone(),
                session_id: session_id.clone(),
                user_text: user_text.clone(),
                ai_text: ai_text.clone(),
                touched_files: touched_files.clone(),
            };
            upsert_raw_turn_conn(&store.conn, &raw_turn)?;

            // 2. 防垃圾门禁：寒暄/短确认只留 L3，不建 L1/L2 节点和时间线
            if is_trivial_turn(&user_text, &ai_text, &touched_files) {
                tracing::debug!(turn_id = %turn_id, "neural_memory: skipped trivial turn L1/L2 distillation");
                return Ok(None);
            }

            let open_tasks = load_tasks_conn(&store.conn, Some(&agent.id))?
                .into_iter()
                .filter(|t| t.status != "done")
                .take(10)
                .collect::<Vec<_>>();
            let profiles = load_profiles_conn(&store.conn, Some(&agent.id))?
                .get(&agent.id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(15)
                .collect::<Vec<_>>();

            Ok(Some((
                cfg, agent, turn_id, ts, project, wd_str, open_tasks, profiles,
            )))
        });

        let Ok(Some((cfg, agent, turn_id, ts, project, wd_str, open_tasks, profiles))) =
            capture_info
        else {
            return;
        };

        // 3. 若配置为纯本地静默规则模式，直接走规则提炼
        if cfg.distill_mode == "auto_silent" {
            let _ = with_store(|store| {
                apply_fallback_distill_conn(
                    &store.conn,
                    &agent.id,
                    &session_id,
                    &turn_id,
                    &ts,
                    &project,
                    &wd_str,
                    &user_text,
                    &ai_text,
                    &touched_files,
                    "rule",
                )
            });
            return;
        }

        // 4. 后台直连 LlmProvider 提炼（绝不经过 /v1/chat/completions，杜绝临时幽灵会话）
        let distill_res = run_llm_distill(
            &config,
            &provider_name,
            &cfg.distill_model,
            &working_dir,
            &agent,
            &project,
            &open_tasks,
            &profiles,
            &user_text,
            &ai_text,
            &touched_files,
        )
        .await;

        match distill_res {
            Some(parsed) => {
                if !parsed.should_record {
                    tracing::debug!(turn_id = %turn_id, "neural_memory: LLM marked turn as should_record=false");
                    return;
                }
                let _ = with_store(|store| {
                    apply_llm_distill_conn(
                        &store.conn,
                        &agent.id,
                        &session_id,
                        &turn_id,
                        &ts,
                        &project,
                        &wd_str,
                        &user_text,
                        &ai_text,
                        &touched_files,
                        parsed,
                    )
                });
            }
            None => {
                // 提炼超时或异常时降级写入轻量目录索引（受防垃圾门禁保护）
                let _ = with_store(|store| {
                    apply_fallback_distill_conn(
                        &store.conn,
                        &agent.id,
                        &session_id,
                        &turn_id,
                        &ts,
                        &project,
                        &wd_str,
                        &user_text,
                        &ai_text,
                        &touched_files,
                        "fallback",
                    )
                });
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
async fn run_llm_distill(
    config: &Config,
    provider_name: &str,
    distill_model: &str,
    working_dir: &Path,
    agent: &NeuralAgent,
    project: &str,
    open_tasks: &[NeuralTask],
    profiles: &[ProfileFact],
    user_text: &str,
    ai_text: &str,
    touched_files: &[String],
) -> Option<DistillOutput> {
    let custom_model = distill_model.trim();
    let mut runtime =
        if !custom_model.is_empty() && config.resolve_model(Some(custom_model)).is_ok() {
            zap_coding::CodingRuntimeConfig::from_config(
                config,
                working_dir,
                Some(custom_model),
                None,
                false,
                false,
            )
        } else {
            let mut rt = zap_coding::CodingRuntimeConfig::from_config(
                config,
                working_dir,
                Some(provider_name),
                None,
                false,
                false,
            );
            if !custom_model.is_empty() {
                rt.model = custom_model.to_string();
            }
            rt
        };
    runtime.mcp = false;
    runtime.max_tokens = Some(1200);

    let coding_cfg = crate::kernel_runtime::coding_config_from_runtime(&runtime);
    let options = ChatOptions {
        max_tokens: Some(1200),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory = crate::runtime_host::coding_provider_factory();
    let ephemeral_sid = format!("neural-distill-{}", uuid::Uuid::new_v4());
    let provider =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&ephemeral_sid)))
            .await
            .ok()?
            .ok()?;

    let tasks_desc = if open_tasks.is_empty() {
        "无".to_string()
    } else {
        open_tasks
            .iter()
            .map(|t| format!("id={}, title={}, status={}", t.id, t.title, t.status))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let profiles_desc = if profiles.is_empty() {
        "无".to_string()
    } else {
        profiles
            .iter()
            .map(|p| format!("- {}{}", if p.pinned { "[置顶] " } else { "" }, p.text))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let user_excerpt: String = user_text.chars().take(1500).collect();
    let ai_excerpt: String = ai_text.chars().take(2000).collect();
    let files_str = if touched_files.is_empty() {
        "无".to_string()
    } else {
        touched_files.join(", ")
    };

    let prompt = format!(
        "你是记忆提炼引擎。请严格输出纯 JSON 对象（禁止任何 Markdown 围栏或多余说明）。\n\
        当前智能体: {} ({})\n当前项目: {}\n\n\
        已有未完成任务:\n{}\n\n\
        已有核心档案:\n{}\n\n\
        本轮用户提问:\n{}\n\n\
        本轮AI回复摘要:\n{}\n\n\
        触及文件: {}\n\n\
        请输出如下结构的 JSON:\n\
        {{\n\
          \"should_record\": true/false (纯寒暄、无实质进展或纯确认词必须返回 false),\n\
          \"profile_updates\": [{{\"text\": \"长期偏好/规范/约定\", \"pin\": true/false}}],\n\
          \"profile_removes\": [\"被明确推翻的旧档案关键字\"],\n\
          \"task_actions\": [{{\"action\": \"create|update|complete\", \"task_id\": \"匹配已有任务id，新建可空\", \"title\": \"简短任务名\", \"progress_note\": \"已完成部分\", \"remaining_note\": \"未完成部分\"}}],\n\
          \"node_actions\": [{{\"name\": \"核心知识点/模块名(4~24字)\", \"summary\": \"精炼总结(30~100字)\", \"mastery\": 60~95, \"status\": \"active|review|solid\"}}],\n\
          \"timeline\": {{\"user_goal\": \"一句话问题意图\", \"summary\": \"核心摘要\", \"ai_outcome\": \"实际结果\", \"keywords\": [\"关键词\", \"同义词或英文缩写如DM8\"]}}\n\
        }}",
        agent.label,
        agent.id,
        if project.is_empty() { "全局" } else { project },
        tasks_desc,
        profiles_desc,
        user_excerpt,
        ai_excerpt,
        files_str
    );

    let fut = async move {
        let msgs = [
            Message::system("You are a precise JSON-only memory distillation engine."),
            Message::user(prompt),
        ];
        let mut stream = provider.chat_stream(&msgs, &[], &options).await.ok()?;
        let mut raw = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(delta) => raw.push_str(&delta),
                StreamEvent::Done { .. } | StreamEvent::Error(_) => break,
                _ => {}
            }
        }
        parse_distill_json(&raw)
    };

    tokio::time::timeout(DISTILL_TIMEOUT, fut)
        .await
        .ok()
        .flatten()
}

#[allow(clippy::too_many_arguments)]
fn apply_llm_distill_conn(
    conn: &Connection,
    agent_id: &str,
    session_id: &str,
    turn_id: &str,
    ts: &str,
    project: &str,
    working_dir: &str,
    user_text: &str,
    ai_text: &str,
    touched_files: &[String],
    out: DistillOutput,
) -> Result<(), String> {
    // 1. 移除被推翻的旧档案
    for rem in &out.profile_removes {
        let r = rem.trim();
        if r.chars().count() >= 2 {
            conn.execute(
                "DELETE FROM profiles WHERE agent_id = ?1 AND text LIKE ?2",
                params![agent_id, format!("%{r}%")],
            )
            .ok();
        }
    }

    // 2. 更新/追加核心档案（支持 pin 置顶）
    let existing_profiles = load_profiles_conn(conn, Some(agent_id))?
        .get(agent_id)
        .cloned()
        .unwrap_or_default();
    for item in out.profile_updates {
        let (text, pin) = match item {
            DistillProfileInput::Obj(o) => (o.text.trim().to_string(), o.pin),
            DistillProfileInput::Str(s) => (s.trim().to_string(), false),
        };
        if text.is_empty() {
            continue;
        }
        if existing_profiles.iter().any(|p| p.text == text) {
            continue;
        }
        upsert_profile_conn(
            conn,
            agent_id,
            &ProfileFact {
                id: new_id("pf"),
                text,
                project: project.to_string(),
                pinned: pin,
                updated_at: ts.to_string(),
                source_turn_id: turn_id.to_string(),
            },
        )?;
    }

    // 3. 任务状态推进（优先精确 task_id，禁止随意子串误销账）
    let mut all_tasks = load_tasks_conn(conn, Some(agent_id))?;
    for act in out.task_actions {
        let title = act.title.trim();
        let tid = act.task_id.trim();
        let matched_idx = if !tid.is_empty() {
            all_tasks.iter().position(|t| t.id == tid)
        } else if !title.is_empty() {
            all_tasks
                .iter()
                .position(|t| t.status != "done" && t.title.eq_ignore_ascii_case(title))
        } else {
            None
        };

        match act.action.as_str() {
            "complete" => {
                if let Some(idx) = matched_idx {
                    let mut task = all_tasks[idx].clone();
                    task.status = "done".into();
                    if !act.progress_note.trim().is_empty() {
                        task.progress_note = act.progress_note.trim().to_string();
                    }
                    task.remaining_note.clear();
                    task.done_at = ts.to_string();
                    task.updated_at = ts.to_string();
                    if !task.source_turn_ids.contains(&turn_id.to_string()) {
                        task.source_turn_ids.push(turn_id.to_string());
                    }
                    upsert_task_conn(conn, &task)?;
                    all_tasks[idx] = task;
                }
            }
            "create" | "update" => {
                if let Some(idx) = matched_idx {
                    let mut task = all_tasks[idx].clone();
                    task.status = "in_progress".into();
                    if !act.progress_note.trim().is_empty() {
                        task.progress_note = act.progress_note.trim().to_string();
                    }
                    if !act.remaining_note.trim().is_empty() {
                        task.remaining_note = act.remaining_note.trim().to_string();
                    }
                    for f in touched_files {
                        if !task.touched_files.contains(f) {
                            task.touched_files.push(f.clone());
                        }
                    }
                    if !task.source_turn_ids.contains(&turn_id.to_string()) {
                        task.source_turn_ids.push(turn_id.to_string());
                    }
                    task.updated_at = ts.to_string();
                    upsert_task_conn(conn, &task)?;
                    all_tasks[idx] = task;
                } else if !title.is_empty() {
                    let task = NeuralTask {
                        id: new_id("task"),
                        agent: agent_id.to_string(),
                        project: project.to_string(),
                        working_dir: working_dir.to_string(),
                        title: title.to_string(),
                        progress_note: act.progress_note.trim().to_string(),
                        remaining_note: act.remaining_note.trim().to_string(),
                        touched_files: touched_files.to_vec(),
                        session_id: session_id.to_string(),
                        source_turn_ids: vec![turn_id.to_string()],
                        status: "in_progress".into(),
                        created_at: ts.to_string(),
                        updated_at: ts.to_string(),
                        done_at: String::new(),
                    };
                    upsert_task_conn(conn, &task)?;
                    all_tasks.push(task);
                }
            }
            _ => {}
        }
    }

    // 4. 更新知识节点并按同会话/最近节点自动连线
    let existing_nodes = load_nodes_conn(conn, Some(agent_id))?;
    let mut created_node_ids = Vec::new();
    for na in out.node_actions {
        let name = na.name.trim();
        if is_noise_node_name(name) {
            continue;
        }
        let existing = existing_nodes
            .iter()
            .find(|n| n.name.eq_ignore_ascii_case(name));
        let mut node = match existing {
            Some(old) => old.clone(),
            None => NeuralNode {
                id: new_id(agent_id),
                name: name.to_string(),
                cat: agent_id.to_string(),
                project: project.to_string(),
                status: na.status.clone(),
                mastery: na.mastery.clamp(20, 100),
                tier: 0,
                windows: default_windows(),
                summary: String::new(),
                events: Vec::new(),
                updated_at: ts.to_string(),
            },
        };
        if !na.summary.trim().is_empty() {
            node.summary = na.summary.trim().to_string();
        }
        node.status = if na.status.is_empty() {
            "active".into()
        } else {
            na.status
        };
        node.mastery = na.mastery.clamp(20, 100);
        node.updated_at = ts.to_string();
        node.events.insert(
            0,
            NodeEvent {
                ts: ts.to_string(),
                text: node.summary.clone(),
            },
        );
        node.events.truncate(12);
        upsert_node_conn(conn, &node)?;
        created_node_ids.push(node.id);
    }

    if let Some(first_new) = created_node_ids.first() {
        if let Some(prev) = existing_nodes.first() {
            if prev.id != *first_new {
                conn.execute(
                    "INSERT OR IGNORE INTO links (source, target) VALUES (?1, ?2)",
                    params![prev.id, first_new],
                )
                .ok();
            }
        }
    }

    // 5. 写入时间线索引
    if let Some(tl) = out.timeline {
        let sum = tl.summary.trim();
        let goal = tl.user_goal.trim();
        if !sum.is_empty() || !goal.is_empty() {
            insert_timeline_conn(
                conn,
                &TimelineEvent {
                    id: new_id("tl"),
                    ts: ts.to_string(),
                    agent: agent_id.to_string(),
                    project: project.to_string(),
                    working_dir: working_dir.to_string(),
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    raw_file: "sqlite".into(),
                    user_goal: if goal.is_empty() {
                        user_text.chars().take(60).collect()
                    } else {
                        goal.to_string()
                    },
                    summary: if sum.is_empty() {
                        ai_text.chars().take(120).collect()
                    } else {
                        sum.to_string()
                    },
                    ai_outcome: tl.ai_outcome.trim().to_string(),
                    keywords: tl.keywords,
                    touched_files: touched_files.to_vec(),
                    distilled_by: "llm".into(),
                },
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_fallback_distill_conn(
    conn: &Connection,
    agent_id: &str,
    session_id: &str,
    turn_id: &str,
    ts: &str,
    project: &str,
    working_dir: &str,
    user_text: &str,
    ai_text: &str,
    touched_files: &[String],
    mode: &str,
) -> Result<(), String> {
    let goal: String = user_text
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .chars()
        .take(60)
        .collect();
    if is_noise_node_name(&goal) {
        return Ok(());
    }
    let outcome: String = ai_text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
        .chars()
        .take(120)
        .collect();

    insert_timeline_conn(
        conn,
        &TimelineEvent {
            id: new_id("tl"),
            ts: ts.to_string(),
            agent: agent_id.to_string(),
            project: project.to_string(),
            working_dir: working_dir.to_string(),
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            raw_file: "sqlite".into(),
            user_goal: goal.clone(),
            summary: goal.clone(),
            ai_outcome: outcome,
            keywords: Vec::new(),
            touched_files: touched_files.to_vec(),
            distilled_by: mode.to_string(),
        },
    )?;

    let existing_nodes = load_nodes_conn(conn, Some(agent_id))?;
    let short_name: String = goal.chars().take(24).collect();
    let node_id = new_id(agent_id);
    upsert_node_conn(
        conn,
        &NeuralNode {
            id: node_id.clone(),
            name: short_name,
            cat: agent_id.to_string(),
            project: project.to_string(),
            status: "active".into(),
            mastery: 72,
            tier: 0,
            windows: default_windows(),
            summary: format!("记录于 {ts}：{goal}"),
            events: vec![NodeEvent {
                ts: ts.to_string(),
                text: format!("对话沉淀：{goal}"),
            }],
            updated_at: ts.to_string(),
        },
    )?;
    if let Some(prev) = existing_nodes.first() {
        conn.execute(
            "INSERT OR IGNORE INTO links (source, target) VALUES (?1, ?2)",
            params![prev.id, node_id],
        )
        .ok();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 给前端插件提供的 RPC 分发器（/api/plugins-rpc/neural-agents）
// ---------------------------------------------------------------------------

pub fn handle_rpc(action: &str, payload: &Value) -> Result<Value, String> {
    match action {
        "neural:bootstrap" => with_store(|store| build_bootstrap_payload(store)),

        "neural:set-session-agent" => {
            let sid = payload
                .get("sessionId")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let aid = payload
                .get("agentId")
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            with_store(|store| {
                set_session_agent_conn(&store.conn, sid, aid)?;
                Ok(json!({ "success": true, "sessionId": sid, "agentId": aid }))
            })
        }

        "neural:update-config" => with_store(|store| {
            let mut cfg = get_config_conn(&store.conn);
            if let Some(v) = payload.get("activeAgent").and_then(|v| v.as_str()) {
                cfg.active_agent = v.to_string();
            }
            if let Some(v) = payload.get("distillMode").and_then(|v| v.as_str()) {
                cfg.distill_mode = v.to_string();
            }
            if let Some(v) = payload.get("distillModel").and_then(|v| v.as_str()) {
                cfg.distill_model = v.to_string();
            }
            if let Some(v) = payload.get("timelineAutoCapture").and_then(|v| v.as_bool()) {
                cfg.timeline_auto_capture = v;
            }
            if let Some(v) = payload.get("showFloatingCapsule").and_then(|v| v.as_bool()) {
                cfg.show_floating_capsule = v;
            }
            save_config_conn(&store.conn, &cfg)?;
            Ok(json!({ "success": true, "config": cfg }))
        }),

        "neural:save-agents" => {
            let agents_val = payload.get("agents").cloned().unwrap_or(Value::Null);
            let agents: Vec<NeuralAgent> =
                serde_json::from_value(agents_val).map_err(|e| format!("智能体格式错误: {e}"))?;
            with_store(|store| {
                store
                    .conn
                    .execute("DELETE FROM agents", [])
                    .map_err(|e| e.to_string())?;
                for (idx, a) in agents.iter().enumerate() {
                    upsert_agent_conn(&store.conn, a, idx as i64)?;
                }
                Ok(json!({ "success": true, "agents": load_agents_conn(&store.conn)? }))
            })
        }

        "neural:reset-agents" => with_store(|store| {
            store
                .conn
                .execute("DELETE FROM agents", [])
                .map_err(|e| e.to_string())?;
            for (idx, a) in default_agents().iter().enumerate() {
                upsert_agent_conn(&store.conn, a, idx as i64)?;
            }
            Ok(json!({ "success": true, "agents": load_agents_conn(&store.conn)? }))
        }),

        "neural:save-graph" => {
            let nodes: Vec<NeuralNode> =
                serde_json::from_value(payload.get("nodes").cloned().unwrap_or_else(|| json!([])))
                    .map_err(|e| format!("节点数据格式错误: {e}"))?;
            let links: Vec<NeuralLink> =
                serde_json::from_value(payload.get("links").cloned().unwrap_or_else(|| json!([])))
                    .map_err(|e| format!("连线数据格式错误: {e}"))?;
            with_store(|store| {
                store
                    .conn
                    .execute("DELETE FROM nodes", [])
                    .map_err(|e| e.to_string())?;
                store
                    .conn
                    .execute("DELETE FROM nodes_fts", [])
                    .map_err(|e| e.to_string())?;
                store
                    .conn
                    .execute("DELETE FROM links", [])
                    .map_err(|e| e.to_string())?;
                for n in &nodes {
                    upsert_node_conn(&store.conn, n)?;
                }
                for l in &links {
                    store
                        .conn
                        .execute(
                            "INSERT OR IGNORE INTO links (source, target) VALUES (?1, ?2)",
                            params![l.source, l.target],
                        )
                        .ok();
                }
                Ok(json!({ "success": true }))
            })
        }

        "neural:delete-profile" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let agent = payload.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                store
                    .conn
                    .execute(
                        "DELETE FROM profiles WHERE id = ?1 AND (?2 = '' OR agent_id = ?2)",
                        params![id, agent],
                    )
                    .map_err(|e| e.to_string())?;
                build_bootstrap_payload(store)
            })
        }

        "neural:task-action" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let op = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                if op == "delete" {
                    store
                        .conn
                        .execute("DELETE FROM tasks WHERE id = ?1", params![id])
                        .map_err(|e| e.to_string())?;
                } else if op == "done" {
                    store
                        .conn
                        .execute(
                            "UPDATE tasks SET status = 'done', done_at = ?1, updated_at = ?1, remaining_note = '' WHERE id = ?2",
                            params![now_ts_minute(), id],
                        )
                        .map_err(|e| e.to_string())?;
                }
                build_bootstrap_payload(store)
            })
        }

        "neural:delete-timeline" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let ts = payload.get("ts").and_then(|v| v.as_str()).unwrap_or("");
            let agent = payload.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                if !id.is_empty() {
                    store
                        .conn
                        .execute("DELETE FROM timeline WHERE id = ?1", params![id])
                        .map_err(|e| e.to_string())?;
                    store
                        .conn
                        .execute("DELETE FROM timeline_fts WHERE event_id = ?1", params![id])
                        .ok();
                } else if !ts.is_empty() {
                    store
                        .conn
                        .execute(
                            "DELETE FROM timeline WHERE ts = ?1 AND (?2 = '' OR agent_id = ?2)",
                            params![ts, agent],
                        )
                        .map_err(|e| e.to_string())?;
                }
                build_bootstrap_payload(store)
            })
        }

        "neural:clear-memory" => {
            let agent = payload
                .get("agent")
                .and_then(|v| v.as_str())
                .unwrap_or("all");
            with_store(|store| {
                if agent == "all" || agent.is_empty() {
                    store.conn.execute_batch(
                        "DELETE FROM profiles; DELETE FROM tasks; DELETE FROM nodes; DELETE FROM nodes_fts;
                         DELETE FROM links; DELETE FROM timeline; DELETE FROM timeline_fts;
                         DELETE FROM raw_turns; DELETE FROM raw_turns_fts;",
                    ).map_err(|e| e.to_string())?;
                } else {
                    store
                        .conn
                        .execute("DELETE FROM profiles WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM tasks WHERE agent_id = ?1", params![agent])
                        .ok();
                    store.conn.execute(
                        "DELETE FROM links WHERE source IN (SELECT id FROM nodes WHERE agent_id = ?1)
                            OR target IN (SELECT id FROM nodes WHERE agent_id = ?1)",
                        params![agent],
                    ).ok();
                    store
                        .conn
                        .execute("DELETE FROM nodes WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM nodes_fts WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM timeline WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM timeline_fts WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM raw_turns WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM raw_turns_fts WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                }
                build_bootstrap_payload(store)
            })
        }

        "neural:request-raw" => {
            let turn_id = payload.get("turnId").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                let raw = load_raw_turn_conn(&store.conn, turn_id)?;
                Ok(json!({ "success": true, "raw": raw }))
            })
        }

        "neural:import-legacy" => {
            let dir = payload.get("dir").and_then(|v| v.as_str()).unwrap_or("");
            if dir.trim().is_empty() {
                return Err("请提供有效的旧版数据目录路径".into());
            }
            with_store(|store| {
                let stats = import_legacy_dir_conn(&mut store.conn, Path::new(dir.trim()))?;
                let mut boot = build_bootstrap_payload(store)?;
                boot["importStats"] = stats;
                Ok(boot)
            })
        }

        "neural:set-data-dir" => {
            let new_dir_str = payload
                .get("dir")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            let target_dir = if new_dir_str.is_empty() {
                default_data_dir()
            } else {
                PathBuf::from(new_dir_str)
            };
            with_store(|store| {
                if target_dir == store.data_dir {
                    return build_bootstrap_payload(store);
                }
                std::fs::create_dir_all(&target_dir)
                    .map_err(|e| format!("无法创建目标目录: {e}"))?;
                let target_db = target_dir.join("neural_memory.db");
                let old_db = store.data_dir.join("neural_memory.db");

                // 如果目标目录尚无 sqlite，先执行 WAL checkpoint 并拷贝当前库过去
                if !target_db.exists() && old_db.exists() {
                    let _ = store.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                    std::fs::copy(&old_db, &target_db)
                        .map_err(|e| format!("复制数据库到新目录失败: {e}"))?;
                }
                let mut new_conn = open_connection(&target_dir)?;
                bootstrap_initial_data(&mut new_conn, &target_dir);
                let mut cfg = get_config_conn(&new_conn);
                cfg.memory_data_dir = new_dir_str.to_string();
                save_config_conn(&new_conn, &cfg)?;
                save_data_dir_pointer(&target_dir)?;

                store.conn = new_conn;
                store.data_dir = target_dir;
                build_bootstrap_payload(store)
            })
        }

        "neural:export-backup" => with_store(|store| {
            let mut boot = build_bootstrap_payload(store)?;
            let raw_turns: Vec<RawTurn> = {
                let mut stmt = store
                    .conn
                    .prepare(
                        "SELECT turn_id, ts, agent_id, project, working_dir, session_id, user_text, ai_text, touched_files_json
                         FROM raw_turns ORDER BY ts DESC",
                    )
                    .map_err(|e| e.to_string())?;
                let rows = stmt
                    .query_map([], |row| {
                        let tf_str: String = row.get(8)?;
                        Ok(RawTurn {
                            turn_id: row.get(0)?,
                            ts: row.get(1)?,
                            agent: row.get(2)?,
                            project: row.get(3)?,
                            working_dir: row.get(4)?,
                            session_id: row.get(5)?,
                            user_text: row.get(6)?,
                            ai_text: row.get(7)?,
                            touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
                        })
                    })
                    .map_err(|e| e.to_string())?;
                rows.flatten().collect()
            };
            boot["rawTurns"] = serde_json::to_value(raw_turns).unwrap_or_else(|_| json!([]));
            Ok(boot)
        }),

        _ => Err(format!("未知 neural_memory RPC action: {action}")),
    }
}

fn build_bootstrap_payload(store: &StoreInner) -> Result<Value, String> {
    let cfg = get_config_conn(&store.conn);
    let agents = load_agents_conn(&store.conn)?;
    let profiles = load_profiles_conn(&store.conn, None)?;
    let tasks = load_tasks_conn(&store.conn, None)?;
    let nodes = load_nodes_conn(&store.conn, None)?;
    let links = load_links_conn(&store.conn)?;
    let timeline = load_timeline_conn(&store.conn, None, 200)?;

    let mut session_map: HashMap<String, String> = HashMap::new();
    if let Ok(mut stmt) = store
        .conn
        .prepare("SELECT session_id, agent_id FROM session_agents")
    {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        {
            for r in rows.flatten() {
                session_map.insert(r.0, r.1);
            }
        }
    }

    let raw_count: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM raw_turns", [], |r| r.get(0))
        .unwrap_or(0);

    Ok(json!({
        "success": true,
        "dataDir": store.data_dir.to_string_lossy(),
        "dbPath": store.data_dir.join("neural_memory.db").to_string_lossy(),
        "config": cfg,
        "agents": agents,
        "sessionMap": session_map,
        "profiles": profiles,
        "tasks": tasks,
        "nodes": nodes,
        "links": links,
        "timeline": timeline,
        "rawTurnCount": raw_count,
    }))
}
