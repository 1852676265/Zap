//! SQLite 存储层（WAL 读写分离 + 内存热缓存 + 预编译缓存 + 5 抽屉/路牌/草稿队列 + 留存淘汰）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::redact::redact_raw_turn;
#[cfg(test)]
use super::schema::init_schema;
use super::schema::{open_connection, open_read_connection};
use super::search::{build_fts_column_text, extract_storage_anchors, is_noise_node_name};
use super::types::{
    default_active_status, default_journal_kind, default_link_relation, default_profile_kind,
    default_task_item_kind, default_windows, new_id, normalize_project_key_str, now_iso,
    now_ts_minute, DraftTurn, MemoryAnchor, NeuralAgent, NeuralConfig, NeuralLink, NeuralNode,
    NeuralTask, NodeFact, ProfileFact, RawTurn, TimelineEvent, MAX_NODES_PER_AGENT,
    MAX_NON_PINNED_PROFILES, MAX_RAW_TURNS_PER_AGENT, MAX_TIMELINE_PER_AGENT,
};

pub(crate) struct StoreInner {
    pub(crate) data_dir: PathBuf,
    pub(crate) conn: Connection,
}

struct WriteStoreState {
    inner: Option<StoreInner>,
    generation: u64,
}

static WRITE_STORE: OnceLock<Mutex<WriteStoreState>> = OnceLock::new();

const MAX_READ_POOL_SIZE: usize = 4;

struct ReadPoolState {
    data_dir: PathBuf,
    generation: u64,
    conns: Vec<Connection>,
}

static READ_POOL: OnceLock<Mutex<ReadPoolState>> = OnceLock::new();

#[derive(Default)]
struct HotMetaCache {
    config: Option<NeuralConfig>,
    agents: Option<Vec<NeuralAgent>>,
    session_agents: HashMap<String, String>,
    known_projects: HashMap<String, Vec<KnownProject>>,
    thin_directories: HashMap<(String, String, String), String>,
}

static HOT_CACHE: OnceLock<RwLock<HotMetaCache>> = OnceLock::new();

fn write_store_mutex() -> &'static Mutex<WriteStoreState> {
    WRITE_STORE.get_or_init(|| {
        Mutex::new(WriteStoreState {
            inner: None,
            generation: 1,
        })
    })
}

fn read_pool_mutex() -> &'static Mutex<ReadPoolState> {
    READ_POOL.get_or_init(|| {
        Mutex::new(ReadPoolState {
            data_dir: PathBuf::new(),
            generation: 0,
            conns: Vec::new(),
        })
    })
}

fn hot_cache_lock() -> &'static RwLock<HotMetaCache> {
    HOT_CACHE.get_or_init(|| RwLock::new(HotMetaCache::default()))
}

pub(crate) fn invalidate_hot_config() {
    if let Ok(mut g) = hot_cache_lock().write() {
        g.config = None;
    }
}

pub(crate) fn invalidate_hot_agents() {
    if let Ok(mut g) = hot_cache_lock().write() {
        g.agents = None;
    }
}

pub(crate) fn invalidate_hot_session(session_id: &str) {
    let sid = session_id.trim();
    if let Ok(mut g) = hot_cache_lock().write() {
        if sid.is_empty() {
            g.session_agents.clear();
        } else {
            g.session_agents.remove(sid);
        }
    }
}

pub(crate) fn invalidate_hot_content(agent_id: Option<&str>) {
    if let Ok(mut g) = hot_cache_lock().write() {
        match agent_id {
            Some(aid) => {
                g.known_projects.remove(aid);
                g.thin_directories.retain(|(a, _, _), _| a != aid);
            }
            None => {
                g.known_projects.clear();
                g.thin_directories.clear();
            }
        }
    }
}

pub(crate) fn invalidate_all_hot_caches() {
    if let Ok(mut g) = hot_cache_lock().write() {
        *g = HotMetaCache::default();
    }
}

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

pub(crate) fn default_data_dir() -> PathBuf {
    zap_config::Config::config_dir().join("neural-memory")
}

#[cfg(not(test))]
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

#[cfg(test)]
fn resolve_configured_data_dir() -> PathBuf {
    static TEST_DIR: OnceLock<PathBuf> = OnceLock::new();
    TEST_DIR
        .get_or_init(|| {
            let dir =
                std::env::temp_dir().join(format!("zap-neural-memory-test-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            dir
        })
        .clone()
}

pub(crate) fn save_data_dir_pointer(dir: &Path) -> Result<(), String> {
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

fn ensure_write_store_initialized(guard: &mut WriteStoreState) -> Result<(PathBuf, u64), String> {
    if guard.inner.is_none() {
        let dir = resolve_configured_data_dir();
        let mut conn = open_connection(&dir)?;
        bootstrap_initial_data(&mut conn, &dir);
        guard.inner = Some(StoreInner {
            data_dir: dir.clone(),
            conn,
        });
        invalidate_all_hot_caches();
    }
    let inner = guard.inner.as_ref().expect("store initialized above");
    Ok((inner.data_dir.clone(), guard.generation))
}

/// 获取全局 SQLite 写连接（序列化所有写入与事务，保证原子性）。
pub(crate) fn with_write<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut StoreInner) -> Result<R, String>,
{
    let mut guard = write_store_mutex()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let _ = ensure_write_store_initialized(&mut guard)?;
    let inner = guard.inner.as_mut().expect("store initialized above");
    f(inner)
}

/// 切换全局数据目录后递增代次、排空旧只读连接池并清空热缓存。
pub(crate) fn notify_store_dir_switched(new_dir: &Path) {
    let new_gen = {
        let mut wg = write_store_mutex()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        wg.generation = wg.generation.saturating_add(1);
        wg.generation
    };
    {
        let mut rp = read_pool_mutex().lock().unwrap_or_else(|e| e.into_inner());
        rp.conns.clear();
        rp.data_dir = new_dir.to_path_buf();
        rp.generation = new_gen;
    }
    invalidate_all_hot_caches();
}

struct PooledReadConn {
    conn: Option<Connection>,
    data_dir: PathBuf,
    generation: u64,
}

impl Drop for PooledReadConn {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            if let Ok(mut pool) = read_pool_mutex().lock() {
                if pool.generation == self.generation
                    && pool.data_dir == self.data_dir
                    && pool.conns.len() < MAX_READ_POOL_SIZE
                {
                    pool.conns.push(conn);
                }
            }
        }
    }
}

/// 获取独立只读 SQLite 连接（真正释放 WAL 读写并发，后台提炼写库时前台 0 阻塞）。
pub(crate) fn with_read<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&Connection, &Path) -> Result<R, String>,
{
    // 快速路径：若只读连接池已就绪，完全不碰写锁直接取连接
    let fast_meta = {
        let mut pool = read_pool_mutex().lock().unwrap_or_else(|e| e.into_inner());
        if pool.generation > 0 && !pool.data_dir.as_os_str().is_empty() {
            let conn_opt = pool.conns.pop();
            Some((pool.data_dir.clone(), pool.generation, conn_opt))
        } else {
            None
        }
    };

    let (data_dir, generation, pooled_conn) = match fast_meta {
        Some((dir, gen, Some(conn))) => (dir, gen, conn),
        Some((dir, gen, None)) => {
            let conn = open_read_connection(&dir)?;
            (dir, gen, conn)
        }
        None => {
            // 首次冷启动：先确保主库完成建表/v3 迁移，再初始化只读池元数据
            let (dir, gen) = {
                let mut wg = write_store_mutex()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                ensure_write_store_initialized(&mut wg)?
            };
            {
                let mut pool = read_pool_mutex().lock().unwrap_or_else(|e| e.into_inner());
                if pool.generation != gen || pool.data_dir != dir {
                    pool.conns.clear();
                    pool.data_dir = dir.clone();
                    pool.generation = gen;
                }
            }
            let conn = open_read_connection(&dir)?;
            (dir, gen, conn)
        }
    };

    let guard = PooledReadConn {
        conn: Some(pooled_conn),
        data_dir: data_dir.clone(),
        generation,
    };
    f(guard.conn.as_ref().expect("read conn active"), &data_dir)
}

/// 进程级热缓存读取：当前配置。
pub(crate) fn get_cached_config() -> Result<NeuralConfig, String> {
    if let Ok(g) = hot_cache_lock().read() {
        if let Some(ref cfg) = g.config {
            return Ok(cfg.clone());
        }
    }
    let cfg = with_read(|conn, _| Ok(get_config_conn(conn)))?;
    if let Ok(mut g) = hot_cache_lock().write() {
        g.config = Some(cfg.clone());
    }
    Ok(cfg)
}

/// 进程级热缓存读取：智能体列表。
pub(crate) fn get_cached_agents() -> Result<Vec<NeuralAgent>, String> {
    if let Ok(g) = hot_cache_lock().read() {
        if let Some(ref agents) = g.agents {
            return Ok(agents.clone());
        }
    }
    let agents = with_read(|conn, _| load_agents_conn(conn))?;
    if let Ok(mut g) = hot_cache_lock().write() {
        g.agents = Some(agents.clone());
    }
    Ok(agents)
}

/// 进程级热缓存读取：解析某会话当前绑定的智能体（命中缓存时 0 SQL）。
pub(crate) fn resolve_agent_for_session_cached(
    session_id: &str,
) -> Result<Option<NeuralAgent>, String> {
    let sid = session_id.trim();
    let cfg = get_cached_config()?;
    let mapped = if sid.is_empty() {
        None
    } else {
        let from_cache = hot_cache_lock()
            .read()
            .ok()
            .and_then(|g| g.session_agents.get(sid).cloned());
        match from_cache {
            Some(v) => Some(v),
            None => {
                let db_val: Option<String> = with_read(|conn, _| {
                    let mut stmt = conn
                        .prepare_cached("SELECT agent_id FROM session_agents WHERE session_id = ?1")
                        .map_err(|e| e.to_string())?;
                    stmt.query_row(params![sid], |r| r.get(0))
                        .optional()
                        .map_err(|e| e.to_string())
                })?;
                if let Some(ref aid) = db_val {
                    if let Ok(mut g) = hot_cache_lock().write() {
                        if g.session_agents.len() > 2048 {
                            g.session_agents.clear();
                        }
                        g.session_agents.insert(sid.to_string(), aid.clone());
                    }
                }
                db_val
            }
        }
    };
    let agent_id = mapped.unwrap_or(cfg.active_agent);
    if agent_id.trim().is_empty() || agent_id == "none" {
        return Ok(None);
    }
    let agents = get_cached_agents()?;
    Ok(agents.into_iter().find(|a| a.id == agent_id))
}

/// 进程级热缓存读取：已知项目列表（避免每轮 3 次 SELECT DISTINCT）。
pub(crate) fn list_known_projects_cached(agent_id: &str) -> Result<Vec<KnownProject>, String> {
    if let Ok(g) = hot_cache_lock().read() {
        if let Some(list) = g.known_projects.get(agent_id) {
            return Ok(list.clone());
        }
    }
    let list = with_read(|conn, _| Ok(list_known_projects_conn(conn, agent_id)))?;
    if let Ok(mut g) = hot_cache_lock().write() {
        g.known_projects.insert(agent_id.to_string(), list.clone());
    }
    Ok(list)
}

/// 进程级热缓存读取：极薄“大脑目录”（用于会话开场前缀 0ms 挂载）。
pub(crate) fn get_thin_directory_cached(
    agent_id: &str,
    project_key: &str,
    project_name: &str,
) -> Result<String, String> {
    let key = (
        agent_id.to_string(),
        project_key.to_string(),
        project_name.to_string(),
    );
    if let Ok(g) = hot_cache_lock().read() {
        if let Some(dir) = g.thin_directories.get(&key) {
            return Ok(dir.clone());
        }
    }
    let dir = with_read(|conn, _| {
        build_thin_memory_directory_conn(conn, agent_id, project_key, project_name)
    })?;
    if let Ok(mut g) = hot_cache_lock().write() {
        if g.thin_directories.len() > 256 {
            g.thin_directories.clear();
        }
        g.thin_directories.insert(key, dir.clone());
    }
    Ok(dir)
}

pub(crate) fn default_agents() -> Vec<NeuralAgent> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// 自动迁移旧版数据（移除硬编码外部开发路径）
// ---------------------------------------------------------------------------

fn find_legacy_data_dirs() -> Vec<PathBuf> {
    if cfg!(test) {
        return Vec::new();
    }
    let mut candidates = Vec::new();
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
    let p = zap_config::Config::config_dir()
        .join("plugin-storage")
        .join("_zap_neural-agents")
        .join("storage.json");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

pub(crate) fn bootstrap_initial_data(conn: &mut Connection, _data_dir: &Path) {
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

    for (idx, a) in default_agents().iter().enumerate() {
        let _ = upsert_agent_conn(conn, a, idx as i64);
    }

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

pub(crate) fn import_legacy_dir_conn(conn: &mut Connection, dir: &Path) -> Result<Value, String> {
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

    if let Ok(raw) = std::fs::read_to_string(dir.join("session-map.json")) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&raw) {
            for (sid, aid) in map {
                set_session_agent_conn(conn, &sid, &aid)?;
            }
        }
    }

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

    if let Ok(raw) = std::fs::read_to_string(dir.join("tasks.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<NeuralTask>>(&raw) {
            for mut t in list {
                if t.project_key.is_empty() && !t.working_dir.is_empty() {
                    t.project_key = normalize_project_key_str(&t.working_dir);
                }
                upsert_task_conn(conn, &t)?;
            }
            stats["tasks"] = json!(load_tasks_conn(conn, None).map(|v| v.len()).unwrap_or(0));
        }
    }

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
                if ev.project_key.is_empty() && !ev.working_dir.is_empty() {
                    ev.project_key = normalize_project_key_str(&ev.working_dir);
                }
                insert_timeline_conn(conn, &ev)?;
                count += 1;
            }
            stats["timeline"] = json!(count);
        }
    }

    let mut raw_count = 0usize;
    for sub in ["raw", "archive"] {
        let sub_dir = dir.join(sub);
        if let Ok(entries) = std::fs::read_dir(&sub_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(raw) = std::fs::read_to_string(&p) {
                        if let Ok(turns) = serde_json::from_str::<Vec<RawTurn>>(&raw) {
                            for mut t in turns {
                                if t.project_key.is_empty() && !t.working_dir.is_empty() {
                                    t.project_key = normalize_project_key_str(&t.working_dir);
                                }
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

pub(crate) fn upsert_agent_conn(
    conn: &Connection,
    a: &NeuralAgent,
    sort_order: i64,
) -> Result<(), String> {
    let label = if a.label.trim().is_empty() {
        format!("{} {}", a.icon, a.name)
    } else {
        a.label.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO agents (id, icon, name, label, color, hint, system_prompt, sort_order)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            a.id,
            a.icon,
            a.name,
            label,
            a.color,
            a.hint,
            a.system_prompt,
            sort_order
        ],
    )
    .map_err(|e| e.to_string())?;
    invalidate_hot_agents();
    Ok(())
}

pub(crate) fn load_agents_conn(conn: &Connection) -> Result<Vec<NeuralAgent>, String> {
    let mut stmt = conn
        .prepare_cached(
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

pub(crate) fn has_any_agents_conn(conn: &Connection) -> bool {
    conn.prepare_cached("SELECT 1 FROM agents LIMIT 1")
        .ok()
        .and_then(|mut s| s.query_row([], |_| Ok(true)).optional().ok().flatten())
        .unwrap_or(false)
}

pub(crate) fn get_config_conn(conn: &Connection) -> NeuralConfig {
    let mut cfg = NeuralConfig::default();
    if let Ok(mut stmt) = conn.prepare_cached("SELECT key, value FROM meta") {
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

pub(crate) fn save_config_conn(conn: &Connection, cfg: &NeuralConfig) -> Result<(), String> {
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
    invalidate_hot_config();
    Ok(())
}

pub(crate) fn set_session_agent_conn(
    conn: &Connection,
    session_id: &str,
    agent_id: &str,
) -> Result<(), String> {
    let sid = session_id.trim();
    let aid = agent_id.trim();
    if sid.is_empty() {
        return Ok(());
    }
    let normalized_aid = if aid.is_empty() { "none" } else { aid };
    conn.execute(
        "INSERT OR REPLACE INTO session_agents (session_id, agent_id, updated_at) VALUES (?1, ?2, ?3)",
        params![sid, normalized_aid, now_iso()],
    )
    .map_err(|e| e.to_string())?;
    invalidate_hot_session(sid);
    let mut cfg = get_config_conn(conn);
    if !cfg.active_agent.trim().is_empty() && cfg.active_agent != "none" {
        cfg.active_agent = "none".to_string();
        let _ = save_config_conn(conn, &cfg);
    }
    Ok(())
}

/// 读取某会话当前在 `session_agents` 中显式绑定的智能体 ID（不回退全局草稿）。
pub(crate) fn get_explicit_session_agent_conn(
    conn: &Connection,
    session_id: &str,
) -> Option<String> {
    let sid = session_id.trim();
    if sid.is_empty() {
        return None;
    }
    conn.prepare_cached("SELECT agent_id FROM session_agents WHERE session_id = ?1")
        .ok()?
        .query_row(params![sid], |r| r.get(0))
        .optional()
        .ok()
        .flatten()
}

/// 为新建或草稿会话原子锁定智能体归属（即使当前为 `"none"` 也显式落库），并消费清空全局草稿态 `cfg_active_agent`。
pub(crate) fn bind_fresh_session_agent_if_unset_conn(
    conn: &Connection,
    session_id: &str,
    agent_id: &str,
) -> Result<(), String> {
    let sid = session_id.trim();
    let aid = agent_id.trim();
    if sid.is_empty() {
        return Ok(());
    }
    let normalized_aid = if aid.is_empty() { "none" } else { aid };
    conn.execute(
        "INSERT OR IGNORE INTO session_agents (session_id, agent_id, updated_at) VALUES (?1, ?2, ?3)",
        params![sid, normalized_aid, now_iso()],
    )
    .map_err(|e| e.to_string())?;
    invalidate_hot_session(sid);
    let mut cfg = get_config_conn(conn);
    if !cfg.active_agent.trim().is_empty() && cfg.active_agent != "none" {
        cfg.active_agent = "none".to_string();
        let _ = save_config_conn(conn, &cfg);
    }
    Ok(())
}

pub(crate) fn resolve_agent_for_session_conn(
    conn: &Connection,
    session_id: &str,
) -> Option<NeuralAgent> {
    let cfg = get_config_conn(conn);
    let mapped = get_explicit_session_agent_conn(conn, session_id);
    let agent_id = mapped.unwrap_or_else(|| cfg.active_agent.clone());
    if agent_id.trim().is_empty() || agent_id == "none" {
        return None;
    }
    load_agents_conn(conn)
        .ok()?
        .into_iter()
        .find(|a| a.id == agent_id)
}

// ---------------------------------------------------------------------------
// 精确路牌（memory_anchors）CRUD
// ---------------------------------------------------------------------------

pub(crate) fn replace_anchors_conn(
    conn: &Connection,
    target_kind: &str,
    target_id: &str,
    anchors: &[MemoryAnchor],
) {
    let _ = conn.execute(
        "DELETE FROM memory_anchors WHERE target_kind = ?1 AND target_id = ?2",
        params![target_kind, target_id],
    );
    if anchors.is_empty() {
        return;
    }
    let Ok(mut stmt) = conn.prepare_cached(
        "INSERT OR IGNORE INTO memory_anchors
         (agent_id, project_key, target_kind, target_id, anchor_type, anchor_value, anchor_norm)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    ) else {
        return;
    };
    for a in anchors {
        let _ = stmt.execute(params![
            a.agent_id,
            a.project_key,
            a.target_kind,
            a.target_id,
            a.anchor_type,
            a.anchor_value,
            a.anchor_norm
        ]);
    }
}

/// 按规范化路牌列表秒查命中的目标 ID 及命中路牌个数（B-Tree 索引直达）。
pub(crate) fn lookup_anchor_hits_conn(
    conn: &Connection,
    agent_id: &str,
    anchor_norms: &[String],
    target_kind: &str,
) -> HashMap<String, usize> {
    let mut hits: HashMap<String, usize> = HashMap::new();
    if anchor_norms.is_empty() {
        return hits;
    }
    let Ok(mut stmt) = conn.prepare_cached(
        "SELECT target_id FROM memory_anchors
         WHERE agent_id = ?1 AND anchor_norm = ?2 AND target_kind = ?3",
    ) else {
        return hits;
    };
    for norm in anchor_norms {
        if let Ok(rows) = stmt.query_map(params![agent_id, norm, target_kind], |r| {
            r.get::<_, String>(0)
        }) {
            for tid in rows.flatten() {
                *hits.entry(tid).or_insert(0) += 1;
            }
        }
    }
    hits
}

// ---------------------------------------------------------------------------
// 抽屉 1：长期规矩与习惯（profiles）
// ---------------------------------------------------------------------------

fn map_profile_row(row: &rusqlite::Row<'_>, offset: usize) -> rusqlite::Result<ProfileFact> {
    let pinned_int: i64 = row.get(offset + 4)?;
    let kind: String = row
        .get::<_, String>(offset + 7)
        .unwrap_or_else(|_| default_profile_kind());
    let status: String = row
        .get::<_, String>(offset + 8)
        .unwrap_or_else(|_| default_active_status());
    Ok(ProfileFact {
        id: row.get(offset)?,
        project: row.get(offset + 1)?,
        project_key: row.get(offset + 2)?,
        text: row.get(offset + 3)?,
        pinned: pinned_int != 0,
        updated_at: row.get(offset + 5)?,
        source_turn_id: row.get(offset + 6)?,
        kind: if kind.is_empty() {
            default_profile_kind()
        } else {
            kind
        },
        status: if status.is_empty() {
            default_active_status()
        } else {
            status
        },
        superseded_by: row.get::<_, String>(offset + 9).unwrap_or_default(),
        invalid_at: row.get::<_, String>(offset + 10).unwrap_or_default(),
        hit_count: row.get::<_, i64>(offset + 11).unwrap_or(0),
        last_used_at: row.get::<_, String>(offset + 12).unwrap_or_default(),
    })
}

pub(crate) fn upsert_profile_conn(
    conn: &Connection,
    agent_id: &str,
    f: &ProfileFact,
) -> Result<(), String> {
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
    let kind = if f.kind.trim().is_empty() {
        default_profile_kind()
    } else {
        f.kind.clone()
    };
    let status = if f.status.trim().is_empty() {
        default_active_status()
    } else {
        f.status.clone()
    };

    // 先查旧 rowid，避免 INSERT OR REPLACE 换 rowid 导致旧 FTS 孤儿行
    let old_rowid: Option<i64> = conn
        .prepare_cached("SELECT rowid FROM profiles WHERE id = ?1")
        .ok()
        .and_then(|mut s| {
            s.query_row(params![id], |r| r.get(0))
                .optional()
                .ok()
                .flatten()
        });
    if let Some(orid) = old_rowid {
        let _ = conn.execute("DELETE FROM profiles_fts WHERE rowid = ?1", params![orid]);
    }

    conn.execute(
        "INSERT OR REPLACE INTO profiles (
            id, agent_id, project, project_key, text, pinned, updated_at, source_turn_id,
            kind, status, superseded_by, invalid_at, hit_count, last_used_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            id,
            agent_id,
            f.project,
            f.project_key,
            f.text,
            if f.pinned { 1 } else { 0 },
            updated_at,
            f.source_turn_id,
            kind,
            status,
            f.superseded_by,
            f.invalid_at,
            f.hit_count,
            f.last_used_at,
        ],
    )
    .map_err(|e| e.to_string())?;

    if let Ok(Some(new_rid)) = conn
        .prepare_cached("SELECT rowid FROM profiles WHERE id = ?1")
        .and_then(|mut s| s.query_row(params![id], |r| r.get::<_, i64>(0)).optional())
    {
        let _ = conn.execute(
            "INSERT OR REPLACE INTO profiles_fts (rowid, profile_id, agent_id, text_col)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                new_rid,
                id,
                agent_id,
                build_fts_column_text(&[&f.text, &f.project])
            ],
        );
    }

    let anchors = extract_storage_anchors(
        agent_id,
        &f.project_key,
        "profile",
        &id,
        &[&f.text],
        &[],
        "",
    );
    replace_anchors_conn(conn, "profile", &id, &anchors);

    // 按 (agent_id, project_key, project) 分项目淘汰非置顶的过期老事实（按热度 + 时间保底）
    let prune_rows: Vec<(i64, String)> = conn
        .prepare_cached(
            "SELECT rowid, id FROM profiles
             WHERE agent_id = ?1 AND project_key = ?2 AND project = ?3 AND pinned = 0 AND status = 'active'
             ORDER BY hit_count DESC, updated_at DESC, rowid DESC
             LIMIT -1 OFFSET ?4",
        )
        .ok()
        .and_then(|mut s| {
            s.query_map(
                params![agent_id, f.project_key, f.project, MAX_NON_PINNED_PROFILES as i64],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
            )
            .ok()
            .map(|rows| rows.flatten().collect())
        })
        .unwrap_or_default();
    for (rid, pid) in prune_rows {
        let _ = conn.execute("DELETE FROM profiles_fts WHERE rowid = ?1", params![rid]);
        let _ = conn.execute(
            "DELETE FROM memory_anchors WHERE target_kind = 'profile' AND target_id = ?1",
            params![pid],
        );
        let _ = conn.execute("DELETE FROM profiles WHERE id = ?1", params![pid]);
    }
    invalidate_hot_content(Some(agent_id));
    Ok(())
}

pub(crate) fn delete_profile_with_indexes_conn(
    conn: &Connection,
    profile_id: &str,
    agent_filter: &str,
) -> Result<(), String> {
    let old_rowid: Option<(i64, String)> = conn
        .prepare_cached(
            "SELECT rowid, agent_id FROM profiles WHERE id = ?1 AND (?2 = '' OR agent_id = ?2)",
        )
        .map_err(|e| e.to_string())?
        .query_row(params![profile_id, agent_filter], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()
        .map_err(|e| e.to_string())?;
    if let Some((rid, aid)) = old_rowid {
        let _ = conn.execute("DELETE FROM profiles_fts WHERE rowid = ?1", params![rid]);
        let _ = conn.execute(
            "DELETE FROM memory_anchors WHERE target_kind = 'profile' AND target_id = ?1",
            params![profile_id],
        );
        conn.execute("DELETE FROM profiles WHERE id = ?1", params![profile_id])
            .map_err(|e| e.to_string())?;
        invalidate_hot_content(Some(&aid));
    }
    Ok(())
}

pub(crate) fn load_profiles_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<HashMap<String, Vec<ProfileFact>>, String> {
    let mut map: HashMap<String, Vec<ProfileFact>> = HashMap::new();
    let sql = if agent_id.is_some() {
        "SELECT agent_id, id, project, project_key, text, pinned, updated_at, source_turn_id,
                kind, status, superseded_by, invalid_at, hit_count, last_used_at
         FROM profiles
         WHERE agent_id = ?1 AND (status = '' OR status = 'active')
         ORDER BY pinned DESC, hit_count DESC, updated_at DESC"
    } else {
        "SELECT agent_id, id, project, project_key, text, pinned, updated_at, source_turn_id,
                kind, status, superseded_by, invalid_at, hit_count, last_used_at
         FROM profiles
         WHERE status = '' OR status = 'active'
         ORDER BY pinned DESC, hit_count DESC, updated_at DESC"
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(String, ProfileFact)> {
        let aid: String = row.get(0)?;
        Ok((aid, map_profile_row(row, 1)?))
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

/// SQL 下推：按项目作用域精确读取生效中的 L1 长期档案（置顶与当前项目优先）。
pub(crate) fn load_profiles_scoped_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
    limit: usize,
) -> Result<Vec<ProfileFact>, String> {
    let sql = if project_key.is_empty() && project_name.is_empty() {
        "SELECT id, project, project_key, text, pinned, updated_at, source_turn_id,
                kind, status, superseded_by, invalid_at, hit_count, last_used_at
         FROM profiles
         WHERE agent_id = ?1 AND (status = '' OR status = 'active')
         ORDER BY pinned DESC, hit_count DESC, updated_at DESC, rowid DESC
         LIMIT ?2"
    } else {
        "SELECT id, project, project_key, text, pinned, updated_at, source_turn_id,
                kind, status, superseded_by, invalid_at, hit_count, last_used_at
         FROM profiles
         WHERE agent_id = ?1
           AND (status = '' OR status = 'active')
           AND (
             (project_key = '' AND project = '')
             OR (?2 != '' AND project_key = ?2)
             OR (project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3))
           )
         ORDER BY
           pinned DESC,
           CASE
             WHEN ?2 != '' AND project_key = ?2 THEN 0
             WHEN project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3) THEN 1
             ELSE 2
           END ASC,
           hit_count DESC,
           updated_at DESC,
           rowid DESC
         LIMIT ?4"
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| map_profile_row(row, 0);
    let rows = if project_key.is_empty() && project_name.is_empty() {
        stmt.query_map(params![agent_id, limit as i64], mapper)
    } else {
        stmt.query_map(
            params![agent_id, project_key, project_name, limit as i64],
            mapper,
        )
    }
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn load_profiles_by_ids_conn(
    conn: &Connection,
    agent_id: &str,
    ids: &[String],
    include_superseded: bool,
) -> Vec<ProfileFact> {
    if ids.is_empty() {
        return Vec::new();
    }
    let sql = if include_superseded {
        "SELECT id, project, project_key, text, pinned, updated_at, source_turn_id,
                kind, status, superseded_by, invalid_at, hit_count, last_used_at
         FROM profiles WHERE agent_id = ?1 AND id = ?2"
    } else {
        "SELECT id, project, project_key, text, pinned, updated_at, source_turn_id,
                kind, status, superseded_by, invalid_at, hit_count, last_used_at
         FROM profiles WHERE agent_id = ?1 AND id = ?2 AND (status = '' OR status = 'active')"
    };
    let Ok(mut stmt) = conn.prepare_cached(sql) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for id in ids {
        if let Ok(Some(p)) = stmt
            .query_row(params![agent_id, id], |r| map_profile_row(r, 0))
            .optional()
        {
            out.push(p);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 抽屉 2：需求与学习进度本（tasks）
// ---------------------------------------------------------------------------

fn map_task_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<NeuralTask> {
    let touched_str: String = row.get(8)?;
    let turns_str: String = row.get(10)?;
    let item_kind: String = row
        .get::<_, String>(15)
        .unwrap_or_else(|_| default_task_item_kind());
    Ok(NeuralTask {
        id: row.get(0)?,
        agent: row.get(1)?,
        project: row.get(2)?,
        project_key: row.get(3)?,
        working_dir: row.get(4)?,
        title: row.get(5)?,
        progress_note: row.get(6)?,
        remaining_note: row.get(7)?,
        touched_files: serde_json::from_str(&touched_str).unwrap_or_default(),
        session_id: row.get(9)?,
        source_turn_ids: serde_json::from_str(&turns_str).unwrap_or_default(),
        status: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
        done_at: row.get(14)?,
        item_kind: if item_kind.is_empty() {
            default_task_item_kind()
        } else {
            item_kind
        },
        next_step: row.get::<_, String>(16).unwrap_or_default(),
        team_run_id: row.get::<_, String>(17).unwrap_or_default(),
        team_task_id: row.get::<_, String>(18).unwrap_or_default(),
    })
}

pub(crate) fn upsert_task_conn(conn: &Connection, t: &NeuralTask) -> Result<(), String> {
    let touched = serde_json::to_string(&t.touched_files).unwrap_or_else(|_| "[]".into());
    let turns = serde_json::to_string(&t.source_turn_ids).unwrap_or_else(|_| "[]".into());
    let pkey = if t.project_key.is_empty() && !t.working_dir.is_empty() {
        normalize_project_key_str(&t.working_dir)
    } else {
        t.project_key.clone()
    };
    let item_kind = if t.item_kind.trim().is_empty() {
        default_task_item_kind()
    } else {
        t.item_kind.clone()
    };

    let old_rowid: Option<i64> = conn
        .prepare_cached("SELECT rowid FROM tasks WHERE id = ?1")
        .ok()
        .and_then(|mut s| {
            s.query_row(params![t.id], |r| r.get(0))
                .optional()
                .ok()
                .flatten()
        });
    if let Some(orid) = old_rowid {
        let _ = conn.execute("DELETE FROM tasks_fts WHERE rowid = ?1", params![orid]);
    }

    conn.execute(
        "INSERT OR REPLACE INTO tasks (
            id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
            touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at,
            item_kind, next_step, team_run_id, team_task_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
        params![
            t.id,
            t.agent,
            t.project,
            pkey,
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
            t.done_at,
            item_kind,
            t.next_step,
            t.team_run_id,
            t.team_task_id,
        ],
    )
    .map_err(|e| e.to_string())?;

    if let Ok(Some(new_rid)) = conn
        .prepare_cached("SELECT rowid FROM tasks WHERE id = ?1")
        .and_then(|mut s| {
            s.query_row(params![t.id], |r| r.get::<_, i64>(0))
                .optional()
        })
    {
        let tf_s = t.touched_files.join(" ");
        let _ = conn.execute(
            "INSERT OR REPLACE INTO tasks_fts (rowid, task_id, agent_id, title_col, detail_col)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                new_rid,
                t.id,
                t.agent,
                build_fts_column_text(&[&t.title, &t.project]),
                build_fts_column_text(&[&t.progress_note, &t.remaining_note, &t.next_step, &tf_s]),
            ],
        );
    }

    let anchors = extract_storage_anchors(
        &t.agent,
        &pkey,
        "task",
        &t.id,
        &[&t.title, &t.progress_note, &t.remaining_note, &t.next_step],
        &t.touched_files,
        &t.team_run_id,
    );
    replace_anchors_conn(conn, "task", &t.id, &anchors);
    invalidate_hot_content(Some(&t.agent));
    Ok(())
}

pub(crate) fn delete_task_with_indexes_conn(
    conn: &Connection,
    task_id: &str,
) -> Result<(), String> {
    let old_rowid: Option<(i64, String)> = conn
        .prepare_cached("SELECT rowid, agent_id FROM tasks WHERE id = ?1")
        .map_err(|e| e.to_string())?
        .query_row(params![task_id], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()
        .map_err(|e| e.to_string())?;
    if let Some((rid, aid)) = old_rowid {
        let _ = conn.execute("DELETE FROM tasks_fts WHERE rowid = ?1", params![rid]);
        let _ = conn.execute(
            "DELETE FROM memory_anchors WHERE target_kind = 'task' AND target_id = ?1",
            params![task_id],
        );
        conn.execute("DELETE FROM tasks WHERE id = ?1", params![task_id])
            .map_err(|e| e.to_string())?;
        invalidate_hot_content(Some(&aid));
    }
    Ok(())
}

pub(crate) fn load_tasks_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<Vec<NeuralTask>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at,
                item_kind, next_step, team_run_id, team_task_id
         FROM tasks WHERE agent_id = ?1 ORDER BY updated_at DESC"
    } else {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at,
                item_kind, next_step, team_run_id, team_task_id
         FROM tasks ORDER BY updated_at DESC"
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], map_task_row)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt
            .query_map([], map_task_row)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

/// SQL 下推：按项目作用域读取未完成需求与学习计划，避免全表读取。
pub(crate) fn load_open_tasks_scoped_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
    limit: usize,
) -> Result<Vec<NeuralTask>, String> {
    let sql = if project_key.is_empty() && project_name.is_empty() {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at,
                item_kind, next_step, team_run_id, team_task_id
         FROM tasks
         WHERE agent_id = ?1 AND status != 'done'
         ORDER BY updated_at DESC, rowid DESC
         LIMIT ?2"
    } else {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at,
                item_kind, next_step, team_run_id, team_task_id
         FROM tasks
         WHERE agent_id = ?1
           AND status != 'done'
           AND (
             (project_key = '' AND project = '')
             OR (?2 != '' AND project_key = ?2)
             OR (project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3))
           )
         ORDER BY
           CASE
             WHEN ?2 != '' AND project_key = ?2 THEN 0
             WHEN project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3) THEN 1
             ELSE 2
           END ASC,
           updated_at DESC,
           rowid DESC
         LIMIT ?4"
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let rows = if project_key.is_empty() && project_name.is_empty() {
        stmt.query_map(params![agent_id, limit as i64], map_task_row)
    } else {
        stmt.query_map(
            params![agent_id, project_key, project_name, limit as i64],
            map_task_row,
        )
    }
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn load_tasks_by_ids_conn(
    conn: &Connection,
    agent_id: &str,
    ids: &[String],
) -> Vec<NeuralTask> {
    if ids.is_empty() {
        return Vec::new();
    }
    let Ok(mut stmt) = conn.prepare_cached(
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at,
                item_kind, next_step, team_run_id, team_task_id
         FROM tasks WHERE agent_id = ?1 AND id = ?2",
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for id in ids {
        if let Ok(Some(t)) = stmt
            .query_row(params![agent_id, id], map_task_row)
            .optional()
        {
            out.push(t);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 抽屉 3：知识与经验活卡片（nodes）+ 有类型关联边（links）
// ---------------------------------------------------------------------------

pub(crate) fn map_node_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<NeuralNode> {
    let win_str: String = row.get(8)?;
    let ev_str: String = row.get(10)?;
    let aliases_str: String = row.get::<_, String>(12).unwrap_or_else(|_| "[]".into());
    let tags_str: String = row.get::<_, String>(13).unwrap_or_else(|_| "[]".into());
    let facts_str: String = row.get::<_, String>(14).unwrap_or_else(|_| "[]".into());
    Ok(NeuralNode {
        id: row.get(0)?,
        cat: row.get(1)?,
        name: row.get(2)?,
        project: row.get(3)?,
        project_key: row.get(4)?,
        status: row.get(5)?,
        mastery: row.get(6)?,
        tier: row.get(7)?,
        windows: serde_json::from_str(&win_str).unwrap_or_else(|_| default_windows()),
        summary: row.get(9)?,
        events: serde_json::from_str(&ev_str).unwrap_or_default(),
        updated_at: row.get(11)?,
        aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
        search_tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        facts: serde_json::from_str(&facts_str).unwrap_or_default(),
        hit_count: row.get::<_, i64>(15).unwrap_or(0),
        last_used_at: row.get::<_, String>(16).unwrap_or_default(),
    })
}

pub(crate) fn upsert_node_conn(conn: &Connection, n: &NeuralNode) -> Result<(), String> {
    let windows_json = serde_json::to_string(&n.windows).unwrap_or_else(|_| "[\"all\"]".into());
    let events_json = serde_json::to_string(&n.events).unwrap_or_else(|_| "[]".into());
    let aliases_json = serde_json::to_string(&n.aliases).unwrap_or_else(|_| "[]".into());
    let tags_json = serde_json::to_string(&n.search_tags).unwrap_or_else(|_| "[]".into());
    let mut facts = n.facts.clone();
    if facts.is_empty() && !n.summary.trim().is_empty() {
        facts.push(NodeFact {
            id: new_id("nf"),
            kind: "fact".into(),
            text: n.summary.trim().to_string(),
            status: "active".into(),
            updated_at: if n.updated_at.trim().is_empty() {
                now_ts_minute()
            } else {
                n.updated_at.clone()
            },
            ..Default::default()
        });
    }
    let facts_json = serde_json::to_string(&facts).unwrap_or_else(|_| "[]".into());
    let updated_at = if n.updated_at.trim().is_empty() {
        now_ts_minute()
    } else {
        n.updated_at.clone()
    };

    let old_rowid: Option<i64> = conn
        .prepare_cached("SELECT rowid FROM nodes WHERE id = ?1")
        .ok()
        .and_then(|mut s| {
            s.query_row(params![n.id], |r| r.get(0))
                .optional()
                .ok()
                .flatten()
        });
    if let Some(orid) = old_rowid {
        let _ = conn.execute("DELETE FROM nodes_fts WHERE rowid = ?1", params![orid]);
    }

    conn.execute(
        "INSERT OR REPLACE INTO nodes (
            id, agent_id, name, project, project_key, status, mastery, tier, windows_json,
            summary, events_json, updated_at, aliases_json, search_tags_json, facts_json,
            hit_count, last_used_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
        params![
            n.id,
            n.cat,
            n.name,
            n.project,
            n.project_key,
            n.status,
            n.mastery,
            n.tier,
            windows_json,
            n.summary,
            events_json,
            updated_at,
            aliases_json,
            tags_json,
            facts_json,
            n.hit_count,
            n.last_used_at,
        ],
    )
    .map_err(|e| e.to_string())?;

    let alias_joined = n.aliases.join(" ");
    let tags_joined = n.search_tags.join(" ");
    let active_facts_joined = facts
        .iter()
        .filter(|f| f.is_active())
        .map(|f| f.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let events_joined = n
        .events
        .iter()
        .take(6)
        .map(|e| e.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let anchors = extract_storage_anchors(
        &n.cat,
        &n.project_key,
        "node",
        &n.id,
        &[
            &n.name,
            &alias_joined,
            &tags_joined,
            &n.summary,
            &active_facts_joined,
        ],
        &[],
        "",
    );
    let anchor_joined = anchors
        .iter()
        .map(|a| a.anchor_value.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    if let Ok(Some(new_rid)) = conn
        .prepare_cached("SELECT rowid FROM nodes WHERE id = ?1")
        .and_then(|mut s| {
            s.query_row(params![n.id], |r| r.get::<_, i64>(0))
                .optional()
        })
    {
        let _ = conn.execute(
            "INSERT OR REPLACE INTO nodes_fts (rowid, node_id, agent_id, title_col, tags_col, summary_col, body_col)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                new_rid,
                n.id,
                n.cat,
                build_fts_column_text(&[&n.name, &alias_joined]),
                build_fts_column_text(&[&tags_joined, &n.project, &anchor_joined]),
                build_fts_column_text(&[&n.summary, &active_facts_joined]),
                build_fts_column_text(&[&events_joined]),
            ],
        );
    }
    replace_anchors_conn(conn, "node", &n.id, &anchors);
    invalidate_hot_content(Some(&n.cat));
    Ok(())
}

pub(crate) fn delete_node_with_indexes_conn(
    conn: &Connection,
    node_id: &str,
) -> Result<(), String> {
    let old_rowid: Option<(i64, String)> = conn
        .prepare_cached("SELECT rowid, agent_id FROM nodes WHERE id = ?1")
        .map_err(|e| e.to_string())?
        .query_row(params![node_id], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()
        .map_err(|e| e.to_string())?;
    if let Some((rid, aid)) = old_rowid {
        let _ = conn.execute("DELETE FROM nodes_fts WHERE rowid = ?1", params![rid]);
        let _ = conn.execute(
            "DELETE FROM memory_anchors WHERE target_kind = 'node' AND target_id = ?1",
            params![node_id],
        );
        let _ = conn.execute(
            "DELETE FROM links WHERE source = ?1 OR target = ?1",
            params![node_id],
        );
        conn.execute("DELETE FROM nodes WHERE id = ?1", params![node_id])
            .map_err(|e| e.to_string())?;
        invalidate_hot_content(Some(&aid));
    }
    Ok(())
}

pub(crate) fn load_nodes_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<Vec<NeuralNode>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json,
                summary, events_json, updated_at, aliases_json, search_tags_json, facts_json,
                hit_count, last_used_at
         FROM nodes WHERE agent_id = ?1 ORDER BY updated_at DESC"
    } else {
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json,
                summary, events_json, updated_at, aliases_json, search_tags_json, facts_json,
                hit_count, last_used_at
         FROM nodes ORDER BY updated_at DESC"
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], map_node_row)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt
            .query_map([], map_node_row)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

pub(crate) fn load_nodes_by_ids_conn(
    conn: &Connection,
    agent_id: &str,
    ids: &[String],
) -> Vec<NeuralNode> {
    if ids.is_empty() {
        return Vec::new();
    }
    let Ok(mut stmt) = conn.prepare_cached(
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json,
                summary, events_json, updated_at, aliases_json, search_tags_json, facts_json,
                hit_count, last_used_at
         FROM nodes WHERE agent_id = ?1 AND id = ?2",
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for id in ids {
        if let Ok(Some(n)) = stmt
            .query_row(params![agent_id, id], map_node_row)
            .optional()
        {
            out.push(n);
        }
    }
    out
}

/// 1-hop 脑图关联扩展：读取与给定节点直接相连的高价值邻居节点。
pub(crate) fn load_neighbor_nodes_conn(
    conn: &Connection,
    agent_id: &str,
    node_id: &str,
    limit: usize,
) -> Vec<(NeuralNode, String)> {
    let Ok(mut stmt) = conn.prepare_cached(
        "SELECT n.id, n.agent_id, n.name, n.project, n.project_key, n.status, n.mastery, n.tier,
                n.windows_json, n.summary, n.events_json, n.updated_at, n.aliases_json,
                n.search_tags_json, n.facts_json, n.hit_count, n.last_used_at, l.relation
         FROM links l
         JOIN nodes n ON (
             (l.source = ?2 AND n.id = l.target)
             OR (l.target = ?2 AND n.id = l.source)
         )
         WHERE n.agent_id = ?1 AND n.id != ?2
         ORDER BY
           CASE l.relation
             WHEN 'depends_on' THEN 0
             WHEN 'causes' THEN 1
             WHEN 'part_of' THEN 2
             ELSE 3
           END ASC,
           n.hit_count DESC,
           n.mastery DESC,
           n.updated_at DESC
         LIMIT ?3",
    ) else {
        return Vec::new();
    };
    stmt.query_map(params![agent_id, node_id, limit as i64], |r| {
        let node = map_node_row(r)?;
        let rel: String = r
            .get::<_, String>(17)
            .unwrap_or_else(|_| default_link_relation());
        Ok((node, rel))
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

/// 仅查询当前/全局范围内最近的若干完整节点（供写入前先查后合对照，含完整摘要与要点）。
pub(crate) fn load_recent_nodes_scoped_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
    limit: usize,
) -> Vec<NeuralNode> {
    let sql = if project_key.is_empty() && project_name.is_empty() {
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json,
                summary, events_json, updated_at, aliases_json, search_tags_json, facts_json,
                hit_count, last_used_at
         FROM nodes
         WHERE agent_id = ?1
         ORDER BY hit_count DESC, updated_at DESC
         LIMIT ?2"
    } else {
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json,
                summary, events_json, updated_at, aliases_json, search_tags_json, facts_json,
                hit_count, last_used_at
         FROM nodes
         WHERE agent_id = ?1
           AND ((?2 != '' AND project_key = ?2) OR (project_key = '' AND LOWER(project) = LOWER(?3)) OR project = '')
         ORDER BY
           CASE
             WHEN ?2 != '' AND project_key = ?2 THEN 0
             WHEN project_key = '' AND LOWER(project) = LOWER(?3) THEN 1
             ELSE 2
           END ASC,
           hit_count DESC,
           updated_at DESC
         LIMIT ?4"
    };
    let Ok(mut stmt) = conn.prepare_cached(sql) else {
        return Vec::new();
    };
    if project_key.is_empty() && project_name.is_empty() {
        stmt.query_map(params![agent_id, limit as i64], map_node_row)
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
    } else {
        stmt.query_map(
            params![agent_id, project_key, project_name, limit as i64],
            map_node_row,
        )
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
    }
}

pub(crate) fn upsert_link_conn(conn: &Connection, source: &str, target: &str, relation: &str) {
    if source.is_empty() || target.is_empty() || source == target {
        return;
    }
    let rel = if relation.trim().is_empty() {
        "related"
    } else {
        relation.trim()
    };
    let _ = conn.execute(
        "INSERT INTO links (source, target, relation)
         SELECT ?1, ?2, ?3
         WHERE NOT EXISTS (
             SELECT 1 FROM links
             WHERE (source = ?1 AND target = ?2) OR (source = ?2 AND target = ?1)
         )",
        params![source, target, rel],
    );
}

pub(crate) fn load_links_conn(conn: &Connection) -> Result<Vec<NeuralLink>, String> {
    let mut stmt = conn
        .prepare_cached("SELECT source, target, relation FROM links")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let rel: String = r
                .get::<_, String>(2)
                .unwrap_or_else(|_| default_link_relation());
            Ok(NeuralLink {
                source: r.get(0)?,
                target: r.get(1)?,
                relation: if rel.is_empty() {
                    default_link_relation()
                } else {
                    rel
                },
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 抽屉 4：每日精华大事记（timeline）
// ---------------------------------------------------------------------------

pub(crate) fn map_timeline_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TimelineEvent> {
    let kw_str: String = row.get(12)?;
    let tf_str: String = row.get(13)?;
    let kind: String = row
        .get::<_, String>(15)
        .unwrap_or_else(|_| default_journal_kind());
    let node_ids_str: String = row.get::<_, String>(17).unwrap_or_else(|_| "[]".into());
    Ok(TimelineEvent {
        id: row.get(0)?,
        ts: row.get(1)?,
        agent: row.get(2)?,
        project: row.get(3)?,
        project_key: row.get(4)?,
        working_dir: row.get(5)?,
        session_id: row.get(6)?,
        turn_id: row.get(7)?,
        raw_file: row.get(8)?,
        user_goal: row.get(9)?,
        summary: row.get(10)?,
        ai_outcome: row.get(11)?,
        keywords: serde_json::from_str(&kw_str).unwrap_or_default(),
        touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
        distilled_by: row.get(14)?,
        kind: if kind.is_empty() {
            default_journal_kind()
        } else {
            kind
        },
        team_run_id: row.get::<_, String>(16).unwrap_or_default(),
        node_ids: serde_json::from_str(&node_ids_str).unwrap_or_default(),
    })
}

pub(crate) fn insert_timeline_conn(conn: &Connection, ev: &TimelineEvent) -> Result<(), String> {
    let id = if ev.id.trim().is_empty() {
        new_id("tl")
    } else {
        ev.id.clone()
    };
    let kw_json = serde_json::to_string(&ev.keywords).unwrap_or_else(|_| "[]".into());
    let tf_json = serde_json::to_string(&ev.touched_files).unwrap_or_else(|_| "[]".into());
    let node_ids_json = serde_json::to_string(&ev.node_ids).unwrap_or_else(|_| "[]".into());
    let pkey = if ev.project_key.is_empty() && !ev.working_dir.is_empty() {
        normalize_project_key_str(&ev.working_dir)
    } else {
        ev.project_key.clone()
    };
    let kind = if ev.kind.trim().is_empty() {
        default_journal_kind()
    } else {
        ev.kind.clone()
    };

    let old_rowid: Option<i64> = conn
        .prepare_cached("SELECT rowid FROM timeline WHERE id = ?1")
        .ok()
        .and_then(|mut s| {
            s.query_row(params![id], |r| r.get(0))
                .optional()
                .ok()
                .flatten()
        });
    if let Some(orid) = old_rowid {
        let _ = conn.execute("DELETE FROM timeline_fts WHERE rowid = ?1", params![orid]);
    }

    conn.execute(
        "INSERT OR REPLACE INTO timeline (
            id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
            user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by,
            kind, team_run_id, node_ids_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
        params![
            id,
            ev.ts,
            ev.agent,
            ev.project,
            pkey,
            ev.working_dir,
            ev.session_id,
            ev.turn_id,
            ev.raw_file,
            ev.user_goal,
            ev.summary,
            ev.ai_outcome,
            kw_json,
            tf_json,
            ev.distilled_by,
            kind,
            ev.team_run_id,
            node_ids_json,
        ],
    )
    .map_err(|e| e.to_string())?;

    let kw_joined = ev.keywords.join(" ");
    let tf_joined = ev.touched_files.join(" ");
    if let Ok(Some(new_rid)) = conn
        .prepare_cached("SELECT rowid FROM timeline WHERE id = ?1")
        .and_then(|mut s| s.query_row(params![id], |r| r.get::<_, i64>(0)).optional())
    {
        let _ = conn.execute(
            "INSERT OR REPLACE INTO timeline_fts (rowid, event_id, agent_id, keywords_col, goal_col, summary_col)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                new_rid,
                id,
                ev.agent,
                build_fts_column_text(&[&kw_joined, &tf_joined]),
                build_fts_column_text(&[&ev.user_goal, &ev.project]),
                build_fts_column_text(&[&ev.summary, &ev.ai_outcome]),
            ],
        );
    }

    let anchors = extract_storage_anchors(
        &ev.agent,
        &pkey,
        "timeline",
        &id,
        &[&ev.user_goal, &ev.summary, &ev.ai_outcome, &kw_joined],
        &ev.touched_files,
        &ev.team_run_id,
    );
    replace_anchors_conn(conn, "timeline", &id, &anchors);
    invalidate_hot_content(Some(&ev.agent));
    Ok(())
}

pub(crate) fn load_timeline_conn(
    conn: &Connection,
    agent_id: Option<&str>,
    limit: usize,
) -> Result<Vec<TimelineEvent>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by,
                kind, team_run_id, node_ids_json
         FROM timeline WHERE agent_id = ?1 ORDER BY ts DESC LIMIT ?2"
    } else {
        "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by,
                kind, team_run_id, node_ids_json
         FROM timeline ORDER BY ts DESC LIMIT ?1"
    };
    let mut stmt = conn.prepare_cached(sql).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid, limit as i64], map_timeline_row)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt
            .query_map(params![limit as i64], map_timeline_row)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

pub(crate) fn load_timeline_by_ids_conn(
    conn: &Connection,
    agent_id: &str,
    ids: &[String],
) -> Vec<TimelineEvent> {
    if ids.is_empty() {
        return Vec::new();
    }
    let Ok(mut stmt) = conn.prepare_cached(
        "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by,
                kind, team_run_id, node_ids_json
         FROM timeline WHERE agent_id = ?1 AND id = ?2",
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for id in ids {
        if let Ok(Some(ev)) = stmt
            .query_row(params![agent_id, id], map_timeline_row)
            .optional()
        {
            out.push(ev);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 抽屉 5：原始对话冷仓库（raw_turns，不建 FTS5 倒排，只存档与按需溯源）
// ---------------------------------------------------------------------------

pub(crate) fn upsert_raw_turn_conn(conn: &Connection, t: &RawTurn) -> Result<(), String> {
    let clean = redact_raw_turn(t.clone());
    let tf_json = serde_json::to_string(&clean.touched_files).unwrap_or_else(|_| "[]".into());
    let pkey = if clean.project_key.is_empty() && !clean.working_dir.is_empty() {
        normalize_project_key_str(&clean.working_dir)
    } else {
        clean.project_key.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO raw_turns (turn_id, ts, agent_id, project, project_key, working_dir, session_id, user_text, ai_text, touched_files_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            clean.turn_id,
            clean.ts,
            clean.agent,
            clean.project,
            pkey,
            clean.working_dir,
            clean.session_id,
            clean.user_text,
            clean.ai_text,
            tf_json
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn load_raw_turn_conn(
    conn: &Connection,
    turn_id: &str,
) -> Result<Option<RawTurn>, String> {
    let mut stmt = conn
        .prepare_cached(
            "SELECT turn_id, ts, agent_id, project, project_key, working_dir, session_id, user_text, ai_text, touched_files_json
             FROM raw_turns WHERE turn_id = ?1",
        )
        .map_err(|e| e.to_string())?;
    stmt.query_row(params![turn_id], |row| {
        let tf_str: String = row.get(9)?;
        Ok(RawTurn {
            turn_id: row.get(0)?,
            ts: row.get(1)?,
            agent: row.get(2)?,
            project: row.get(3)?,
            project_key: row.get(4)?,
            working_dir: row.get(5)?,
            session_id: row.get(6)?,
            user_text: row.get(7)?,
            ai_text: row.get(8)?,
            touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
        })
    })
    .optional()
    .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// 持久化草稿队列（draft_turns）：聊天当下轻写入，按自然触发点打包整理，重启不丢
// ---------------------------------------------------------------------------

pub(crate) fn enqueue_draft_turn_conn(conn: &Connection, d: &DraftTurn) -> Result<bool, String> {
    let tf_json = serde_json::to_string(&d.touched_files).unwrap_or_else(|_| "[]".into());
    let changed = conn
        .execute(
            "INSERT OR IGNORE INTO draft_turns (
                turn_id, session_id, agent_id, project, project_key, working_dir,
                user_text, ai_text, touched_files_json, provider_name,
                explicit_remember, retry_count, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                d.turn_id,
                d.session_id,
                d.agent_id,
                d.project,
                d.project_key,
                d.working_dir,
                d.user_text,
                d.ai_text,
                tf_json,
                d.provider_name,
                if d.explicit_remember { 1 } else { 0 },
                d.retry_count,
                d.created_at,
            ],
        )
        .map_err(|e| e.to_string())?;
    Ok(changed > 0)
}

fn map_draft_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DraftTurn> {
    let tf_str: String = row.get(9)?;
    let exp_int: i64 = row.get(11)?;
    Ok(DraftTurn {
        id: row.get(0)?,
        turn_id: row.get(1)?,
        session_id: row.get(2)?,
        agent_id: row.get(3)?,
        project: row.get(4)?,
        project_key: row.get(5)?,
        working_dir: row.get(6)?,
        user_text: row.get(7)?,
        ai_text: row.get(8)?,
        touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
        provider_name: row.get(10)?,
        explicit_remember: exp_int != 0,
        retry_count: row.get(12)?,
        created_at: row.get(13)?,
    })
}

pub(crate) fn load_pending_drafts_for_scope_conn(
    conn: &Connection,
    agent_id: &str,
    session_id: &str,
) -> Result<Vec<DraftTurn>, String> {
    let mut stmt = conn
        .prepare_cached(
            "SELECT id, turn_id, session_id, agent_id, project, project_key, working_dir,
                    user_text, ai_text, touched_files_json, provider_name,
                    explicit_remember, retry_count, created_at
             FROM draft_turns
             WHERE agent_id = ?1 AND session_id = ?2
             ORDER BY id ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![agent_id, session_id], map_draft_row)
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn load_all_pending_draft_groups_conn(
    conn: &Connection,
    agent_filter: Option<&str>,
) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    if let Some(aid) = agent_filter {
        let mut stmt = conn
            .prepare_cached(
                "SELECT DISTINCT agent_id, session_id FROM draft_turns
                 WHERE agent_id = ?1 ORDER BY MIN(id) ASC",
            )
            .or_else(|_| {
                conn.prepare_cached(
                    "SELECT agent_id, session_id FROM draft_turns
                     WHERE agent_id = ?1 GROUP BY agent_id, session_id ORDER BY MIN(id) ASC",
                )
            })
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![aid], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(|e| e.to_string())?;
        for r in rows.flatten() {
            out.push(r);
        }
    } else {
        let mut stmt = conn
            .prepare_cached(
                "SELECT agent_id, session_id FROM draft_turns
                 GROUP BY agent_id, session_id ORDER BY MIN(id) ASC",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(|e| e.to_string())?;
        for r in rows.flatten() {
            out.push(r);
        }
    }
    Ok(out)
}

pub(crate) fn delete_draft_turns_conn(conn: &Connection, draft_ids: &[i64]) {
    if draft_ids.is_empty() {
        return;
    }
    if let Ok(mut stmt) = conn.prepare_cached("DELETE FROM draft_turns WHERE id = ?1") {
        for id in draft_ids {
            let _ = stmt.execute(params![id]);
        }
    }
}

pub(crate) fn bump_draft_retry_conn(conn: &Connection, draft_ids: &[i64]) {
    if draft_ids.is_empty() {
        return;
    }
    if let Ok(mut stmt) =
        conn.prepare_cached("UPDATE draft_turns SET retry_count = retry_count + 1 WHERE id = ?1")
    {
        for id in draft_ids {
            let _ = stmt.execute(params![id]);
        }
    }
}

pub(crate) fn count_pending_drafts_conn(conn: &Connection, agent_filter: Option<&str>) -> usize {
    if let Some(aid) = agent_filter {
        conn.prepare_cached("SELECT COUNT(*) FROM draft_turns WHERE agent_id = ?1")
            .ok()
            .and_then(|mut s| s.query_row(params![aid], |r| r.get::<_, i64>(0)).ok())
            .unwrap_or(0) as usize
    } else {
        conn.prepare_cached("SELECT COUNT(*) FROM draft_turns")
            .ok()
            .and_then(|mut s| s.query_row([], |r| r.get::<_, i64>(0)).ok())
            .unwrap_or(0) as usize
    }
}

// ---------------------------------------------------------------------------
// 使用率热度回写 + 极薄大脑目录 + 已知项目 + 留存淘汰
// ---------------------------------------------------------------------------

/// 记录本轮命中的高价值记忆使用次数（越常帮上忙越靠前，不影响当前读取延迟）。
pub(crate) fn record_memory_hits_conn(
    conn: &Connection,
    node_ids: &[String],
    profile_ids: &[String],
) {
    let ts = now_ts_minute();
    if !node_ids.is_empty() {
        if let Ok(mut stmt) = conn.prepare_cached(
            "UPDATE nodes SET hit_count = hit_count + 1, last_used_at = ?1 WHERE id = ?2",
        ) {
            for nid in node_ids {
                let _ = stmt.execute(params![ts, nid]);
            }
        }
    }
    if !profile_ids.is_empty() {
        if let Ok(mut stmt) = conn.prepare_cached(
            "UPDATE profiles SET hit_count = hit_count + 1, last_used_at = ?1 WHERE id = ?2",
        ) {
            for pid in profile_ids {
                let _ = stmt.execute(params![ts, pid]);
            }
        }
    }
}

/// 构造极薄「大脑目录」（供开场静态前缀 0ms 挂载：活跃需求/学习计划清单 + Top 知识卡片标题清单）。
pub(crate) fn build_thin_memory_directory_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
) -> Result<String, String> {
    let mut parts: Vec<String> = Vec::new();

    let open_tasks = load_open_tasks_scoped_conn(conn, agent_id, project_key, project_name, 8)?;
    if !open_tasks.is_empty() {
        let mut lines = Vec::new();
        for t in &open_tasks {
            let kind_tag = if t.item_kind == "study" {
                "study"
            } else if t.item_kind == "bugfix" {
                "bugfix"
            } else {
                "task"
            };
            let proj_tag = if t.project.is_empty() {
                String::new()
            } else {
                format!("[{}] ", t.project)
            };
            let next_part = if !t.next_step.trim().is_empty() {
                format!(" | next: {}", t.next_step.trim())
            } else if !t.remaining_note.trim().is_empty() {
                format!(" | remaining: {}", t.remaining_note.trim())
            } else {
                String::new()
            };
            let team_part = if !t.team_run_id.trim().is_empty() {
                format!(" [team={}]", t.team_run_id.trim())
            } else {
                String::new()
            };
            lines.push(format!(
                "- ({kind_tag}/{}, id={}) {proj_tag}{}{next_part}{team_part}",
                t.status, t.id, t.title
            ));
        }
        parts.push(format!(
            "[Active Progress & Study Plans]\n{}",
            lines.join("\n")
        ));
    }

    let nodes = load_recent_nodes_scoped_conn(conn, agent_id, project_key, project_name, 16);
    if !nodes.is_empty() {
        let titles: Vec<String> = nodes
            .iter()
            .map(|n| {
                let p = if n.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", n.project)
                };
                if n.aliases.is_empty() {
                    format!("{p}{} ({}%)", n.name, n.mastery)
                } else {
                    let alias_preview = n
                        .aliases
                        .iter()
                        .take(2)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("/");
                    format!("{p}{} [aka: {}] ({}%)", n.name, alias_preview, n.mastery)
                }
            })
            .collect();
        parts.push(format!(
            "[Knowledge Card Index (call `memory_recall` for full details)]\n- {}",
            titles.join("\n- ")
        ));
    }

    Ok(parts.join("\n\n"))
}

#[derive(Debug, Clone)]
pub(crate) struct KnownProject {
    pub(crate) name: String,
    pub(crate) key: String,
}

/// 查询库内所有出现过的项目（同时保留展示名与规范化路径键，使用覆盖索引与预编译语句）。
pub(crate) fn list_known_projects_conn(conn: &Connection, agent_id: &str) -> Vec<KnownProject> {
    let mut seen: HashMap<String, KnownProject> = HashMap::new();
    for sql in [
        "SELECT DISTINCT project, project_key FROM nodes WHERE agent_id = ?1 AND (project != '' OR project_key != '')",
        "SELECT DISTINCT project, project_key FROM timeline WHERE agent_id = ?1 AND (project != '' OR project_key != '')",
        "SELECT DISTINCT project, project_key FROM tasks WHERE agent_id = ?1 AND (project != '' OR project_key != '')",
    ] {
        if let Ok(mut stmt) = conn.prepare_cached(sql) {
            if let Ok(rows) = stmt.query_map(params![agent_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for (name, key) in rows.flatten() {
                    let clean_name = name.trim().to_string();
                    let clean_key = key.trim().to_string();
                    if clean_name.is_empty() && clean_key.is_empty() {
                        continue;
                    }
                    let dedup_k = if !clean_key.is_empty() {
                        clean_key.clone()
                    } else {
                        clean_name.to_lowercase()
                    };
                    seen.entry(dedup_k).or_insert(KnownProject {
                        name: clean_name,
                        key: clean_key,
                    });
                }
            }
        }
    }
    let mut out: Vec<KnownProject> = seen.into_values().collect();
    out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.key.cmp(&b.key)));
    out
}

/// 控制每个智能体的存储规模，使用 rowid 精准清理 FTS 与路牌，不再全表扫描 FTS 影表。
pub(crate) fn prune_retention_conn(conn: &Connection, agent_id: &str) {
    if conn.execute_batch("SAVEPOINT neural_prune;").is_err() {
        return;
    }
    let res = (|| -> rusqlite::Result<()> {
        // 1. 批量淘汰超出上限的冷层 raw_turns（已移除 raw_turns_fts）
        conn.execute(
            "DELETE FROM raw_turns WHERE turn_id IN (
                 SELECT turn_id FROM raw_turns WHERE agent_id = ?1
                 ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?2
             )",
            params![agent_id, MAX_RAW_TURNS_PER_AGENT as i64],
        )?;

        // 2. 批量淘汰超出上限的 timeline 及 rowid 对应 FTS / anchors
        let tl_prune: Vec<(i64, String)> = {
            let mut stmt = conn.prepare_cached(
                "SELECT rowid, id FROM timeline WHERE agent_id = ?1
                 ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?2",
            )?;
            let rows = stmt.query_map(params![agent_id, MAX_TIMELINE_PER_AGENT as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            rows.flatten().collect()
        };
        for (rid, tid) in tl_prune {
            conn.execute("DELETE FROM timeline_fts WHERE rowid = ?1", params![rid])?;
            conn.execute(
                "DELETE FROM memory_anchors WHERE target_kind = 'timeline' AND target_id = ?1",
                params![tid],
            )?;
            conn.execute("DELETE FROM timeline WHERE id = ?1", params![tid])?;
        }

        // 3. 批量淘汰超出上限的 nodes（优先保留 solid、高命中热度 hit_count、高连接度 Hub 与高掌握度节点）
        let node_prune: Vec<(i64, String)> = {
            let mut stmt = conn.prepare_cached(
                "SELECT n.rowid, n.id FROM nodes n
                 WHERE n.agent_id = ?1
                 ORDER BY
                   CASE WHEN n.status = 'solid' THEN 1 ELSE 0 END DESC,
                   n.hit_count DESC,
                   (SELECT COUNT(*) FROM links l WHERE l.source = n.id OR l.target = n.id) DESC,
                   n.mastery DESC,
                   n.updated_at DESC,
                   n.rowid DESC
                 LIMIT -1 OFFSET ?2",
            )?;
            let rows = stmt.query_map(params![agent_id, MAX_NODES_PER_AGENT as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            rows.flatten().collect()
        };
        for (rid, nid) in node_prune {
            conn.execute("DELETE FROM nodes_fts WHERE rowid = ?1", params![rid])?;
            conn.execute(
                "DELETE FROM memory_anchors WHERE target_kind = 'node' AND target_id = ?1",
                params![nid],
            )?;
            conn.execute(
                "DELETE FROM links WHERE source = ?1 OR target = ?1",
                params![nid],
            )?;
            conn.execute("DELETE FROM nodes WHERE id = ?1", params![nid])?;
        }
        Ok(())
    })();
    if res.is_ok() {
        let _ = conn.execute_batch("RELEASE SAVEPOINT neural_prune;");
        invalidate_hot_content(Some(agent_id));
    } else {
        let _ = conn
            .execute_batch("ROLLBACK TO SAVEPOINT neural_prune; RELEASE SAVEPOINT neural_prune;");
    }
}

/// 测试辅助：创建内存 SQLite 并初始化 v3 表结构。
#[cfg(test)]
pub(crate) fn setup_mem_db() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    init_schema(&conn).expect("init_schema");
    for (idx, a) in default_agents().iter().enumerate() {
        upsert_agent_conn(&conn, a, idx as i64).expect("seed agent");
    }
    let cfg = NeuralConfig {
        active_agent: "backend".into(),
        ..Default::default()
    };
    save_config_conn(&conn, &cfg).expect("save cfg");
    conn
}
