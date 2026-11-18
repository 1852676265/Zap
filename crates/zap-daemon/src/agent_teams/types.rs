//! `zap-agent-teams` 核心数据模型、DAG 波次拓扑切分与状态机原语。

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use chrono::Local;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_REPAIRS: u32 = 2;
pub const HARD_CAP_MAX_REPAIRS: u32 = 3;
pub const MAX_CONTRACT_CHARS: usize = 3600;
pub const MAX_EXCERPT_CHARS: usize = 2400;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamGlobalConfig {
    #[serde(default = "default_true")]
    pub auto_verify: bool,
    #[serde(default = "default_max_repairs")]
    pub max_repairs: u32,
    #[serde(default)]
    pub default_model: String,
    #[serde(default = "default_true")]
    pub show_input_capsule: bool,
    #[serde(default = "default_max_parallel")]
    pub max_parallel_per_wave: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_repairs() -> u32 {
    DEFAULT_MAX_REPAIRS
}

fn default_max_parallel() -> usize {
    3
}

impl Default for TeamGlobalConfig {
    fn default() -> Self {
        Self {
            auto_verify: true,
            max_repairs: DEFAULT_MAX_REPAIRS,
            default_model: String::new(),
            show_input_capsule: true,
            max_parallel_per_wave: 3,
        }
    }
}

/// 记住每个职能领域（`sql` / `backend` / `frontend` / `general`）上次确认的默认项目目录与专属记忆智能体。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoleBindingPreset {
    pub domain: String,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default = "default_agent_none")]
    pub agent_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub updated_at: String,
}

fn default_agent_none() -> String {
    "none".to_string()
}

/// 一次完整的跨项目协作编排单（支持 `pipeline` 研发接力流水线 与 `triage` 跨库联合排障会诊）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamRun {
    pub id: String,
    pub title: String,
    pub user_goal: String,
    /// `"pipeline"` | `"triage"`
    #[serde(default = "default_mode_pipeline")]
    pub mode: String,
    /// `"draft"` | `"running"` | `"paused"` | `"completed"` | `"failed"` | `"cancelled"`
    #[serde(default = "default_status_draft")]
    pub status: String,
    /// 乐观锁递增版本号（CAS 防旧快照覆盖）
    #[serde(default)]
    pub revision: u64,
    #[serde(default = "default_true")]
    pub auto_verify: bool,
    #[serde(default = "default_max_repairs")]
    pub max_repairs: u32,
    #[serde(default)]
    pub lead_model: String,
    #[serde(default)]
    pub source_triage_id: String,
    #[serde(default)]
    pub synthesis_report: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub finished_at: String,
}

fn default_mode_pipeline() -> String {
    "pipeline".to_string()
}

fn default_status_draft() -> String {
    "draft".to_string()
}

/// 单个跨仓库子任务节点（绑定独立 `working_dir` 与专属 `neural_memory` 智能体）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamTask {
    pub id: String,
    pub run_id: String,
    /// `"sql"` | `"backend"` | `"frontend"` | `"general"`
    #[serde(default = "default_domain_general")]
    pub domain: String,
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub acceptance_criteria: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default = "default_wave_one")]
    pub wave: usize,
    #[serde(default)]
    pub sort_order: usize,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default)]
    pub project_hash: String,
    #[serde(default = "default_agent_none")]
    pub agent_id: String,
    #[serde(default)]
    pub agent_label: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub session_id: String,
    /// `"pending"` | `"ready"` | `"running"` | `"verifying"` | `"repairing"` | `"completed"` | `"failed"` | `"cancelled"`
    #[serde(default = "default_task_pending")]
    pub status: String,
    /// 单调递增执行尝试计数
    #[serde(default)]
    pub attempt: u32,
    /// 当前有效尝试的唯一栅栏令牌（防旧异步任务回写污染）
    #[serde(default)]
    pub attempt_id: String,
    /// 已触发自动打回返工的次数
    #[serde(default)]
    pub repair_count: u32,
    #[serde(default)]
    pub contract_summary: String,
    #[serde(default)]
    pub last_output_excerpt: String,
    /// `""` | `"passed"` | `"rejected"` | `"skipped"`
    #[serde(default)]
    pub verify_verdict: String,
    #[serde(default)]
    pub verify_feedback: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
    #[serde(default)]
    pub started_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub finished_at: String,
}

fn default_domain_general() -> String {
    "general".to_string()
}

fn default_wave_one() -> usize {
    1
}

fn default_task_pending() -> String {
    "pending".to_string()
}

/// 上游任务产出的结构化接口/表结构契约（自动注入下游依赖任务，替代全文堆砌防爆上下文）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamContract {
    pub id: String,
    pub run_id: String,
    pub from_task_id: String,
    pub domain: String,
    pub title: String,
    pub schema_or_api: String,
    #[serde(default)]
    pub notes: String,
    pub created_at: String,
}

/// 只追加事件账本（Append-Only Ledger），用于前端实时追踪波次、契约发布、跨项目问询与返工原因。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamEvent {
    pub seq: i64,
    pub run_id: String,
    pub ts: String,
    pub kind: String,
    #[serde(default)]
    pub task_id: String,
    #[serde(default)]
    pub actor: String,
    pub summary: String,
    #[serde(default)]
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamRunDetail {
    pub run: TeamRun,
    pub tasks: Vec<TeamTask>,
    pub contracts: Vec<TeamContract>,
    pub events: Vec<TeamEvent>,
}

pub(crate) fn now_ts() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

pub(crate) fn new_team_id(prefix: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tail = &uuid::Uuid::new_v4().simple().to_string()[..6];
    format!("{prefix}-{ts:x}-{tail}")
}

pub(crate) fn normalize_domain(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase();
    if lower.contains("sql")
        || lower.contains("db")
        || lower.contains("database")
        || lower.contains("数据库")
    {
        "sql".to_string()
    } else if lower.contains("back")
        || lower.contains("server")
        || lower.contains("api")
        || lower.contains("后端")
    {
        "backend".to_string()
    } else if lower.contains("front")
        || lower.contains("web")
        || lower.contains("ui")
        || lower.contains("前端")
    {
        "frontend".to_string()
    } else {
        "general".to_string()
    }
}

pub(crate) fn domain_badge_cn(domain: &str) -> &'static str {
    match domain {
        "sql" => "SQL/数据库",
        "backend" => "后端服务",
        "frontend" => "前端界面",
        _ => "通用任务",
    }
}

pub(crate) fn truncate_chars_safe(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let head_len = (max_chars * 2) / 3;
    let tail_len = max_chars.saturating_sub(head_len + 32);
    let total = trimmed.chars().count();
    let head: String = trimmed.chars().take(head_len).collect();
    let tail: String = trimmed
        .chars()
        .skip(total.saturating_sub(tail_len))
        .collect();
    format!("{head}\n...[truncated]...\n{tail}")
}

/// 校验并计算 DAG 执行波次（Kahn 拓扑分层）。
/// 自动剔除不存在的 `depends_on` ID 与自依赖；若检测到环路则返回 `Err`。
pub fn compute_and_assign_waves(
    tasks: &mut [TeamTask],
) -> Result<BTreeMap<usize, Vec<String>>, String> {
    if tasks.is_empty() {
        return Ok(BTreeMap::new());
    }

    let valid_ids: HashSet<String> = tasks.iter().map(|t| t.id.clone()).collect();
    let order_index: HashMap<String, usize> = tasks
        .iter()
        .enumerate()
        .map(|(idx, t)| (t.id.clone(), idx))
        .collect();

    let mut indegree: HashMap<String, usize> = HashMap::new();
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

    for task in tasks.iter_mut() {
        let mut seen = HashSet::new();
        task.depends_on
            .retain(|dep| dep != &task.id && valid_ids.contains(dep) && seen.insert(dep.clone()));
        indegree.insert(task.id.clone(), task.depends_on.len());
        for dep in &task.depends_on {
            dependents
                .entry(dep.clone())
                .or_default()
                .push(task.id.clone());
        }
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    let mut wave_by_id: HashMap<String, usize> = HashMap::new();

    let mut initial_roots: Vec<String> = indegree
        .iter()
        .filter_map(|(id, &deg)| (deg == 0).then(|| id.clone()))
        .collect();
    initial_roots.sort_by_key(|id| order_index.get(id).copied().unwrap_or(usize::MAX));

    for root in initial_roots {
        wave_by_id.insert(root.clone(), 1);
        queue.push_back(root);
    }

    let mut visited = 0usize;
    while let Some(cur) = queue.pop_front() {
        visited += 1;
        let cur_wave = wave_by_id.get(&cur).copied().unwrap_or(1);
        if let Some(children) = dependents.get(&cur) {
            let mut sorted_children = children.clone();
            sorted_children.sort_by_key(|id| order_index.get(id).copied().unwrap_or(usize::MAX));
            for child in sorted_children {
                let next_wave = wave_by_id
                    .get(&child)
                    .copied()
                    .unwrap_or(1)
                    .max(cur_wave + 1);
                wave_by_id.insert(child.clone(), next_wave);
                if let Some(deg) = indegree.get_mut(&child) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(child);
                    }
                }
            }
        }
    }

    if visited != tasks.len() {
        return Err("任务依赖存在循环引用（DAG 环路），请移除互相等待的前置依赖".to_string());
    }

    let mut waves: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (idx, task) in tasks.iter_mut().enumerate() {
        let w = wave_by_id.get(&task.id).copied().unwrap_or(1);
        task.wave = w;
        task.sort_order = idx;
        waves.entry(w).or_default().push(task.id.clone());
    }
    Ok(waves)
}
