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
    /// 全局全角色提示词模板自定义覆写（template_id -> 自定义模板；缺失或空值表示使用内置默认模板）
    #[serde(default)]
    pub prompt_templates: BTreeMap<String, String>,
    /// 启用精简提示词模式（去除重复说教与交叉冗余，SQL 仅提取 DDL 块；默认开启）
    #[serde(default = "default_true")]
    pub compact_prompts: bool,
    /// 主控经理（鲸鱼管家）默认绑定的专属记忆智能体 ID（默认 `"none"`）
    #[serde(default = "default_agent_none")]
    pub default_lead_agent_id: String,
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
            prompt_templates: BTreeMap::new(),
            compact_prompts: true,
            default_lead_agent_id: "none".to_string(),
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

/// 用户自主新增的跨团队项目卡片（不限制固定领域，可自由增删改与挑选卡通形象）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamProjectPreset {
    #[serde(default)]
    pub id: String,
    pub name: String,
    #[serde(default = "default_domain_general")]
    pub domain: String,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default = "default_agent_none")]
    pub agent_id: String,
    #[serde(default)]
    pub avatar_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub sort_order: usize,
    #[serde(default)]
    pub updated_at: String,
}

fn default_agent_none() -> String {
    "none".to_string()
}

/// 随新建任务附带的截图/原型参考图（支持 Ctrl+V 粘贴上传）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamImage {
    #[serde(alias = "media_type")]
    pub media_type: String,
    pub data: String,
}

/// 一次完整的跨项目协作任务单（支持 `pipeline` 四阶段研发流 与 `triage` 跨库联合排障会诊）。
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
    /// 五阶段编排状态：`"planning"` | `"sql_binding"` | `"sql_designing"` | `"awaiting_approval"` | `"team_binding"` | `"contracting"` | `"executing"` | `"completed"` | `"triage"`
    #[serde(default)]
    pub stage: String,
    /// 本次任务是否涉及新建/修改数据库表结构
    #[serde(default)]
    pub requires_db: bool,
    /// 阶段 1：主控生成的整体《执行计划》
    #[serde(default)]
    pub execution_plan: String,
    /// 阶段 2：SQL 智能体设计的《表结构 DDL + 字段说明》
    #[serde(default)]
    pub sql_schema_draft: String,
    /// 阶段 3：主控基于已确认表结构约定的《前后端统一接口、字典与权限方案》
    #[serde(default)]
    pub interface_spec: String,
    /// 阶段 4 交付汇总：可直接复制执行的达梦/SQL DDL 脚本
    #[serde(default)]
    pub delivery_sql: String,
    /// 阶段 2 SQL 架构师专属绑定与持久会话（防切换工作目录后被旧项目 .meta 劫持）
    #[serde(default)]
    pub sql_working_dir: String,
    #[serde(default = "default_agent_none")]
    pub sql_agent_id: String,
    #[serde(default)]
    pub sql_model: String,
    #[serde(default)]
    pub sql_session_id: String,
    /// 本单级各角色提示词覆写（template_id -> 自定义模板内容）
    #[serde(default)]
    pub prompt_overrides: BTreeMap<String, String>,
    /// 新建任务附带的截图/效果图（支持视觉识别与前后端还原）
    #[serde(default)]
    pub goal_images: Vec<TeamImage>,
    /// 乐观锁递增版本号（CAS 防旧快照覆盖）
    #[serde(default)]
    pub revision: u64,
    #[serde(default = "default_true")]
    pub auto_verify: bool,
    #[serde(default = "default_max_repairs")]
    pub max_repairs: u32,
    #[serde(default)]
    pub lead_model: String,
    /// 本单主控经理（鲸鱼管家）绑定的专属记忆智能体 ID（默认 `"none"`）
    #[serde(default = "default_agent_none")]
    pub lead_agent_id: String,
    /// 当前所属迭代期号（从 1 开始递增）
    #[serde(default = "default_round_one")]
    pub current_round_no: u32,
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

fn default_round_one() -> u32 {
    1
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
    pub project_slot_id: String,
    #[serde(default)]
    pub project_name: String,
    #[serde(default)]
    pub avatar_id: String,
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
    /// 单成员级完整提示词覆写（非空时完全替换派发给该成员的主提示词）
    #[serde(default)]
    pub custom_prompt: String,
    /// 该成员专属参考图（例如不同前端各自的样式图；非空时优先于 run.goal_images 下发给该成员）
    #[serde(default)]
    pub images: Vec<TeamImage>,
    /// 本次定向改造/联动变更指令（非空时以增量改造模式唤醒既有会话执行）
    #[serde(default)]
    pub revision_note: String,
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
    /// `""` | `"passed"` | `"rejected"` | `"skipped"` | `"qa_rework"` | `"routed_rework"` | `"retest"`
    #[serde(default)]
    pub verify_verdict: String,
    #[serde(default)]
    pub verify_feedback: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
    /// 当前这一期单独记录的改动文件（开新一期时对选中成员清零；`touched_files` 保留累计全量）
    #[serde(default)]
    pub round_touched_files: Vec<String>,
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

/// 单期迭代中某个成员的执行快照（区分是否参与本期，并仅记录本期改动文件）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamRoundTaskSnapshot {
    pub id: String,
    #[serde(default = "default_domain_general")]
    pub domain: String,
    #[serde(default)]
    pub project_name: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub participated: bool,
    #[serde(default = "default_wave_one")]
    pub wave: usize,
    #[serde(default)]
    pub repair_count: u32,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default = "default_agent_none")]
    pub agent_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub instruction: String,
    #[serde(default)]
    pub contract_summary: String,
    #[serde(default)]
    pub last_output_excerpt: String,
    #[serde(default)]
    pub verify_verdict: String,
    #[serde(default)]
    pub verify_feedback: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
}

/// 团队整体分期迭代存档快照（第 1 期 / 第 2 期 …）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamRound {
    pub round_no: u32,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub requirement: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub requires_db: bool,
    #[serde(default)]
    pub execution_plan: String,
    #[serde(default)]
    pub sql_schema_draft: String,
    #[serde(default)]
    pub interface_spec: String,
    #[serde(default)]
    pub delivery_sql: String,
    #[serde(default)]
    pub synthesis_report: String,
    #[serde(default)]
    pub tasks: Vec<TeamRoundTaskSnapshot>,
    #[serde(default)]
    pub contracts: Vec<TeamContract>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub finished_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamRunDetail {
    pub run: TeamRun,
    pub tasks: Vec<TeamTask>,
    pub contracts: Vec<TeamContract>,
    pub events: Vec<TeamEvent>,
    #[serde(default)]
    pub rounds: Vec<TeamRound>,
}

pub(crate) fn clean_path_str(raw: &str) -> String {
    let mut s = raw.trim().replace('\\', "/");
    if let Some(rest) = s.strip_prefix("//?/UNC/") {
        s = format!("//{rest}");
    } else if let Some(rest) = s.strip_prefix("//?/") {
        s = rest.to_string();
    }
    while s.len() > 3 && s.ends_with('/') {
        s.pop();
    }
    s
}

pub(crate) fn path_to_clean_str(p: &std::path::Path) -> String {
    clean_path_str(&p.to_string_lossy())
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
        || lower.contains("达梦")
    {
        "sql".to_string()
    } else if lower.contains("back")
        || lower.contains("server")
        || lower.contains("api")
        || lower.contains("java")
        || lower.contains("rust")
        || lower.contains("后端")
    {
        "backend".to_string()
    } else if lower.contains("front")
        || lower.contains("web")
        || lower.contains("ui")
        || lower.contains("vue")
        || lower.contains("react")
        || lower.contains("前端")
    {
        "frontend".to_string()
    } else if lower.contains("mobile")
        || lower.contains("app")
        || lower.contains("flutter")
        || lower.contains("uniapp")
        || lower.contains("小程序")
        || lower.contains("移动")
    {
        "mobile".to_string()
    } else if lower.contains("qa")
        || lower.contains("test")
        || lower.contains("测试")
        || lower.contains("验收")
    {
        "qa".to_string()
    } else {
        "general".to_string()
    }
}

pub(crate) fn domain_badge_cn(domain: &str) -> &'static str {
    match domain {
        "sql" => "SQL/数据库",
        "backend" => "后端服务",
        "frontend" => "前端界面",
        "mobile" => "移动端/小程序",
        "qa" => "测试/验收",
        _ => "通用项目",
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
