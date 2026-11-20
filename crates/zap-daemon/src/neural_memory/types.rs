//! 数据结构、5 抽屉记忆模型、时间格式与项目标识辅助函数。

use std::path::Path;

use chrono::Local;
use serde::{Deserialize, Serialize};

pub(crate) const MAX_NON_PINNED_PROFILES: usize = 60;
pub(crate) const MAX_L3_EVIDENCE_CHARS: usize = 3600;
pub(crate) const MAX_RECALL_RESULT_CHARS: usize = 9000;
pub(crate) const MAX_RAW_TURNS_PER_AGENT: usize = 2000;
pub(crate) const MAX_TIMELINE_PER_AGENT: usize = 1200;
pub(crate) const MAX_NODES_PER_AGENT: usize = 500;
pub(crate) const DISTILL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(45);
/// 草稿攒批整理：同一 `(agent, session)` 静默等待秒数。
pub(crate) const DRAFT_DEBOUNCE_SECS: u64 = 90;
/// 草稿攒批整理：同一 `(agent, session)` 累计达到多少轮自动触发一次阶段整理。
pub(crate) const DRAFT_BATCH_TRIGGER_TURNS: usize = 5;

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

pub(crate) fn default_active_agent() -> String {
    "none".to_string()
}

pub(crate) fn default_distill_mode() -> String {
    "llm_auto".to_string()
}

pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn default_active_status() -> String {
    "active".to_string()
}

pub(crate) fn default_profile_kind() -> String {
    "rule".to_string()
}

pub(crate) fn default_task_item_kind() -> String {
    "requirement".to_string()
}

pub(crate) fn default_fact_kind() -> String {
    "fact".to_string()
}

pub(crate) fn default_link_relation() -> String {
    "related".to_string()
}

pub(crate) fn default_journal_kind() -> String {
    "work".to_string()
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

/// 抽屉 1：长期规矩、习惯与业务暗口径（支持作废替代与使用率热度）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileFact {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub project_key: String,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub source_turn_id: String,
    /// `"rule"` | `"preference"` | `"domain_convention"` | `"study_pref"`
    #[serde(default = "default_profile_kind")]
    pub kind: String,
    /// `"active"` | `"superseded"`
    #[serde(default = "default_active_status")]
    pub status: String,
    #[serde(default)]
    pub superseded_by: String,
    #[serde(default)]
    pub invalid_at: String,
    #[serde(default)]
    pub hit_count: i64,
    #[serde(default)]
    pub last_used_at: String,
}

impl Default for ProfileFact {
    fn default() -> Self {
        Self {
            id: String::new(),
            text: String::new(),
            project: String::new(),
            project_key: String::new(),
            pinned: false,
            updated_at: String::new(),
            source_turn_id: String::new(),
            kind: default_profile_kind(),
            status: default_active_status(),
            superseded_by: String::new(),
            invalid_at: String::new(),
            hit_count: 0,
            last_used_at: String::new(),
        }
    }
}

impl ProfileFact {
    pub fn is_active(&self) -> bool {
        self.status.is_empty() || self.status == "active"
    }
}

/// 抽屉 2：需求与学习进度本（支持工作需求、学习计划、以及关联 Team 需求单号）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralTask {
    pub id: String,
    pub agent: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub project_key: String,
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
    /// `"requirement"` | `"study"` | `"bugfix"` | `"general"`
    #[serde(default = "default_task_item_kind")]
    pub item_kind: String,
    /// 下一步计划 / 明天该学啥 / 待联调收尾动作
    #[serde(default)]
    pub next_step: String,
    /// 若来自 Agent Teams 协作单，关联 `team_runs.id`
    #[serde(default)]
    pub team_run_id: String,
    /// 若来自 Agent Teams 子任务，关联 `team_tasks.id`
    #[serde(default)]
    pub team_task_id: String,
}

impl Default for NeuralTask {
    fn default() -> Self {
        Self {
            id: String::new(),
            agent: String::new(),
            project: String::new(),
            project_key: String::new(),
            working_dir: String::new(),
            title: String::new(),
            progress_note: String::new(),
            remaining_note: String::new(),
            touched_files: Vec::new(),
            session_id: String::new(),
            source_turn_ids: Vec::new(),
            status: default_task_status(),
            created_at: String::new(),
            updated_at: String::new(),
            done_at: String::new(),
            item_kind: default_task_item_kind(),
            next_step: String::new(),
            team_run_id: String::new(),
            team_task_id: String::new(),
        }
    }
}

pub(crate) fn default_task_status() -> String {
    "in_progress".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NodeEvent {
    #[serde(default)]
    pub ts: String,
    #[serde(default)]
    pub text: String,
}

/// 抽屉 3 卡片内的独立原子要点：支持增量补充、融合更新与单条标作废，杜绝“覆写失忆”。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeFact {
    pub id: String,
    /// `"fact"` | `"decision"` | `"pitfall"` | `"contract"` | `"study_note"`
    #[serde(default = "default_fact_kind")]
    pub kind: String,
    pub text: String,
    /// `"active"` | `"superseded"`
    #[serde(default = "default_active_status")]
    pub status: String,
    #[serde(default)]
    pub superseded_by: String,
    #[serde(default)]
    pub invalid_at: String,
    #[serde(default)]
    pub source_turn_id: String,
    #[serde(default)]
    pub updated_at: String,
}

impl Default for NodeFact {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: default_fact_kind(),
            text: String::new(),
            status: default_active_status(),
            superseded_by: String::new(),
            invalid_at: String::new(),
            source_turn_id: String::new(),
            updated_at: String::new(),
        }
    }
}

impl NodeFact {
    pub fn is_active(&self) -> bool {
        self.status.is_empty() || self.status == "active"
    }
}

/// 抽屉 3：知识与经验活卡片（对应脑图节点，向后兼容旧版脑图字段）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralNode {
    pub id: String,
    pub name: String,
    pub cat: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub project_key: String,
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
    /// 节点别名（换种说法也能归入同一张卡，防止碎片节点分裂）
    #[serde(default)]
    pub aliases: Vec<String>,
    /// 写入期由模型生成的双语/大白话检索标签
    #[serde(default)]
    pub search_tags: Vec<String>,
    /// 卡片内的多条演进式独立要点（支持新增/融合/标作废）
    #[serde(default)]
    pub facts: Vec<NodeFact>,
    /// 历史命中引用次数（越常帮上忙越靠前）
    #[serde(default)]
    pub hit_count: i64,
    #[serde(default)]
    pub last_used_at: String,
}

impl Default for NeuralNode {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            cat: String::new(),
            project: String::new(),
            project_key: String::new(),
            status: default_node_status(),
            mastery: default_mastery(),
            tier: 0,
            windows: default_windows(),
            summary: String::new(),
            events: Vec::new(),
            updated_at: String::new(),
            aliases: Vec::new(),
            search_tags: Vec::new(),
            facts: Vec::new(),
            hit_count: 0,
            last_used_at: String::new(),
        }
    }
}

impl NeuralNode {
    /// 获取当前仍生效的独立要点切片。
    pub fn active_facts(&self) -> Vec<&NodeFact> {
        self.facts.iter().filter(|f| f.is_active()).collect()
    }
}

pub(crate) fn default_node_status() -> String {
    "active".to_string()
}

pub(crate) fn default_mastery() -> i64 {
    75
}

pub(crate) fn default_windows() -> Vec<String> {
    vec!["all".into(), "today".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralLink {
    pub source: String,
    pub target: String,
    /// `"related"` | `"depends_on"` | `"causes"` | `"part_of"`
    #[serde(default = "default_link_relation")]
    pub relation: String,
}

impl Default for NeuralLink {
    fn default() -> Self {
        Self {
            source: String::new(),
            target: String::new(),
            relation: default_link_relation(),
        }
    }
}

/// 抽屉 4：每日精华大事记（阶段整理产出，不再记录碎嘴和降级废话）。
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
    pub project_key: String,
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
    /// `"work"` | `"study"` | `"milestone"` | `"team"`
    #[serde(default = "default_journal_kind")]
    pub kind: String,
    #[serde(default)]
    pub team_run_id: String,
    /// 本条大事记关联的脑图卡片 ID 列表
    #[serde(default)]
    pub node_ids: Vec<String>,
}

impl Default for TimelineEvent {
    fn default() -> Self {
        Self {
            id: String::new(),
            ts: String::new(),
            agent: String::new(),
            project: String::new(),
            project_key: String::new(),
            working_dir: String::new(),
            session_id: String::new(),
            turn_id: String::new(),
            raw_file: String::new(),
            user_goal: String::new(),
            summary: String::new(),
            ai_outcome: String::new(),
            keywords: Vec::new(),
            touched_files: Vec::new(),
            distilled_by: String::new(),
            kind: default_journal_kind(),
            team_run_id: String::new(),
            node_ids: Vec::new(),
        }
    }
}

/// 抽屉 5：原始对话冷仓库（仅存档留底与按需溯源，不建暴力 FTS 全文索引）。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RawTurn {
    pub turn_id: String,
    pub ts: String,
    pub agent: String,
    #[serde(default)]
    pub project: String,
    #[serde(default)]
    pub project_key: String,
    #[serde(default)]
    pub working_dir: String,
    #[serde(default)]
    pub session_id: String,
    pub user_text: String,
    pub ai_text: String,
    #[serde(default)]
    pub touched_files: Vec<String>,
}

/// 精确路牌索引条目：将文件、表名、接口、符号、错误码、Team 单号直接映射到记忆实体。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MemoryAnchor {
    pub(crate) agent_id: String,
    pub(crate) project_key: String,
    /// `"node"` | `"task"` | `"profile"` | `"timeline"`
    pub(crate) target_kind: String,
    pub(crate) target_id: String,
    /// `"file"` | `"symbol"` | `"table"` | `"endpoint"` | `"error_code"` | `"module"` | `"team_run"`
    pub(crate) anchor_type: String,
    pub(crate) anchor_value: String,
    pub(crate) anchor_norm: String,
}

/// 持久化草稿队列记录：聊天当下轻量落盘，按自然触发点打包整理，软件重启绝不丢记忆。
#[derive(Debug, Clone)]
pub(crate) struct DraftTurn {
    pub(crate) id: i64,
    pub(crate) turn_id: String,
    pub(crate) session_id: String,
    pub(crate) agent_id: String,
    pub(crate) project: String,
    pub(crate) project_key: String,
    pub(crate) working_dir: String,
    pub(crate) user_text: String,
    pub(crate) ai_text: String,
    pub(crate) touched_files: Vec<String>,
    pub(crate) provider_name: String,
    pub(crate) explicit_remember: bool,
    pub(crate) retry_count: i64,
    pub(crate) created_at: String,
}

pub(crate) fn now_ts_minute() -> String {
    Local::now().format("%Y-%m-%d %H:%M").to_string()
}

pub(crate) fn now_iso() -> String {
    Local::now().to_rfc3339()
}

pub(crate) fn new_id(prefix: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let rand_tail = &uuid::Uuid::new_v4().simple().to_string()[..6];
    format!("{prefix}-{ts:x}-{rand_tail}")
}

/// 供界面展示的项目短名（取工作目录末级目录名）。
pub(crate) fn normalize_project_name(working_dir: &Path) -> String {
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

/// 稳定的项目隔离键：规范化工作目录完整路径，防止同名目录（如两个 `server`）串记忆。
pub(crate) fn normalize_project_key(working_dir: &Path) -> String {
    let raw = working_dir.to_string_lossy();
    normalize_project_key_str(raw.as_ref())
}

pub(crate) fn normalize_project_key_str(raw: &str) -> String {
    let trimmed = raw.trim().replace('\\', "/");
    let trimmed = trimmed.trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    #[cfg(windows)]
    {
        trimmed.to_lowercase()
    }
    #[cfg(not(windows))]
    {
        // Windows 盘符路径（例如测试里传入的 D:/work/...）也统一小写化
        if trimmed.len() >= 2 && trimmed.as_bytes()[1] == b':' {
            trimmed.to_lowercase()
        } else {
            trimmed.to_string()
        }
    }
}
