//! 数据结构、时间格式与项目标识辅助函数。

use std::path::Path;

use chrono::Local;
use serde::{Deserialize, Serialize};

pub(crate) const MAX_NON_PINNED_PROFILES: usize = 40;
pub(crate) const MAX_L3_EVIDENCE_CHARS: usize = 3600;
pub(crate) const MAX_RECALL_RESULT_CHARS: usize = 9000;
pub(crate) const MAX_RAW_TURNS_PER_AGENT: usize = 2000;
pub(crate) const MAX_TIMELINE_PER_AGENT: usize = 2000;
pub(crate) const MAX_NODES_PER_AGENT: usize = 500;
pub(crate) const DISTILL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(45);

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
    pub project_key: String,
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
}

pub(crate) fn default_task_status() -> String {
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
