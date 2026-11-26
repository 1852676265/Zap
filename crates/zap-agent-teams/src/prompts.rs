//! `zap-agent-teams` 全角色提示词模板注册中心、目录特征感知器与单遍安全渲染器。
//!
//! 优先级顺序（高 -> 低）：
//! 1. 单成员级完整覆写（`TeamTask.custom_prompt`，非空时直接生效）
//! 2. 本单级角色覆写（`TeamRun.prompt_overrides[template_id]`）
//! 3. 全局提示词模板（`TeamGlobalConfig.prompt_templates[template_id]`）
//! 4. 内置默认模板（`PROMPT_REGISTRY`）

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use serde::Serialize;

use super::types::{truncate_chars_safe, TeamContract, TeamProjectPreset, TeamRun, TeamTask};

pub const TPL_LEAD_PLANNER_PIPELINE: &str = "lead_planner_pipeline";
pub const TPL_LEAD_PLANNER_TRIAGE: &str = "lead_planner_triage";
pub const TPL_LEAD_INTERFACE_SPEC: &str = "lead_interface_spec";
pub const TPL_SQL_DESIGN_INITIAL: &str = "sql_design_initial";
pub const TPL_SQL_DESIGN_REVISE: &str = "sql_design_revise";
pub const TPL_SYS_GUARD_READONLY: &str = "sys_guard_readonly";
pub const TPL_SYS_GUARD_COLLAB: &str = "sys_guard_collab";
pub const TPL_WORKER_PIPELINE: &str = "worker_pipeline";
pub const TPL_WORKER_TRIAGE: &str = "worker_triage";
pub const TPL_WORKER_REVISION: &str = "worker_revision";
pub const TPL_MEMBER_NOTE: &str = "member_note";
pub const TPL_VERIFIER_GATE: &str = "verifier_gate";
pub const TPL_WORKER_REPAIR: &str = "worker_repair";
pub const TPL_TRIAGE_SYNTHESIS: &str = "triage_synthesis";
pub const TPL_PEER_CONSULTATION: &str = "peer_consultation";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PromptDomainScope {
    None,
    Sql,
    Member,
    Peer,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTemplateMeta {
    pub id: &'static str,
    pub group: &'static str,
    pub domain_scope: PromptDomainScope,
    pub group_label: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub placeholders: &'static [&'static str],
    pub default_template: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum DirProfileKind {
    Missing,
    Empty,
    SqlOnly,
    Codebase,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirProfile {
    pub kind: DirProfileKind,
    pub working_dir: String,
    pub sql_count: usize,
    pub code_count: usize,
    pub scanned_files: usize,
    pub scanned_dirs: usize,
    pub truncated: bool,
    pub sample_sql_files: Vec<String>,
    pub sample_code_files: Vec<String>,
}

fn domain_label(domain: &str) -> &str {
    match domain {
        "sql" => "SQL",
        "backend" => "Backend",
        "frontend" => "Frontend",
        "mobile" => "Mobile",
        "qa" => "QA",
        "general" => "General",
        other => other,
    }
}

impl DirProfile {
    pub fn to_hint(&self) -> String {
        let dir_label = if self.working_dir.is_empty() {
            "(unspecified)"
        } else {
            &self.working_dir
        };
        match self.kind {
            DirProfileKind::Missing => {
                format!("Dir `{dir_label}`: unavailable/missing; do not assume repo structure.")
            }
            DirProfileKind::Empty if self.scanned_files == 0 && !self.truncated => {
                format!("Dir `{dir_label}`: empty (0 files); design from requirements & memory.")
            }
            DirProfileKind::Empty => format!(
                "Dir `{dir_label}`: no SQL/source in sample (scanned {}, does not establish entire dir is empty).",
                self.scanned_files
            ),
            DirProfileKind::SqlOnly => format!(
                "Dir `{dir_label}`: SQL-only (observed SQL: {}; samples: [{}]). Read SQL scripts only; do not assume app source.",
                self.sql_count,
                self.sample_sql_files.join(", ")
            ),
            DirProfileKind::Codebase => {
                let samples: Vec<String> = self
                    .sample_code_files
                    .iter()
                    .chain(self.sample_sql_files.iter())
                    .take(6)
                    .cloned()
                    .collect();
                format!(
                    "Dir `{dir_label}`: codebase (observed SQL: {}, source: {}; samples: [{}]).",
                    self.sql_count,
                    self.code_count,
                    samples.join(", ")
                )
            }
        }
    }
}

pub(crate) fn extract_compact_sql_ddl(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut blocks = Vec::new();
    let lower = trimmed.to_ascii_lowercase();
    let mut offset = 0usize;
    while let Some(rel_start) = lower[offset..].find("```sql") {
        let fence_start = offset + rel_start + 6;
        if let Some(rel_end) = trimmed[fence_start..].find("```") {
            let block = trimmed[fence_start..fence_start + rel_end].trim();
            if !block.is_empty() {
                blocks.push(block.to_string());
            }
            offset = fence_start + rel_end + 3;
        } else {
            break;
        }
    }
    if !blocks.is_empty() {
        return blocks.join("\n\n");
    }
    let sql_lines: Vec<&str> = trimmed
        .lines()
        .filter(|line| {
            let t = line.trim();
            let u = t.to_ascii_uppercase();
            u.starts_with("CREATE ")
                || u.starts_with("ALTER ")
                || u.starts_with("COMMENT ON ")
                || u.starts_with("INSERT INTO ")
                || u.starts_with("DROP ")
                || (!t.starts_with('#')
                    && !t.starts_with('|')
                    && (t.ends_with(',') || t.ends_with(';') || t == ");"))
        })
        .collect();
    if sql_lines.len() >= 3 {
        sql_lines.join("\n")
    } else {
        trimmed.to_string()
    }
}

fn collapse_blank_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 1 && !out.is_empty() {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line.trim_end());
        }
    }
    out.trim().to_string()
}

/// Bounded read-only sample: depth 3, 400 files, 100 directories, 1000 entries per directory.
/// Symlinks are not followed; kinds describe observations, not exhaustive absence proofs.
pub fn inspect_dir_profile(dir: &Path) -> DirProfile {
    let dir_str = super::types::path_to_clean_str(dir);
    if dir.as_os_str().is_empty()
        || std::fs::symlink_metadata(dir)
            .map(|m| !m.is_dir() || m.file_type().is_symlink())
            .unwrap_or(true)
        || dir.ancestors().any(|p| {
            !p.as_os_str().is_empty()
                && std::fs::symlink_metadata(p)
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
        })
    {
        return DirProfile {
            kind: DirProfileKind::Missing,
            working_dir: dir_str,
            sql_count: 0,
            code_count: 0,
            scanned_files: 0,
            scanned_dirs: 0,
            truncated: false,
            sample_sql_files: Vec::new(),
            sample_code_files: Vec::new(),
        };
    }

    let mut sql_count = 0usize;
    let mut code_count = 0usize;
    let mut sample_sql_files = Vec::new();
    let mut sample_code_files = Vec::new();
    let mut visited = 0usize;
    let mut scanned_dirs = 0usize;
    let mut truncated = false;
    let mut stack = vec![(dir.to_path_buf(), 0usize)];

    while let Some((cur, depth)) = stack.pop() {
        if visited >= 400 || scanned_dirs >= 100 {
            truncated = true;
            break;
        }
        scanned_dirs += 1;
        let Ok(entries) = std::fs::read_dir(&cur) else {
            truncated = true;
            continue;
        };
        // Never select a nondeterministic prefix of readdir. Oversized directories
        // are omitted entirely, with an explicit incomplete flag.
        let mut sorted = Vec::new();
        let mut entry_count = 0;
        for entry in entries.take(1001) {
            entry_count += 1;
            match entry {
                Ok(entry) => sorted.push(entry),
                Err(_) => truncated = true,
            }
        }
        if entry_count > 1000 {
            truncated = true;
            continue;
        }
        sorted.sort_by_key(|entry| entry.file_name());
        let mut children = Vec::new();
        for entry in sorted {
            if visited >= 400 {
                truncated = true;
                break;
            }
            let p = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.')
                || matches!(
                    name.as_str(),
                    "node_modules" | "target" | "dist" | "build" | "vendor" | "__pycache__"
                )
            {
                continue;
            }
            let Ok(file_type) = entry.file_type() else {
                truncated = true;
                continue;
            };
            if file_type.is_symlink() {
                truncated = true;
                continue;
            }
            if file_type.is_dir() {
                if depth < 3 && scanned_dirs + stack.len() + children.len() < 100 {
                    children.push((p, depth + 1));
                } else {
                    truncated = true;
                }
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            visited += 1;
            let rel = p
                .strip_prefix(dir)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "sql" {
                sql_count += 1;
                if sample_sql_files.len() < 8 {
                    sample_sql_files.push(rel);
                }
            } else if matches!(
                ext.as_str(),
                "java"
                    | "kt"
                    | "rs"
                    | "go"
                    | "py"
                    | "ts"
                    | "tsx"
                    | "js"
                    | "jsx"
                    | "vue"
                    | "cs"
                    | "php"
                    | "rb"
                    | "xml"
                    | "prisma"
                    | "graphql"
            ) || matches!(
                name.as_str(),
                "pom.xml" | "Cargo.toml" | "package.json" | "go.mod" | "build.gradle"
            ) {
                code_count += 1;
                if sample_code_files.len() < 8 {
                    sample_code_files.push(rel);
                }
            }
        }
        stack.extend(children.into_iter().rev());
    }
    sample_sql_files.sort();
    sample_code_files.sort();

    let kind = if code_count > 0 {
        DirProfileKind::Codebase
    } else if sql_count > 0 {
        DirProfileKind::SqlOnly
    } else {
        DirProfileKind::Empty
    };

    DirProfile {
        kind,
        working_dir: dir_str,
        sql_count,
        code_count,
        scanned_files: visited,
        scanned_dirs,
        truncated,
        sample_sql_files,
        sample_code_files,
    }
}

/// 单遍占位符替换（绝不对已替换值二次扫描，确保 Vue `{{ item }}` 等内容不会被误替换）。
pub fn render_template_single_pass(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len() + 256);
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + 2..];
        if let Some(end_rel) = after_open.find("}}") {
            let key_raw = &after_open[..end_rel];
            let key = key_raw.trim();
            let valid_key = !key.is_empty()
                && key.len() <= 48
                && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if valid_key {
                if let Some((_, val)) = vars.iter().find(|(k, _)| *k == key) {
                    out.push_str(val);
                    rest = &after_open[end_rel + 2..];
                    continue;
                }
            }
            out.push_str(&rest[start..start + 2 + end_rel + 2]);
            rest = &after_open[end_rel + 2..];
            continue;
        }
        out.push_str("{{");
        rest = after_open;
    }
    out.push_str(rest);
    out
}

pub fn get_prompt_meta(template_id: &str) -> Option<&'static PromptTemplateMeta> {
    PROMPT_REGISTRY.iter().find(|m| m.id == template_id)
}

pub fn list_prompt_metas() -> &'static [PromptTemplateMeta] {
    PROMPT_REGISTRY
}

/// 按优先级解析有效模板文本及其来源标记：`run_override` > `global_template` > `default`。
pub fn resolve_effective_template(
    template_id: &str,
    domain_hint: Option<&str>,
    run_overrides: Option<&BTreeMap<String, String>>,
    global_templates: &BTreeMap<String, String>,
) -> (String, &'static str) {
    let domain_key = domain_hint
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|d| format!("{template_id}:{d}"));

    if let Some(ov) = run_overrides {
        if let Some(dk) = domain_key.as_deref() {
            if let Some(v) = ov
                .get(dk)
                .map(String::as_str)
                .filter(|s| !s.trim().is_empty())
            {
                return (v.to_string(), "run_override");
            }
        }
        if let Some(v) = ov
            .get(template_id)
            .map(String::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            return (v.to_string(), "run_override");
        }
    }

    if let Some(dk) = domain_key.as_deref() {
        if let Some(v) = global_templates
            .get(dk)
            .map(String::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            return (v.to_string(), "global_template");
        }
    }
    if let Some(v) = global_templates
        .get(template_id)
        .map(String::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        return (v.to_string(), "global_template");
    }

    let def = get_prompt_meta(template_id)
        .map(|m| m.default_template)
        .unwrap_or("");
    (def.to_string(), "default")
}

const DEFAULT_LEAD_PLANNER_PIPELINE: &str = r#"Decompose the requirement into a multi-repo plan.
Rules:
- `requires_db`: true only if DB schema/table/column changes are needed (Stage 2 handles DDL, so omit `sql` tasks unless the request is SQL-only).
- Split concrete specialist tasks (`backend`, `frontend`, `mobile`, `qa`); keep `depends_on: []` between coding tasks so they run concurrently (only `qa` may depend on them).
- Write `title`, `execution_plan`, `description`, `acceptance_criteria` in the user's language, keeping descriptions crisp and actionable.
Output ONLY raw JSON (no markdown fences):
{"title":"<=36 chars","requires_db":true,"execution_plan":"Concise markdown: approach, DB need, per-project scope","tasks":[{"key":"t1","project_id":"id from Projects or empty","domain":"backend|frontend|mobile|qa|general|sql","title":"","description":"","acceptance_criteria":"","depends_on":[]}]}

Projects:
{{user_projects}}

Repos:
{{known_repos}}

Goal:
{{user_goal}}"#;

const DEFAULT_LEAD_PLANNER_TRIAGE: &str = r#"Decompose this incident into parallel read-only investigation tasks (`depends_on: []`) across relevant domains (`sql`, `backend`, `frontend`, `mobile`, `qa`, `general`).
Write text values in the user's language, keeping instructions concise.
Output ONLY raw JSON (no markdown fences):
{"title":"<=36 chars","requires_db":false,"execution_plan":"Concise investigation plan","tasks":[{"key":"t1","project_id":"id from Projects or empty","domain":"sql|backend|frontend|mobile|qa|general","title":"","description":"","acceptance_criteria":"","depends_on":[]}]}

Projects:
{{user_projects}}

Repos:
{{known_repos}}

Incident:
{{user_goal}}"#;

const DEFAULT_LEAD_INTERFACE_SPEC: &str = r#"Lock a concise, unambiguous shared API, dict, and permission spec in Chinese Markdown for concurrent development:
1. REST routes (method + full path, query/body/response fields in camelCase matching schema).
2. Shared `dict_type` / enum codes.
3. Menu/button permission strings.
4. Brief per-repo responsibilities.
Omit long prose.

Goal: {{user_goal}}
Plan: {{execution_plan}}
Schema:
{{sql_schema_draft}}
Members:
{{task_summary}}"#;

const DEFAULT_SQL_DESIGN_INITIAL: &str = r#"SQL Architect (read-only). {{dir_profile_hint}}
Design the schema for user approval without creating/modifying files.
Goal: {{user_goal}}
Plan: {{execution_plan}}
Output concise Chinese Markdown: (1) tables summary, (2) complete executable DDL in ```sql``` (keys, types, defaults, comments, dict SQL), (3) brief key-field table."#;

const DEFAULT_SQL_DESIGN_REVISE: &str = r#"Revise the SQL schema per feedback (read-only; do not write files). {{dir_profile_hint}}
Feedback: {{feedback}}
Previous schema:
{{sql_schema_draft}}
Output concise Chinese Markdown with a brief change note, updated ```sql``` DDL (respecting existing project dialect/audit conventions), and updated key-field summary."#;

const DEFAULT_SYS_GUARD_READONLY: &str = r#"TEAM READ-ONLY ({{working_dir}}): Inspect existing files and memory read-only; NEVER create, modify, or delete any file."#;

const DEFAULT_SYS_GUARD_COLLAB: &str = r#"TEAM COLLABORATION ({{working_dir}}): Focus strictly on your repo/domain. Follow locked SQL/API names verbatim. Use `team_ask_peer` for peer details and `team_publish_contract` when finalizing API/schema changes."#;

const DEFAULT_SYS_GUARD_QA_VERIFY: &str = r#"TEAM QA VERIFICATION GATE ({{working_dir}}): Inspect code across upstream repositories and run build/test/verification commands ONLY. NEVER modify, create, or patch business source files. If defects are found, report exact file paths, root causes, and attribute every defect to its responsible upstream teammate `task_id` so the orchestrator routes rework back to that specialist."#;

const DEFAULT_WORKER_PIPELINE: &str = r#"Task [{{domain_label}} | Wave {{wave}} | {{project_name}}]: {{task_title}}
Dir: {{working_dir}}
Goal: {{user_goal}}
Scope: {{task_description}}
Done when: {{acceptance_criteria}}

{{sql_schema_section}}

{{interface_spec_section}}

{{upstream_section}}

{{domain_boundary_section}}

{{contract_publish_section}}"#;

const DEFAULT_WORKER_TRIAGE: &str = r#"Incident Investigation [{{domain_label}}]: {{task_title}}
Dir: {{working_dir}}
Incident: {{user_goal}}
Focus: {{task_description}}
Done when: {{acceptance_criteria}}
Inspect read-only; cite exact paths, lines, SQL, and API fields (use `team_ask_peer` if needed). Output structured root-cause evidence."#;

const DEFAULT_WORKER_REVISION: &str = r#"Incremental Revision [{{domain_label}} | Wave {{wave}} | {{project_name}}]: {{task_title}}
Dir: {{working_dir}}
Revision Instruction (HIGHEST PRIORITY):
{{revision_note}}
{{member_images_hint}}

Previously Touched Files: {{touched_files}}
Original Scope Context: {{task_description}}
Done when: {{acceptance_criteria}}

{{upstream_section}}

{{interface_spec_section}}

{{sql_schema_section}}

{{contract_publish_section}}

Rule: Apply ONLY the requested revision on top of your existing code in this session. If upstream contracts above update API routes, parameters, or fields, strictly follow the latest upstream contracts over any older baseline spec."#;

const DEFAULT_MEMBER_NOTE: &str = r#"Follow-up [{{domain_label}} · {{task_title}} | {{working_dir}}]:
{{note}}
Execute using existing session context and report changes concisely."#;

const DEFAULT_VERIFIER_GATE: &str = r#"Independent QA Verifier. Pass (`"passed": true`) if core requirements, contract names/routes, and changes are sound; reject ONLY for missing requirements, mismatched contract fields/routes, or clear syntax/logic bugs.
Output ONLY JSON: {"passed":true/false,"summary":"One-sentence Chinese conclusion","repair_instructions":"Exact fixes if rejected, else empty"}

Task [{{domain}}]: {{task_title}}
Scope: {{task_description}}
Done when: {{acceptance_criteria}}
SQL: {{approved_sql}}
Spec: {{locked_interface_spec}}
Upstream: {{upstream_contracts}}
Published: {{published_contract}}
Files:
{{touched_files_snippets}}
Output:
{{worker_output}}"#;

const DEFAULT_WORKER_REPAIR: &str = r#"[Repair {{repair_attempt}}/{{max_repairs}} | {{working_dir}}]
Rejected: {{verifier_summary}}
Fixes required: {{repair_instructions}}
Done when: {{acceptance_criteria}}
Fix directly, re-check syntax/build, and update `team_publish_contract` if signatures changed."#;

const DEFAULT_TRIAGE_SYNTHESIS: &str = r#"Synthesize parallel repository findings into an actionable Chinese Markdown Root-Cause Report:
1. Root cause (failing layer(s), exact paths/fields).
2. Cross-layer evidence chain (UI -> API -> DB).
3. Ordered per-repo fixes.
Incident: {{user_goal}}
Findings:
{{findings}}"#;

const DEFAULT_PEER_CONSULTATION: &str = r#"Answer on behalf of peer [{{target_badge}}] concisely using exact paths, columns, or API fields from evidence (state explicitly if undefined).
Question: {{question}}
Contracts: {{contracts_text}}
Peer excerpt: {{peer_excerpt}}
Code evidence:
{{code_evidence}}"#;

pub const PROMPT_REGISTRY: &[PromptTemplateMeta] = &[
    PromptTemplateMeta {
        id: TPL_LEAD_PLANNER_PIPELINE,
        domain_scope: PromptDomainScope::None,
        group: "planner",
        group_label: "主控规划与契约",
        title: "主控 · 流水线拆单与计划 (阶段 1)",
        description: "分析需求与项目卡片库，判定是否需改表并拆解前后端/移动端并发分工（需保留 JSON 输出结构）",
        placeholders: &["user_goal", "user_projects", "known_repos", "current_working_dir"],
        default_template: DEFAULT_LEAD_PLANNER_PIPELINE,
    },
    PromptTemplateMeta {
        id: TPL_LEAD_PLANNER_TRIAGE,
        domain_scope: PromptDomainScope::None,
        group: "planner",
        group_label: "主控规划与契约",
        title: "主控 · 跨库联合会诊拆单",
        description: "将线上故障现象拆解为多仓并发只读勘察任务（需保留 JSON 输出结构）",
        placeholders: &["user_goal", "user_projects", "known_repos", "current_working_dir"],
        default_template: DEFAULT_LEAD_PLANNER_TRIAGE,
    },
    PromptTemplateMeta {
        id: TPL_LEAD_INTERFACE_SPEC,
        domain_scope: PromptDomainScope::None,
        group: "planner",
        group_label: "主控规划与契约",
        title: "主控 · 统一接口与权限方案 (阶段 3)",
        description: "根据已确认表结构锁定 REST 路径、请求响应字段、字典码值与权限标识",
        placeholders: &["user_goal", "execution_plan", "sql_schema_draft", "task_summary"],
        default_template: DEFAULT_LEAD_INTERFACE_SPEC,
    },
    PromptTemplateMeta {
        id: TPL_SQL_DESIGN_INITIAL,
        domain_scope: PromptDomainScope::Sql,
        group: "sql",
        group_label: "SQL 表结构架构师",
        title: "SQL 架构师 · 初次设计 DDL (阶段 2)",
        description: "根据绑定目录特征（纯 SQL 脚本目录 / 空目录 / 代码仓）只读设计表结构 DDL",
        placeholders: &["user_goal", "execution_plan", "working_dir", "dir_profile_hint"],
        default_template: DEFAULT_SQL_DESIGN_INITIAL,
    },
    PromptTemplateMeta {
        id: TPL_SQL_DESIGN_REVISE,
        domain_scope: PromptDomainScope::Sql,
        group: "sql",
        group_label: "SQL 表结构架构师",
        title: "SQL 架构师 · 依据修改意见修订 DDL",
        description: "审稿退回修改时向 SQL 架构师派发的只读修订指令",
        placeholders: &["feedback", "sql_schema_draft", "user_goal", "execution_plan", "working_dir", "dir_profile_hint"],
        default_template: DEFAULT_SQL_DESIGN_REVISE,
    },
    PromptTemplateMeta {
        id: TPL_SYS_GUARD_READONLY,
        domain_scope: PromptDomainScope::None,
        group: "worker",
        group_label: "执行成员与系统规则",
        title: "系统守护规则 · 只读勘察模式",
        description: "注入到 SQL 设计与会诊只读子会话的 System Guard，约束禁止写入文件",
        placeholders: &["working_dir", "session_id", "dir_profile_hint"],
        default_template: DEFAULT_SYS_GUARD_READONLY,
    },
    PromptTemplateMeta {
        id: TPL_SYS_GUARD_COLLAB,
        domain_scope: PromptDomainScope::None,
        group: "worker",
        group_label: "执行成员与系统规则",
        title: "系统守护规则 · 协同开发模式",
        description: "注入到后端/前端/移动端等编码子会话的 System Guard，约束遵守锁定契约",
        placeholders: &["working_dir", "session_id"],
        default_template: DEFAULT_SYS_GUARD_COLLAB,
    },
    PromptTemplateMeta {
        id: TPL_WORKER_PIPELINE,
        domain_scope: PromptDomainScope::Member,
        group: "worker",
        group_label: "执行成员与系统规则",
        title: "研发成员 · 流水线子任务提示词",
        description: "派发给后端、前端、移动端、测试等各端研发子会话的主提示词模板（亦可在单个成员上独占覆写）",
        placeholders: &[
            "user_goal",
            "wave",
            "domain",
            "domain_label",
            "project_name",
            "working_dir",
            "task_title",
            "task_description",
            "acceptance_criteria",
            "sql_schema_section",
            "interface_spec_section",
            "upstream_section",
            "domain_boundary_section",
            "contract_publish_section",
        ],
        default_template: DEFAULT_WORKER_PIPELINE,
    },
    PromptTemplateMeta {
        id: TPL_WORKER_TRIAGE,
        domain_scope: PromptDomainScope::Member,
        group: "worker",
        group_label: "执行成员与系统规则",
        title: "排障成员 · 跨库只读排查提示词",
        description: "派发给联合会诊各成员的只读勘察主提示词模板",
        placeholders: &[
            "user_goal",
            "domain",
            "domain_label",
            "project_name",
            "working_dir",
            "task_title",
            "task_description",
            "acceptance_criteria",
        ],
        default_template: DEFAULT_WORKER_TRIAGE,
    },
    PromptTemplateMeta {
        id: TPL_WORKER_REVISION,
        domain_scope: PromptDomainScope::Member,
        group: "worker",
        group_label: "执行成员与系统规则",
        title: "研发成员 · 定向改造与联动变更提示词",
        description: "对已完成或已有代码的成员发起局部微调或上游变更联动同步时，在原会话上下文中执行的增量改造模板",
        placeholders: &[
            "user_goal",
            "wave",
            "domain",
            "domain_label",
            "project_name",
            "working_dir",
            "task_title",
            "task_description",
            "revision_note",
            "member_images_hint",
            "touched_files",
            "acceptance_criteria",
            "sql_schema_section",
            "interface_spec_section",
            "upstream_section",
            "contract_publish_section",
        ],
        default_template: DEFAULT_WORKER_REVISION,
    },
    PromptTemplateMeta {
        id: TPL_MEMBER_NOTE,
        domain_scope: PromptDomainScope::Member,
        group: "worker",
        group_label: "执行成员与系统规则",
        title: "成员抽屉 · 直达补充指令",
        description: "在工作台成员抽屉发送补充指令时直接唤醒原生子会话的提示词格式",
        placeholders: &["note", "task_title", "domain_label", "project_name", "working_dir"],
        default_template: DEFAULT_MEMBER_NOTE,
    },
    PromptTemplateMeta {
        id: TPL_VERIFIER_GATE,
        domain_scope: PromptDomainScope::Member,
        group: "verify",
        group_label: "验收、返工与跨库协同",
        title: "独立验收侦探 (Verifier Gate)",
        description: "波次完成后由独立只读模型复核改动文件与契约一致性（需保留 JSON 输出结构）",
        placeholders: &[
            "task_title",
            "domain",
            "task_description",
            "acceptance_criteria",
            "approved_sql",
            "locked_interface_spec",
            "upstream_contracts",
            "published_contract",
            "touched_files_snippets",
            "worker_output",
        ],
        default_template: DEFAULT_VERIFIER_GATE,
    },
    PromptTemplateMeta {
        id: TPL_WORKER_REPAIR,
        domain_scope: PromptDomainScope::Member,
        group: "verify",
        group_label: "验收、返工与跨库协同",
        title: "验收打回 · 自动返工指令",
        description: "独立验收未通过时发送回原开发子会话的定向修复提示词",
        placeholders: &[
            "repair_attempt",
            "max_repairs",
            "verifier_summary",
            "repair_instructions",
            "acceptance_criteria",
            "working_dir",
        ],
        default_template: DEFAULT_WORKER_REPAIR,
    },
    PromptTemplateMeta {
        id: TPL_TRIAGE_SYNTHESIS,
        domain_scope: PromptDomainScope::None,
        group: "verify",
        group_label: "验收、返工与跨库协同",
        title: "主控排障官 · 会诊根因汇总报告",
        description: "汇总多仓排查结果并输出跨层证据链与修复清单",
        placeholders: &["user_goal", "findings"],
        default_template: DEFAULT_TRIAGE_SYNTHESIS,
    },
    PromptTemplateMeta {
        id: TPL_PEER_CONSULTATION,
        domain_scope: PromptDomainScope::Peer,
        group: "verify",
        group_label: "验收、返工与跨库协同",
        title: "跨库只读问询应答 (team_ask_peer)",
        description: "子智能体调用 team_ask_peer 跨仓查询对方代码/契约时生成精准回答的提示词",
        placeholders: &["target_badge", "question", "contracts_text", "peer_excerpt", "code_evidence"],
        default_template: DEFAULT_PEER_CONSULTATION,
    },
];

pub(crate) fn compact_interface_spec(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // 兼容旧版 stage3 fallback 产物：剥离与 sql_schema_section 重复的“参考执行计划 / 已确认数据库表结构”长文段
    let mut cleaned = trimmed.to_string();
    if let Some(s2) = cleaned.find("#### 2. 已确认数据库表结构") {
        let end = cleaned[s2..]
            .find("\n#### 3.")
            .or_else(|| cleaned[s2..].find("\n#### 4."))
            .map(|off| s2 + off)
            .unwrap_or(cleaned.len());
        cleaned = format!("{}{}", cleaned[..s2].trim_end(), &cleaned[end..]);
    }
    if let (Some(s1), Some(s4)) = (
        cleaned.find("#### 1. 参考执行计划"),
        cleaned.find("\n#### 4."),
    ) {
        if s1 < s4 {
            cleaned = format!("{}{}", cleaned[..s1].trim_end(), &cleaned[s4..]);
        }
    }
    cleaned.trim().to_string()
}

pub(crate) fn build_worker_sections(
    run: &TeamRun,
    task: &TeamTask,
    has_dependents: bool,
    upstream_contracts: &[TeamContract],
    upstream_excerpts: &[(String, String)],
    compact: bool,
) -> (String, String, String, String, String) {
    let is_revision = !task.revision_note.trim().is_empty();
    let has_spec = !run.interface_spec.trim().is_empty();
    let skip_sql_for_client =
        compact && has_spec && matches!(task.domain.as_str(), "frontend" | "mobile" | "qa");
    let sql_schema_section = if run.sql_schema_draft.trim().is_empty() || skip_sql_for_client {
        String::new()
    } else if compact {
        let ddl = extract_compact_sql_ddl(&run.sql_schema_draft);
        format!("Approved DDL:\n{}", truncate_chars_safe(&ddl, 1600))
    } else {
        format!(
            "Approved database schema (match table and column names exactly):\n{}",
            truncate_chars_safe(&run.sql_schema_draft, 3200)
        )
    };

    let interface_spec_section = if !has_spec {
        String::new()
    } else if is_revision {
        let spec_clean = compact_interface_spec(&run.interface_spec);
        format!(
            "Historical Baseline Spec (reference only; latest revision/upstream contracts win):\n{}",
            truncate_chars_safe(&spec_clean, if compact { 1000 } else { 2200 })
        )
    } else if compact {
        let spec_clean = compact_interface_spec(&run.interface_spec);
        let max_len = if upstream_contracts.is_empty() {
            1600
        } else {
            1100
        };
        format!(
            "Locked API & permissions:\n{}",
            truncate_chars_safe(&spec_clean, max_len)
        )
    } else {
        format!(
            "Locked API, dictionary codes, and permissions (follow exactly; do not rename fields or URLs):\n{}",
            truncate_chars_safe(&run.interface_spec, 3600)
        )
    };

    let upstream_section = if !upstream_contracts.is_empty() {
        let max_ctr = if compact { 1200 } else { 3600 };
        let ctr_block = upstream_contracts
            .iter()
            .map(|c| {
                let body = truncate_chars_safe(&c.schema_or_api, max_ctr);
                if c.notes.trim().is_empty() || (compact && c.notes.chars().count() > 240) {
                    format!(
                        "### Upstream [{}]: {}\n{}",
                        domain_label(&c.domain),
                        c.title,
                        body
                    )
                } else {
                    format!(
                        "### Upstream [{}]: {}\n{}\nNotes: {}",
                        domain_label(&c.domain),
                        c.title,
                        body,
                        c.notes.trim()
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        format!("Upstream contracts:\n{ctr_block}")
    } else if !upstream_excerpts.is_empty() {
        let max_ex = if compact { 700 } else { 1500 };
        let ex_block = upstream_excerpts
            .iter()
            .map(|(k, v)| format!("### Upstream ({k})\n{}", truncate_chars_safe(v, max_ex)))
            .collect::<Vec<_>>()
            .join("\n\n");
        format!("Upstream summaries:\n{ex_block}")
    } else {
        String::new()
    };

    let domain_boundary_section = if task.domain == "backend"
        && !run.sql_schema_draft.trim().is_empty()
    {
        if compact {
            "Rule: User executes DDL manually; implement backend code matching approved schema."
                .to_string()
        } else {
            "Schema delivery boundary: the user approved the DDL/dictionary SQL and will copy and execute it from the delivery workbench. Implement backend business code matching the approved names and types and existing project architecture. Do not connect to a database to execute DDL."
                .to_string()
        }
    } else if task.domain == "sql" && !run.sql_schema_draft.trim().is_empty() {
        if compact {
            "Rule: Edit only .sql scripts and publish final SQL via `team_publish_contract`."
                .to_string()
        } else {
            "SQL agent boundary: the schema is approved. Only prepare/write `.sql` scripts and publish the final SQL contract with `team_publish_contract`; do not modify backend or frontend business source."
                .to_string()
        }
    } else if is_pipeline_qa_gate_task(&run.mode, task) {
        if compact {
            "QA Gate Boundary: Inspect and run build/test checks ONLY; NEVER edit business source files. Attribute every defect to the responsible upstream `task_id` with exact paths and fixes so the orchestrator routes rework back to that specialist."
                .to_string()
        } else {
            "QA Verification Gate Boundary: Your job is independent end-to-end verification, NOT fixing business code. Inspect upstream repositories and run build/test/verification commands read-only; NEVER modify business source files. When defects are found, clearly state the verdict as FAIL and group every defect under the exact responsible upstream `task_id` (with file paths, root cause, and required fixes) so the orchestrator automatically routes rework back to the owning specialist."
                .to_string()
        }
    } else {
        String::new()
    };

    let contract_publish_section = if is_revision
        && (has_dependents || matches!(task.domain.as_str(), "sql" | "backend"))
    {
        "IMPORTANT: After modifying APIs, routes, parameters, or schemas, call `team_publish_contract` with the COMPLETE updated specification (explicitly noting old -> new changes) so downstream members sync accurately.".to_string()
    } else if has_dependents || matches!(task.domain.as_str(), "sql" | "backend") {
        if compact {
            "Publish actual API/schema changes via `team_publish_contract` upon completion."
                .to_string()
        } else {
            "Contract delivery: after implementation and checks, call `team_publish_contract` to publish the actual SQL DDL or backend API paths, request parameters, and response JSON fields to the shared ledger for downstream integration."
                .to_string()
        }
    } else {
        String::new()
    };

    (
        sql_schema_section,
        interface_spec_section,
        upstream_section,
        domain_boundary_section,
        contract_publish_section,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_worker_prompt_with_source(
    run: &TeamRun,
    task: &TeamTask,
    has_dependents: bool,
    upstream_contracts: &[TeamContract],
    upstream_excerpts: &[(String, String)],
    global_templates: &BTreeMap<String, String>,
    draft_template_override: Option<&str>,
    compact: bool,
) -> (String, &'static str, String) {
    let template_id = if !task.revision_note.trim().is_empty() && run.mode != "triage" {
        TPL_WORKER_REVISION
    } else if run.mode == "triage" {
        TPL_WORKER_TRIAGE
    } else {
        TPL_WORKER_PIPELINE
    };
    render_worker_template(
        template_id,
        run,
        task,
        has_dependents,
        upstream_contracts,
        upstream_excerpts,
        global_templates,
        draft_template_override,
        compact,
    )
}

#[allow(clippy::too_many_arguments)]
fn render_worker_template(
    template_id: &str,
    run: &TeamRun,
    task: &TeamTask,
    has_dependents: bool,
    upstream_contracts: &[TeamContract],
    upstream_excerpts: &[(String, String)],
    global_templates: &BTreeMap<String, String>,
    draft_template_override: Option<&str>,
    compact: bool,
) -> (String, &'static str, String) {
    let domain_label = domain_label(&task.domain);
    let project_name = if task.project_name.trim().is_empty() {
        "-"
    } else {
        task.project_name.as_str()
    };
    let wave_str = task.wave.to_string();
    let (sql_sec, spec_sec, up_sec, dom_sec, ctr_sec) = build_worker_sections(
        run,
        task,
        has_dependents,
        upstream_contracts,
        upstream_excerpts,
        compact,
    );

    let use_compact_goal = compact
        && draft_template_override
            .filter(|s| !s.trim().is_empty())
            .is_none()
        && task.custom_prompt.trim().is_empty()
        && !task.description.trim().is_empty();
    let compact_goal_owned = if use_compact_goal {
        truncate_chars_safe(&run.user_goal, 140)
    } else {
        String::new()
    };
    let effective_goal = if use_compact_goal {
        compact_goal_owned.as_str()
    } else {
        run.user_goal.as_str()
    };
    let member_images_hint = if !task.images.is_empty() {
        format!(
            "Member-specific reference image(s) attached ({} image(s)): strictly follow the attached screenshot(s) for THIS member's UI layout/style.",
            task.images.len()
        )
    } else {
        String::new()
    };
    let touched_files_str = if task.touched_files.is_empty() {
        "(none recorded yet)".to_string()
    } else {
        task.touched_files.join(", ")
    };

    let vars = [
        ("user_goal", effective_goal),
        ("wave", wave_str.as_str()),
        ("domain", task.domain.as_str()),
        ("domain_label", domain_label),
        ("project_name", project_name),
        ("working_dir", task.working_dir.as_str()),
        ("task_title", task.title.as_str()),
        ("task_description", task.description.as_str()),
        ("revision_note", task.revision_note.as_str()),
        ("member_images_hint", member_images_hint.as_str()),
        ("touched_files", touched_files_str.as_str()),
        ("acceptance_criteria", task.acceptance_criteria.as_str()),
        ("sql_schema_section", sql_sec.as_str()),
        ("interface_spec_section", spec_sec.as_str()),
        ("upstream_section", up_sec.as_str()),
        ("domain_boundary_section", dom_sec.as_str()),
        ("contract_publish_section", ctr_sec.as_str()),
    ];

    if let Some(custom_draft) = draft_template_override.filter(|s| !s.trim().is_empty()) {
        return (
            render_template_single_pass(custom_draft, &vars),
            "custom_preview",
            custom_draft.to_string(),
        );
    }

    if template_id != TPL_WORKER_REVISION && !task.custom_prompt.trim().is_empty() {
        let raw = task.custom_prompt.as_str();
        let mut rendered_custom = render_template_single_pass(raw, &vars);
        if !member_images_hint.is_empty() && !raw.contains("{{member_images_hint}}") {
            rendered_custom = format!("{}\n\n{member_images_hint}", rendered_custom.trim_end());
        }
        return (rendered_custom, "task_custom", raw.to_string());
    }

    let (tpl, src) = resolve_effective_template(
        template_id,
        Some(&task.domain),
        Some(&run.prompt_overrides),
        global_templates,
    );
    let mut rendered = render_template_single_pass(&tpl, &vars);
    if template_id == TPL_WORKER_PIPELINE
        && !member_images_hint.is_empty()
        && !tpl.contains("{{member_images_hint}}")
    {
        rendered = format!("{}\n\n{member_images_hint}", rendered.trim_end());
    }
    let final_rendered = if src == "default" {
        collapse_blank_lines(&rendered)
    } else {
        rendered
    };
    (final_rendered, src, tpl)
}

pub(crate) fn is_pipeline_qa_gate_task(run_mode: &str, task: &TeamTask) -> bool {
    run_mode == "pipeline" && task.domain == "qa" && !task.depends_on.is_empty()
}

/// 收集 QA 验收节点上游所有可承接缺陷返工的祖先任务（按波次与排序号稳定排列；若存在非 QA 研发祖先则优先仅返回研发节点）。
pub(crate) fn collect_upstream_ancestor_tasks(
    all_tasks: &[TeamTask],
    target_task: &TeamTask,
) -> Vec<TeamTask> {
    let by_id: std::collections::HashMap<&str, &TeamTask> =
        all_tasks.iter().map(|t| (t.id.as_str(), t)).collect();
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<String> = target_task
        .depends_on
        .iter()
        .filter(|d| d.as_str() != target_task.id.as_str())
        .cloned()
        .collect();
    let mut collected: Vec<TeamTask> = Vec::new();
    while let Some(id) = queue.pop_front() {
        if !visited.insert(id.clone()) {
            continue;
        }
        if let Some(t) = by_id.get(id.as_str()) {
            collected.push((*t).clone());
            for dep in &t.depends_on {
                if dep != &target_task.id && !visited.contains(dep) {
                    queue.push_back(dep.clone());
                }
            }
        }
    }
    let has_non_qa = collected.iter().any(|t| t.domain != "qa");
    if has_non_qa {
        collected.retain(|t| t.domain != "qa");
    }
    collected.sort_by(|a, b| {
        a.wave
            .cmp(&b.wave)
            .then_with(|| a.sort_order.cmp(&b.sort_order))
            .then_with(|| a.id.cmp(&b.id))
    });
    collected
}

pub(crate) fn format_qa_upstream_roster(ancestors: &[TeamTask]) -> String {
    if ancestors.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = ancestors
        .iter()
        .map(|t| {
            let proj = if !t.project_name.trim().is_empty() {
                t.project_name.trim()
            } else {
                "-"
            };
            let files = if t.touched_files.is_empty() {
                "(none)".to_string()
            } else {
                truncate_chars_safe(&t.touched_files.join(", "), 220)
            };
            format!(
                "- task_id: \"{}\" | domain: {} | project: {} | working_dir: {} | title: {} | touched_files: {}",
                t.id,
                t.domain,
                proj,
                t.working_dir.trim(),
                truncate_chars_safe(&t.title, 80),
                files
            )
        })
        .collect();
    format!(
        "Upstream Teammates (cite exact `task_id` for any defect so rework is routed to the owning specialist):\n{}",
        lines.join("\n")
    )
}

pub(crate) fn render_qa_verify_sys_guard(working_dir: &str) -> String {
    render_template_single_pass(DEFAULT_SYS_GUARD_QA_VERIFY, &[("working_dir", working_dir)])
}

pub(crate) fn render_qa_retest_prompt(
    run: &TeamRun,
    qa_task: &TeamTask,
    qa_round: u32,
    max_repairs: u32,
    upstream_ancestors: &[TeamTask],
    upstream_contracts: &[TeamContract],
    upstream_excerpts: &[(String, String)],
    compact: bool,
) -> String {
    let (_, spec_sec, up_sec, dom_sec, _) = build_worker_sections(
        run,
        qa_task,
        false,
        upstream_contracts,
        upstream_excerpts,
        compact,
    );
    let roster = format_qa_upstream_roster(upstream_ancestors);
    let mut parts = vec![
        format!(
            "[QA Retest Round {}/{} | {}] {}",
            qa_round,
            max_repairs,
            qa_task.working_dir.trim(),
            qa_task.title
        ),
        format!(
            "Upstream specialists have completed defect fixes from the previous QA round:\n{}",
            qa_task.revision_note.trim()
        ),
        format!("Done when: {}", qa_task.acceptance_criteria.trim()),
    ];
    if !up_sec.trim().is_empty() {
        parts.push(up_sec);
    }
    if !spec_sec.trim().is_empty() {
        parts.push(spec_sec);
    }
    if !roster.trim().is_empty() {
        parts.push(roster);
    }
    if !dom_sec.trim().is_empty() {
        parts.push(dom_sec);
    }
    parts.push(
        "Re-inspect the updated repositories and re-run build/test/regression checks. Do NOT edit business source files. Clearly conclude PASS if all defects are resolved and regression checks pass, or FAIL with exact upstream `task_id` attribution for any remaining/new defects."
            .to_string(),
    );
    collapse_blank_lines(&parts.join("\n\n"))
}

pub(crate) fn render_qa_recheck_prompt(
    qa_task: &TeamTask,
    repair_attempt: u32,
    max_repairs: u32,
    verifier_summary: &str,
    repair_instructions: &str,
    upstream_ancestors: &[TeamTask],
) -> String {
    let roster = format_qa_upstream_roster(upstream_ancestors);
    let mut parts = vec![
        format!(
            "[QA Re-check {}/{} | {}]",
            repair_attempt,
            max_repairs,
            qa_task.working_dir.trim()
        ),
        format!("Incomplete QA check: {}", verifier_summary.trim()),
        format!("Verification required: {}", repair_instructions.trim()),
        format!("Done when: {}", qa_task.acceptance_criteria.trim()),
    ];
    if !roster.trim().is_empty() {
        parts.push(roster);
    }
    parts.push(
        "Rule: Do NOT modify or fix business source files yourself. Complete the read-only verification/tests, clearly state PASS or FAIL, and attribute every discovered defect to the exact responsible upstream `task_id` above."
            .to_string(),
    );
    collapse_blank_lines(&parts.join("\n\n"))
}

pub(crate) fn format_teammates_line(tasks: &[TeamTask], self_task_id: &str) -> String {
    let items: Vec<String> = tasks
        .iter()
        .filter(|t| t.id != self_task_id)
        .map(|t| {
            let name = if !t.project_name.trim().is_empty() {
                t.project_name.trim()
            } else if !t.title.trim().is_empty() {
                t.title.trim()
            } else {
                t.domain.as_str()
            };
            format!("{} | {} | {}", t.id, name, t.domain)
        })
        .collect();
    if items.is_empty() {
        String::new()
    } else {
        format!(
            "Teammates for `team_ask_peer` (pass id, project name, or domain): {}",
            items.join("; ")
        )
    }
}

pub(crate) fn render_sys_guard_with_source(
    read_only_mode: bool,
    working_dir: &str,
    session_id: &str,
    dir_profile_hint: &str,
    run_overrides: Option<&BTreeMap<String, String>>,
    global_templates: &BTreeMap<String, String>,
) -> (String, &'static str) {
    let tpl_id = if read_only_mode {
        TPL_SYS_GUARD_READONLY
    } else {
        TPL_SYS_GUARD_COLLAB
    };
    let (tpl, src) = resolve_effective_template(tpl_id, None, run_overrides, global_templates);
    let rendered = render_template_single_pass(
        &tpl,
        &[
            ("working_dir", working_dir),
            ("session_id", session_id),
            ("dir_profile_hint", dir_profile_hint),
        ],
    );
    (rendered, src)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptPreviewResponse {
    pub template_id: String,
    pub source: String,
    pub raw_template: String,
    pub rendered_prompt: String,
    pub system_guard: String,
    pub dir_profile: Option<DirProfile>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn preview_role_prompt(
    template_id: &str,
    explicit_domain: Option<&str>,
    run: Option<&TeamRun>,
    all_tasks: &[TeamTask],
    target_task: Option<&TeamTask>,
    contracts: &[TeamContract],
    custom_projects: &[TeamProjectPreset],
    global_templates: &BTreeMap<String, String>,
    draft_template: Option<&str>,
    working_dir_override: Option<&str>,
    feedback_override: Option<&str>,
    current_working_dir: &str,
    compact: bool,
) -> Result<PromptPreviewResponse, String> {
    let meta =
        get_prompt_meta(template_id).ok_or_else(|| format!("未知提示词模板 ID: {template_id}"))?;

    let run_overrides = run.map(|r| &r.prompt_overrides);
    let domain_hint = match meta.domain_scope {
        PromptDomainScope::Sql => Some(explicit_domain.unwrap_or("sql")),
        PromptDomainScope::Member => {
            explicit_domain.or_else(|| target_task.map(|t| t.domain.as_str()))
        }
        PromptDomainScope::Peer => Some(explicit_domain.unwrap_or("backend")),
        PromptDomainScope::None => None,
    };
    let (raw_template, source) = if let Some(d) = draft_template.filter(|s| !s.trim().is_empty()) {
        (d.to_string(), "custom_preview")
    } else {
        resolve_effective_template(template_id, domain_hint, run_overrides, global_templates)
    };

    let sample_goal = run
        .map(|r| r.user_goal.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("(Example: add role/permission management, schema design, and backend/frontend integration)");
    let sample_plan = run
        .map(|r| r.execution_plan.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("(Execution plan will be populated after stage 1)");
    let sample_sql = run
        .map(|r| r.sql_schema_draft.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("(Schema DDL will be populated after stage 2)");

    let explicit_wd = working_dir_override.filter(|s| !s.trim().is_empty());
    let task_wd = target_task
        .map(|t| t.working_dir.as_str())
        .filter(|s| !s.trim().is_empty());
    let run_sql_wd = run
        .map(|r| r.sql_working_dir.as_str())
        .filter(|s| !s.trim().is_empty());
    let sql_role = matches!(template_id, TPL_SQL_DESIGN_INITIAL | TPL_SQL_DESIGN_REVISE);
    let sql_wd = explicit_wd
        .or(if sql_role {
            run_sql_wd.or(task_wd)
        } else {
            task_wd.or(run_sql_wd)
        })
        .unwrap_or(if sql_role { "" } else { current_working_dir });

    match template_id {
        TPL_LEAD_PLANNER_PIPELINE | TPL_LEAD_PLANNER_TRIAGE => {
            let user_projects_desc = if custom_projects.is_empty() {
                "(none configured yet)".to_string()
            } else {
                custom_projects
                    .iter()
                    .map(|p| {
                        format!(
                            "- project_id: `{}` | name: `{}` | role_tag: `{}` | dir: `{}` | notes: {}",
                            p.id,
                            p.name,
                            p.domain,
                            p.working_dir,
                            if p.notes.trim().is_empty() { "-" } else { p.notes.as_str() }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("user_goal", sample_goal),
                    ("user_projects", &user_projects_desc),
                    (
                        "known_repos",
                        "(Local repository history omitted in preview)",
                    ),
                    ("current_working_dir", current_working_dir),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        TPL_LEAD_INTERFACE_SPEC => {
            let task_summary = if all_tasks.is_empty() {
                "- [Backend] Example API responsibilities\n- [Frontend] Example admin UI responsibilities".to_string()
            } else {
                all_tasks
                    .iter()
                    .map(|t| {
                        format!(
                            "- [{}] {} (directory: `{}`, responsibilities: {})",
                            domain_label(&t.domain),
                            if t.project_name.is_empty() {
                                &t.title
                            } else {
                                &t.project_name
                            },
                            t.working_dir,
                            t.description
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let (spec_goal, spec_plan, spec_sql) = if compact && draft_template.is_none() {
                (
                    truncate_chars_safe(sample_goal, 600),
                    truncate_chars_safe(sample_plan, 1800),
                    truncate_chars_safe(&extract_compact_sql_ddl(sample_sql), 2200),
                )
            } else {
                (
                    sample_goal.to_string(),
                    sample_plan.to_string(),
                    sample_sql.to_string(),
                )
            };
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("user_goal", &spec_goal),
                    ("execution_plan", &spec_plan),
                    ("sql_schema_draft", &spec_sql),
                    ("task_summary", &task_summary),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        TPL_SQL_DESIGN_INITIAL | TPL_SQL_DESIGN_REVISE => {
            let profile = inspect_dir_profile(Path::new(sql_wd));
            let hint = profile.to_hint();
            let fb = feedback_override
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("(Example feedback: add SORT_NO ordering and REMARK notes columns)");
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("user_goal", sample_goal),
                    ("execution_plan", sample_plan),
                    ("working_dir", sql_wd),
                    ("dir_profile_hint", &hint),
                    ("feedback", fb),
                    ("sql_schema_draft", sample_sql),
                ],
            );
            let (guard, _) = render_sys_guard_with_source(
                true,
                sql_wd,
                "preview",
                &hint,
                run_overrides,
                global_templates,
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: guard,
                dir_profile: Some(profile),
            })
        }
        TPL_SYS_GUARD_READONLY | TPL_SYS_GUARD_COLLAB => {
            let profile = inspect_dir_profile(Path::new(sql_wd));
            let hint = profile.to_hint();
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("working_dir", sql_wd),
                    ("session_id", "preview-session"),
                    ("dir_profile_hint", &hint),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: Some(profile),
            })
        }
        TPL_WORKER_PIPELINE | TPL_WORKER_TRIAGE | TPL_WORKER_REVISION => {
            let fallback_task = TeamTask {
                id: "preview_task".to_string(),
                run_id: run.map(|r| r.id.clone()).unwrap_or_default(),
                domain: domain_hint.unwrap_or("backend").to_string(),
                title: "Example implementation task".to_string(),
                description: "Implement repository APIs, validation, and contract delivery"
                    .to_string(),
                acceptance_criteria: "Build succeeds and API contracts match exactly".to_string(),
                depends_on: Vec::new(),
                wave: 1,
                sort_order: 0,
                project_slot_id: String::new(),
                project_name: "Example project".to_string(),
                avatar_id: String::new(),
                working_dir: sql_wd.to_string(),
                project_hash: String::new(),
                agent_id: "none".to_string(),
                agent_label: String::new(),
                model: String::new(),
                session_id: String::new(),
                custom_prompt: String::new(),
                images: Vec::new(),
                revision_note: if template_id == TPL_WORKER_REVISION {
                    "Rename API route to /api/v2/announcements and publish updated contract"
                        .to_string()
                } else {
                    String::new()
                },
                status: "pending".to_string(),
                attempt: 0,
                attempt_id: String::new(),
                repair_count: 0,
                contract_summary: String::new(),
                last_output_excerpt: String::new(),
                verify_verdict: String::new(),
                verify_feedback: String::new(),
                touched_files: Vec::new(),
                round_touched_files: Vec::new(),
                started_at: String::new(),
                updated_at: String::new(),
                finished_at: String::new(),
            };
            let mut preview_task = target_task.unwrap_or(&fallback_task).clone();
            if let Some(domain) = explicit_domain {
                preview_task.domain = domain.to_string();
            }
            if let Some(wd) = explicit_wd {
                preview_task.working_dir = wd.to_string();
            }
            if template_id == TPL_WORKER_REVISION && preview_task.revision_note.trim().is_empty() {
                preview_task.revision_note = feedback_override
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or("Apply incremental changes and sync updated contract")
                    .to_string();
            }
            let t = &preview_task;
            let dummy_run;
            let effective_run = match run {
                Some(r) => r,
                None => {
                    dummy_run = TeamRun {
                        id: "preview_run".to_string(),
                        title: "Example team task".to_string(),
                        user_goal: sample_goal.to_string(),
                        mode: if template_id == TPL_WORKER_TRIAGE {
                            "triage"
                        } else {
                            "pipeline"
                        }
                        .to_string(),
                        status: "draft".to_string(),
                        stage: "team_binding".to_string(),
                        requires_db: true,
                        execution_plan: sample_plan.to_string(),
                        sql_schema_draft: sample_sql.to_string(),
                        interface_spec: String::new(),
                        delivery_sql: String::new(),
                        sql_working_dir: String::new(),
                        sql_agent_id: "none".to_string(),
                        sql_model: String::new(),
                        sql_session_id: String::new(),
                        prompt_overrides: BTreeMap::new(),
                        goal_images: Vec::new(),
                        revision: 1,
                        auto_verify: true,
                        max_repairs: 2,
                        lead_model: String::new(),
                        lead_agent_id: "none".to_string(),
                        current_round_no: 1,
                        source_triage_id: String::new(),
                        synthesis_report: String::new(),
                        created_at: String::new(),
                        updated_at: String::new(),
                        finished_at: String::new(),
                    };
                    &dummy_run
                }
            };
            let dep_set: HashSet<String> = t.depends_on.iter().cloned().collect();
            let has_dependents = all_tasks
                .iter()
                .any(|other| other.depends_on.iter().any(|d| d == &t.id));
            let up_ctrs: Vec<TeamContract> = contracts
                .iter()
                .filter(|c| dep_set.contains(&c.from_task_id))
                .cloned()
                .collect();
            let up_excerpts: Vec<(String, String)> = all_tasks
                .iter()
                .filter(|x| dep_set.contains(&x.id))
                .map(|x| {
                    let s = if !x.contract_summary.trim().is_empty() {
                        x.contract_summary.clone()
                    } else {
                        x.last_output_excerpt.clone()
                    };
                    (format!("{} - {}", domain_label(&x.domain), x.title), s)
                })
                .collect();
            let (rendered, final_src, final_raw) = render_worker_template(
                template_id,
                effective_run,
                t,
                has_dependents,
                &up_ctrs,
                &up_excerpts,
                global_templates,
                draft_template,
                compact,
            );
            let read_only = template_id == TPL_WORKER_TRIAGE;
            let (mut guard, _) = render_sys_guard_with_source(
                read_only,
                &t.working_dir,
                "preview",
                "",
                run_overrides,
                global_templates,
            );
            let teammates_line = format_teammates_line(all_tasks, &t.id);
            if !teammates_line.is_empty() {
                guard.push('\n');
                guard.push_str(&teammates_line);
            }
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: final_src.to_string(),
                raw_template: final_raw,
                rendered_prompt: rendered,
                system_guard: guard,
                dir_profile: None,
            })
        }
        TPL_MEMBER_NOTE => {
            let t_title = target_task
                .map(|t| t.title.as_str())
                .unwrap_or("Example member task");
            let d_label = domain_label(domain_hint.unwrap_or("backend"));
            let p_name = target_task
                .map(|t| t.project_name.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("Example project");
            let wd = sql_wd;
            let note = feedback_override
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("Add null parameter validation and return standard error codes");
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("note", note),
                    ("task_title", t_title),
                    ("domain_label", d_label),
                    ("project_name", p_name),
                    ("working_dir", wd),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        TPL_VERIFIER_GATE => {
            let t_title = target_task
                .map(|t| t.title.as_str())
                .unwrap_or("Backend API implementation");
            let t_dom = domain_hint.unwrap_or("backend");
            let t_desc = target_task
                .map(|t| t.description.as_str())
                .unwrap_or("Implement business APIs");
            let t_acc = target_task
                .map(|t| t.acceptance_criteria.as_str())
                .unwrap_or("Build succeeds");
            let v_sql = if compact && draft_template.is_none() {
                truncate_chars_safe(&extract_compact_sql_ddl(sample_sql), 900)
            } else {
                sample_sql.to_string()
            };
            let raw_spec = run
                .map(|r| r.interface_spec.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("(none)");
            let v_spec = if compact && draft_template.is_none() && raw_spec != "(none)" {
                truncate_chars_safe(raw_spec, 1000)
            } else {
                raw_spec.to_string()
            };
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("task_title", t_title),
                    ("domain", t_dom),
                    ("task_description", t_desc),
                    ("acceptance_criteria", t_acc),
                    ("approved_sql", &v_sql),
                    ("locked_interface_spec", &v_spec),
                    (
                        "upstream_contracts",
                        "(Preview: upstream contracts are injected at runtime)",
                    ),
                    (
                        "published_contract",
                        "(Preview: member published contract is injected at runtime)",
                    ),
                    (
                        "touched_files_snippets",
                        "(Preview: modified file snippets are read at runtime)",
                    ),
                    (
                        "worker_output",
                        "(Preview: member final output is injected at runtime)",
                    ),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        TPL_WORKER_REPAIR => {
            let t_acc = target_task
                .map(|t| t.acceptance_criteria.as_str())
                .unwrap_or("Build succeeds and contracts match");
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("repair_attempt", "1"),
                    ("max_repairs", "2"),
                    (
                        "verifier_summary",
                        "(Example verdict: response field names do not match the contract)",
                    ),
                    (
                        "repair_instructions",
                        "(Example repair: rename userName to nickName)",
                    ),
                    ("acceptance_criteria", t_acc),
                    ("working_dir", sql_wd),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        TPL_TRIAGE_SYNTHESIS => {
            let findings = if all_tasks.is_empty() {
                "## [Backend] Example investigation evidence\n(Findings are populated at runtime)"
                    .to_string()
            } else {
                all_tasks
                    .iter()
                    .map(|t| {
                        format!(
                            "## [{}] {} (`{}`)\n{}",
                            domain_label(&t.domain),
                            t.title,
                            t.working_dir,
                            t.last_output_excerpt
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n")
            };
            let rendered = render_template_single_pass(
                &raw_template,
                &[("user_goal", sample_goal), ("findings", &findings)],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        TPL_PEER_CONSULTATION => {
            let rendered = render_template_single_pass(
                &raw_template,
                &[
                    ("target_badge", domain_label(domain_hint.unwrap_or("backend"))),
                    ("question", "(Example question: what is the status field type in the paginated users API?)"),
                    ("contracts_text", "(Preview: published contracts are injected at runtime)"),
                    ("peer_excerpt", "(Preview: latest peer output is injected at runtime)"),
                    ("code_evidence", "(Preview: peer source evidence is retrieved at runtime)"),
                ],
            );
            Ok(PromptPreviewResponse {
                template_id: meta.id.to_string(),
                source: source.to_string(),
                raw_template,
                rendered_prompt: rendered,
                system_guard: String::new(),
                dir_profile: None,
            })
        }
        _ => Err(format!("未实现的提示词模板: {template_id}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run() -> TeamRun {
        serde_json::from_value(serde_json::json!({
            "id": "r", "title": "Run", "userGoal": "用户数据 {{wave}}\n\n\n"
        }))
        .unwrap()
    }

    fn task() -> TeamTask {
        serde_json::from_value(serde_json::json!({
            "id": "t", "runId": "r", "title": "Task", "domain": "backend",
            "model": "",
            "sessionId": "",
            "customPrompt": "",
            "images": [],
            "revisionNote": "",
            "workingDir": "task-dir"
        }))
        .unwrap()
    }

    #[test]
    fn registry_unique_and_all_placeholders_render_in_preview() {
        let mut ids = HashSet::new();
        for meta in PROMPT_REGISTRY {
            assert!(ids.insert(meta.id), "duplicate {}", meta.id);
            assert!(
                !meta
                    .default_template
                    .chars()
                    .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
                "core template {}",
                meta.id
            );
            let vars: Vec<_> = meta
                .placeholders
                .iter()
                .map(|key| (*key, "VALUE"))
                .collect();
            let rendered = render_template_single_pass(meta.default_template, &vars);
            assert!(
                !rendered.contains("{{"),
                "undeclared placeholder in {}",
                meta.id
            );
            // Exercise even optional placeholders not used in the built-in text.
            let draft = meta
                .placeholders
                .iter()
                .map(|k| format!("{{{{{k}}}}}"))
                .collect::<Vec<_>>()
                .join("|");
            let preview = preview_role_prompt(
                meta.id,
                None,
                None,
                &[],
                None,
                &[],
                &[],
                &BTreeMap::new(),
                Some(&draft),
                None,
                None,
                "",
                true,
            )
            .unwrap();
            assert!(
                !preview.rendered_prompt.contains("{{"),
                "unbound preview placeholder in {}",
                meta.id
            );
        }
    }

    #[test]
    fn single_pass_preserves_values_whitespace_and_unknown_tokens() {
        let value = " \n```vue\n{{wave}}\n\n\n```\r\n ";
        assert_eq!(
            render_template_single_pass("{{user_goal}}", &[("user_goal", value), ("wave", "9")]),
            value
        );
        assert_eq!(
            render_template_single_pass(" {{unknown}} {{ not-a-key }} {{unfinished", &[]),
            " {{unknown}} {{ not-a-key }} {{unfinished"
        );
    }

    #[test]
    fn override_hierarchy_and_explicit_worker_preview() {
        let mut r = run();
        r.mode = "triage".into();
        let mut t = task();
        let mut globals = BTreeMap::new();
        globals.insert(TPL_WORKER_PIPELINE.into(), "global".into());
        globals.insert(
            format!("{TPL_WORKER_PIPELINE}:backend"),
            "global-domain".into(),
        );
        assert_eq!(
            resolve_effective_template(
                TPL_WORKER_PIPELINE,
                Some("backend"),
                Some(&r.prompt_overrides),
                &globals
            )
            .0,
            "global-domain"
        );
        r.prompt_overrides
            .insert(TPL_WORKER_PIPELINE.into(), "run".into());
        assert_eq!(
            resolve_effective_template(
                TPL_WORKER_PIPELINE,
                Some("backend"),
                Some(&r.prompt_overrides),
                &globals
            )
            .0,
            "run"
        );
        r.prompt_overrides.insert(
            format!("{TPL_WORKER_PIPELINE}:backend"),
            "run-domain {{working_dir}}".into(),
        );
        let preview = preview_role_prompt(
            TPL_WORKER_PIPELINE,
            None,
            Some(&r),
            &[],
            Some(&t),
            &[],
            &[],
            &globals,
            None,
            Some("override-dir"),
            None,
            "cwd",
            true,
        )
        .unwrap();
        assert_eq!(preview.rendered_prompt, "run-domain override-dir");
        assert_eq!(preview.source, "run_override");
        assert!(preview.system_guard.starts_with("TEAM COLLABORATION"));
        t.custom_prompt = " \ncustom {{user_goal}}\n\n\n ".into();
        let actual =
            render_worker_prompt_with_source(&r, &t, false, &[], &[], &globals, None, true);
        assert_eq!(actual.1, "task_custom");
        assert_eq!(actual.0, format!(" \ncustom {}\n\n\n ", r.user_goal));
        let preview = preview_role_prompt(
            TPL_WORKER_PIPELINE,
            None,
            Some(&r),
            &[],
            Some(&t),
            &[],
            &[],
            &globals,
            Some(" draft {{working_dir}} "),
            Some("override-dir"),
            None,
            "cwd",
            true,
        )
        .unwrap();
        assert_eq!(preview.source, "custom_preview");
        assert_eq!(preview.rendered_prompt, " draft override-dir ");
        assert_eq!(
            resolve_effective_template(TPL_WORKER_TRIAGE, None, None, &BTreeMap::new()).1,
            "default"
        );
    }

    #[test]
    fn preview_domain_resolution_preserves_precedence_and_source() {
        let mut r = run();
        let t = task();
        let mut globals = BTreeMap::new();
        globals.insert(
            format!("{TPL_WORKER_PIPELINE}:frontend"),
            "global frontend {{domain}}".into(),
        );
        r.prompt_overrides
            .insert(TPL_WORKER_PIPELINE.into(), "run base {{domain}}".into());
        let preview = preview_role_prompt(
            TPL_WORKER_PIPELINE,
            Some("frontend"),
            Some(&r),
            &[],
            Some(&t),
            &[],
            &[],
            &globals,
            None,
            None,
            None,
            "",
            true,
        )
        .unwrap();
        assert_eq!(preview.rendered_prompt, "run base frontend");
        assert_eq!(preview.raw_template, "run base {{domain}}");
        assert_eq!(preview.source, "run_override");
        r.prompt_overrides.clear();
        let preview = preview_role_prompt(
            TPL_WORKER_PIPELINE,
            Some("frontend"),
            Some(&r),
            &[],
            Some(&t),
            &[],
            &[],
            &globals,
            None,
            None,
            None,
            "",
            true,
        )
        .unwrap();
        assert_eq!(preview.rendered_prompt, "global frontend frontend");
        assert_eq!(preview.source, "global_template");
        // With no task, the fallback worker must use the explicit domain too.
        let preview = preview_role_prompt(
            TPL_WORKER_PIPELINE,
            Some("frontend"),
            None,
            &[],
            None,
            &[],
            &[],
            &globals,
            None,
            None,
            None,
            "",
            true,
        )
        .unwrap();
        assert_eq!(preview.rendered_prompt, "global frontend frontend");
        assert_eq!(t.domain, "backend");
    }

    #[test]
    fn preview_sql_defaults_to_sql_and_non_domain_roles_ignore_task_domain() {
        let t = task();
        for id in [TPL_SQL_DESIGN_INITIAL, TPL_SQL_DESIGN_REVISE] {
            let globals = BTreeMap::from([
                (id.to_string(), "base".to_string()),
                (format!("{id}:sql"), "sql domain".to_string()),
                (format!("{id}:backend"), "wrong task domain".to_string()),
            ]);
            for target in [None, Some(&t)] {
                let preview = preview_role_prompt(
                    id,
                    None,
                    None,
                    &[],
                    target,
                    &[],
                    &[],
                    &globals,
                    None,
                    Some(""),
                    None,
                    "",
                    true,
                )
                .unwrap();
                assert_eq!(preview.raw_template, "sql domain");
                assert_eq!(preview.source, "global_template");
            }
        }
        for id in [
            TPL_SYS_GUARD_READONLY,
            TPL_SYS_GUARD_COLLAB,
            TPL_TRIAGE_SYNTHESIS,
        ] {
            let globals = BTreeMap::from([
                (id.to_string(), "base".to_string()),
                (format!("{id}:backend"), "wrong domain".to_string()),
            ]);
            let preview = preview_role_prompt(
                id,
                None,
                None,
                &[],
                Some(&t),
                &[],
                &[],
                &globals,
                None,
                None,
                None,
                "",
                true,
            )
            .unwrap();
            assert_eq!(preview.raw_template, "base");
            assert_eq!(preview.source, "global_template");
        }
    }

    #[test]
    fn preview_member_verifier_and_repair_use_explicit_or_task_domain() {
        let t = task();
        for id in [TPL_MEMBER_NOTE, TPL_VERIFIER_GATE, TPL_WORKER_REPAIR] {
            let globals = BTreeMap::from([
                (format!("{id}:backend"), "task domain".to_string()),
                (format!("{id}:frontend"), "explicit domain".to_string()),
            ]);
            for (domain, expected) in [(None, "task domain"), (Some("frontend"), "explicit domain")]
            {
                let preview = preview_role_prompt(
                    id,
                    domain,
                    None,
                    &[],
                    Some(&t),
                    &[],
                    &[],
                    &globals,
                    None,
                    None,
                    None,
                    "",
                    true,
                )
                .unwrap();
                assert_eq!(preview.raw_template, expected);
                assert_eq!(preview.source, "global_template");
            }
        }
    }

    #[test]
    fn compact_mode_extracts_ddl_and_skips_sql_for_client_when_spec_present() {
        let mut r = run();
        r.user_goal = "a".repeat(500);
        r.sql_schema_draft = "### 1. 说明\n长篇表格\n```sql\nCREATE TABLE sys_demo (id INT);\n```\n### 2. 字段解释\n| a | b |\n| - | - |".into();
        r.interface_spec = "GET /api/demo -> {id, name}".into();

        let mut be = task();
        be.domain = "backend".into();
        be.description = "Implement demo API".into();
        let (be_prompt, _, _) =
            render_worker_prompt_with_source(&r, &be, true, &[], &[], &BTreeMap::new(), None, true);
        assert!(be_prompt.contains("CREATE TABLE sys_demo"));
        assert!(!be_prompt.contains("长篇表格"));
        assert!(be_prompt.contains("...[truncated]..."));
        assert!(!be_prompt.contains("\n\n\n"));

        let mut fe = task();
        fe.domain = "frontend".into();
        fe.description = "Implement demo page".into();
        let (fe_prompt, _, _) = render_worker_prompt_with_source(
            &r,
            &fe,
            false,
            &[],
            &[],
            &BTreeMap::new(),
            None,
            true,
        );
        assert!(!fe_prompt.contains("CREATE TABLE sys_demo"));
        assert!(fe_prompt.contains("GET /api/demo"));

        // Non-compact retains full SQL even for frontend
        let (fe_full, _, _) = render_worker_prompt_with_source(
            &r,
            &fe,
            false,
            &[],
            &[],
            &BTreeMap::new(),
            None,
            false,
        );
        assert!(fe_full.contains("CREATE TABLE sys_demo"));

        // Compact strips redundant legacy fallback sections from interface_spec
        let mut r_legacy = run();
        r_legacy.interface_spec = "### 约定\n\n#### 1. 参考执行计划\n长篇计划\n\n#### 2. 已确认数据库表结构\n9000字废话\n\n#### 3. 协作分工\n后端做接口\n\n#### 4. 统一接口契约要点\nGET /api/ok".into();
        let (fe_legacy, _, _) = render_worker_prompt_with_source(
            &r_legacy,
            &fe,
            false,
            &[],
            &[],
            &BTreeMap::new(),
            None,
            true,
        );
        assert!(!fe_legacy.contains("9000字废话"));
        assert!(fe_legacy.contains("GET /api/ok"));
    }

    #[test]
    fn revision_prompt_and_member_images_override_baseline_spec() {
        let mut r = run();
        r.interface_spec = "OLD GET /api/old-notice".into();
        let mut t = task();
        t.custom_prompt = "Old v1 custom prompt".into();
        t.revision_note = "Rename route to /api/system/announcements".into();
        t.images = vec![super::super::types::TeamImage {
            media_type: "image/png".into(),
            data: "IMG".into(),
        }];
        let (rendered, src, _) =
            render_worker_prompt_with_source(&r, &t, true, &[], &[], &BTreeMap::new(), None, true);
        assert_eq!(src, "default");
        assert!(rendered.contains("Incremental Revision"));
        assert!(rendered.contains("Rename route to /api/system/announcements"));
        assert!(rendered.contains("Member-specific reference image(s) attached (1 image(s))"));
        assert!(rendered.contains("Historical Baseline Spec"));
        assert!(!rendered.contains("Old v1 custom prompt"));
    }

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "zap-prompts-{}-{stamp}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn empty_readme_sql_and_bounded_profiles() {
        assert_eq!(
            inspect_dir_profile(Path::new("")).kind,
            DirProfileKind::Missing
        );
        let tmp = TempDir::new();
        let empty = inspect_dir_profile(&tmp.0);
        assert_eq!(empty.kind, DirProfileKind::Empty);
        assert_eq!(empty.scanned_files, 0);
        assert!(!empty.truncated);
        std::fs::write(tmp.0.join("README.md"), "reference only").unwrap();
        let readme = inspect_dir_profile(&tmp.0);
        assert_eq!(readme.scanned_files, 1);
        assert!(readme.to_hint().contains("does not establish"));
        for name in ["z.sql", "a.sql"] {
            std::fs::write(tmp.0.join(name), "SELECT 1;").unwrap();
        }
        let sql = inspect_dir_profile(&tmp.0);
        assert_eq!(sql.kind, DirProfileKind::SqlOnly);
        assert_eq!(sql.sample_sql_files, vec!["a.sql", "z.sql"]);
        assert!(sql.to_hint().contains("observed SQL: 2"));
        assert!(!sql.to_hint().contains("Inspect relevant existing source"));
        for i in 0..110 {
            std::fs::create_dir(tmp.0.join(format!("dir{i:03}"))).unwrap();
        }
        let bounded = inspect_dir_profile(&tmp.0);
        assert!(bounded.truncated);
        assert!(bounded.scanned_dirs <= 100);
        assert_eq!(
            bounded.sample_sql_files,
            inspect_dir_profile(&tmp.0).sample_sql_files
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_do_not_escape_scan_root() {
        let root = TempDir::new();
        let outside = TempDir::new();
        std::fs::write(outside.0.join("secret.sql"), "SELECT 1;").unwrap();
        std::os::unix::fs::symlink(&outside.0, root.0.join("linked")).unwrap();
        let profile = inspect_dir_profile(&root.0);
        assert_eq!(profile.sql_count, 0);
        assert!(profile.truncated);
        assert_eq!(
            inspect_dir_profile(&root.0.join("linked")).kind,
            DirProfileKind::Missing
        );
    }
}
