//! `memory_recall` 只读工具：允许模型按关键词、项目与日期范围主动跨项目检索记忆。

use std::collections::HashSet;
use std::path::Path;
use std::sync::RwLock;

use async_trait::async_trait;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};
use zap_kernel::tool::{Tool, ToolContext, ToolResult};

use super::redact::redact_secrets;
use super::search::{
    build_fts_queries, parse_temporal_range, query_nodes_fts_conn, query_raw_turns_fts_conn,
    query_timeline_fts_conn, ProjectScope,
};
use super::store::{
    list_known_projects_conn, load_open_tasks_scoped_conn, load_profiles_scoped_conn,
    load_raw_turn_conn, resolve_agent_for_session_conn, with_store,
};
use super::types::{
    normalize_project_key, normalize_project_key_str, normalize_project_name,
    MAX_RECALL_RESULT_CHARS,
};

#[derive(Debug, Deserialize)]
pub(crate) struct MemoryRecallArgs {
    #[serde(default)]
    pub(crate) query: String,
    #[serde(default)]
    pub(crate) project: String,
    #[serde(default)]
    pub(crate) date_range: String,
    #[serde(default)]
    pub(crate) include_raw: Option<bool>,
    #[serde(default)]
    pub(crate) limit: Option<usize>,
}

pub struct MemoryRecallTool {
    session_id: RwLock<String>,
}

impl MemoryRecallTool {
    pub fn new(session_id: Option<String>) -> Self {
        Self {
            session_id: RwLock::new(session_id.map(|s| s.trim().to_string()).unwrap_or_default()),
        }
    }

    pub fn bind_session_id(&self, session_id: &str) {
        let trimmed = session_id.trim();
        if trimmed.is_empty() {
            return;
        }
        if let Ok(mut guard) = self.session_id.write() {
            *guard = trimmed.to_string();
        }
    }
}

#[async_trait]
impl Tool for MemoryRecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }

    fn description(&self) -> &str {
        "Search the active neural memory agent's long-term profiles, tasks, knowledge nodes, timeline summaries, and verbatim turn transcripts across projects and date ranges. Use project=\"*\" for cross-project search or project=\"<name>\" for a specific project."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, identifiers, error messages, or natural-language topic"
                },
                "project": {
                    "type": "string",
                    "description": "Project filter: omit or empty for current working directory; \"*\" for all projects; or a specific project name/path (e.g. \"tsm-server\")"
                },
                "date_range": {
                    "type": "string",
                    "description": "Optional time filter such as \"today\", \"yesterday\", \"this week\", \"last 3 days\", \"2026-11-15\", etc."
                },
                "include_raw": {
                    "type": "boolean",
                    "description": "Whether to also return verbatim L3 User/Assistant transcripts for matched turns (default false)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum items per section (default 6, max 15)"
                }
            }
        })
    }

    fn read_only_hint(&self) -> bool {
        true
    }

    async fn execute(&self, args: &str, ctx: &ToolContext) -> ToolResult {
        let raw_args = if args.trim().is_empty() { "{}" } else { args };
        let parsed: MemoryRecallArgs = match serde_json::from_str(raw_args) {
            Ok(a) => a,
            Err(e) => {
                return ToolResult {
                    call_id: String::new(),
                    content: format!("Invalid memory_recall arguments: {e}"),
                    is_error: true,
                    images: Vec::new(),
                };
            }
        };
        let sid = self
            .session_id
            .read()
            .map(|g| g.clone())
            .unwrap_or_default();
        let working_dir = ctx.working_dir.clone();

        let res = tokio::task::spawn_blocking(move || {
            with_store(|store| execute_memory_recall_conn(&store.conn, &sid, &working_dir, parsed))
        })
        .await;

        match res {
            Ok(Ok(text)) => ToolResult {
                call_id: String::new(),
                content: text,
                is_error: false,
                images: Vec::new(),
            },
            Ok(Err(e)) => ToolResult {
                call_id: String::new(),
                content: format!("memory_recall failed: {e}"),
                is_error: true,
                images: Vec::new(),
            },
            Err(e) => ToolResult {
                call_id: String::new(),
                content: format!("memory_recall task join error: {e}"),
                is_error: true,
                images: Vec::new(),
            },
        }
    }
}

pub(crate) fn execute_memory_recall_conn(
    conn: &Connection,
    session_id: &str,
    working_dir: &Path,
    args: MemoryRecallArgs,
) -> Result<String, String> {
    let Some(agent) = resolve_agent_for_session_conn(conn, session_id) else {
        return Ok(
            "No neural memory agent is currently active for this session (activeAgent = none)."
                .to_string(),
        );
    };

    let limit = args.limit.unwrap_or(6).clamp(1, 15);
    let include_raw = args.include_raw.unwrap_or(false);
    let current_proj_name = normalize_project_name(working_dir);
    let current_proj_key = normalize_project_key(working_dir);
    let raw_proj = args.project.trim();

    // 区分："" => 当前项目；"*" / "all" => 全项目；其他 => 按指定项目名或路径过滤
    let owned_key: String;
    let owned_name: String;
    let scope: Option<ProjectScope<'_>> = if raw_proj == "*" || raw_proj.eq_ignore_ascii_case("all")
    {
        None
    } else if raw_proj.is_empty() {
        if current_proj_key.is_empty() && current_proj_name.is_empty() {
            None
        } else {
            Some(ProjectScope {
                key: &current_proj_key,
                name: &current_proj_name,
            })
        }
    } else if raw_proj.contains('/') || raw_proj.contains('\\') {
        owned_key = normalize_project_key_str(raw_proj);
        owned_name = normalize_project_name(Path::new(raw_proj));
        Some(ProjectScope {
            key: &owned_key,
            name: &owned_name,
        })
    } else {
        Some(ProjectScope {
            key: "",
            name: raw_proj,
        })
    };

    let known_projects = list_known_projects_conn(conn, &agent.id);
    let date_range = if !args.date_range.trim().is_empty() {
        parse_temporal_range(args.date_range.trim())
    } else {
        parse_temporal_range(args.query.trim())
    };
    let (fts_strict, fts_broad) = build_fts_queries(args.query.trim());

    let mut out = vec![format!(
        "Neural Memory Recall | agent={} ({}) | scope={}",
        agent.label,
        agent.id,
        match scope {
            Some(sc) if !sc.name.is_empty() => sc.name,
            Some(sc) => sc.key,
            None => "*",
        }
    )];

    // 1. 相关 L1 长期档案与未完成任务（SQL 作用域下推：先取足量候选再按相关性排序，避免 >10 条后漏召回）
    let (s_key, s_name) = scope.map(|s| (s.key, s.name)).unwrap_or(("", ""));
    let profiles = load_profiles_scoped_conn(conn, &agent.id, s_key, s_name, 64)?;
    let q_low = args.query.trim().to_lowercase();
    let broad_terms: Vec<String> = fts_broad
        .as_deref()
        .map(|b| {
            b.split(" OR ")
                .map(|t| t.trim().trim_matches('"').to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let mut ranked_profiles: Vec<(bool, bool, super::types::ProfileFact)> = profiles
        .into_iter()
        .filter_map(|p| {
            let p_low = p.text.to_lowercase();
            let is_match = !q_low.is_empty()
                && (p_low.contains(&q_low) || broad_terms.iter().any(|t| p_low.contains(t)));
            if q_low.is_empty() || is_match || p.pinned {
                Some((is_match, p.pinned, p))
            } else {
                None
            }
        })
        .collect();
    if !q_low.is_empty() {
        ranked_profiles.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    }
    let matched_profiles: Vec<_> = ranked_profiles
        .into_iter()
        .map(|(_, _, p)| p)
        .take(8)
        .collect();
    if !matched_profiles.is_empty() {
        out.push(format!(
            "## L1 Profile Facts\n{}",
            matched_profiles
                .iter()
                .map(|p| {
                    let tag = if p.project.is_empty() {
                        String::new()
                    } else {
                        format!("[{}] ", p.project)
                    };
                    format!("- {tag}{}", p.text)
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    let tasks = load_open_tasks_scoped_conn(conn, &agent.id, s_key, s_name, limit)?;
    if !tasks.is_empty() {
        out.push(format!(
            "## L1 Open Tasks\n{}",
            tasks
                .iter()
                .map(|t| {
                    let p = if t.project.is_empty() {
                        String::new()
                    } else {
                        format!("[{}] ", t.project)
                    };
                    let rem = if t.remaining_note.is_empty() {
                        String::new()
                    } else {
                        format!(" | remaining: {}", t.remaining_note)
                    };
                    format!("- ({}, id={}) {p}{}{rem}", t.status, t.id, t.title)
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // 2. L2 知识节点
    let mut nodes = Vec::new();
    let mut seen_nodes = HashSet::new();
    for fq in [fts_strict.as_deref(), fts_broad.as_deref()]
        .into_iter()
        .flatten()
    {
        if nodes.len() >= limit {
            break;
        }
        for n in query_nodes_fts_conn(conn, &agent.id, fq, scope, limit) {
            if seen_nodes.insert(n.id.clone()) {
                nodes.push(n);
            }
            if nodes.len() >= limit {
                break;
            }
        }
    }
    if !nodes.is_empty() {
        out.push(format!(
            "## L2 Knowledge Nodes\n{}",
            nodes
                .iter()
                .map(|n| {
                    let p = if n.project.is_empty() {
                        String::new()
                    } else {
                        format!("[{}] ", n.project)
                    };
                    format!("- {p}{} ({}%): {}", n.name, n.mastery, n.summary)
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // 3. L2 时间线摘要
    let mut timelines = Vec::new();
    let mut seen_tl = HashSet::new();
    for fq in [fts_strict.as_deref(), fts_broad.as_deref()]
        .into_iter()
        .flatten()
    {
        if timelines.len() >= limit {
            break;
        }
        for ev in query_timeline_fts_conn(conn, &agent.id, fq, scope, date_range.as_ref(), limit) {
            if seen_tl.insert(ev.id.clone()) {
                timelines.push(ev);
            }
            if timelines.len() >= limit {
                break;
            }
        }
    }
    if timelines.is_empty() && fts_broad.is_none() {
        let mut conds = vec!["agent_id = ?1".to_string()];
        let mut pvals: Vec<rusqlite::types::Value> =
            vec![rusqlite::types::Value::from(agent.id.clone())];
        let mut pidx = 2usize;
        if let Some(sc) = scope {
            if !sc.key.is_empty() && !sc.name.is_empty() {
                let pk = pidx;
                let pn = pidx + 1;
                pidx += 2;
                conds.push(format!(
                    "((project_key != '' AND project_key = ?{pk}) OR (project_key = '' AND LOWER(project) = LOWER(?{pn})))"
                ));
                pvals.push(rusqlite::types::Value::from(sc.key.to_string()));
                pvals.push(rusqlite::types::Value::from(sc.name.to_string()));
            } else if !sc.key.is_empty() {
                conds.push(format!("project_key = ?{pidx}"));
                pvals.push(rusqlite::types::Value::from(sc.key.to_string()));
                pidx += 1;
            } else if !sc.name.is_empty() {
                conds.push(format!("LOWER(project) = LOWER(?{pidx})"));
                pvals.push(rusqlite::types::Value::from(sc.name.to_string()));
                pidx += 1;
            }
        }
        if let Some(dr) = &date_range {
            let ps = pidx;
            let pe = pidx + 1;
            pidx += 2;
            conds.push(format!("ts >= ?{ps} AND ts <= ?{pe}"));
            pvals.push(rusqlite::types::Value::from(dr.start.clone()));
            pvals.push(rusqlite::types::Value::from(dr.end.clone()));
        }
        pvals.push(rusqlite::types::Value::from(limit as i64));
        let sql = format!(
            "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                    user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
             FROM timeline WHERE {} ORDER BY ts DESC LIMIT ?{pidx}",
            conds.join(" AND ")
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            if let Ok(rows) = stmt.query_map(rusqlite::params_from_iter(pvals), |row| {
                let kw_str: String = row.get(12)?;
                let tf_str: String = row.get(13)?;
                Ok(super::types::TimelineEvent {
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
                })
            }) {
                for ev in rows.flatten() {
                    if seen_tl.insert(ev.id.clone()) {
                        timelines.push(ev);
                    }
                }
            }
        }
    }

    if !timelines.is_empty() {
        out.push(format!(
            "## L2 Timeline Events\n{}",
            timelines
                .iter()
                .map(|ev| {
                    let p = if ev.project.is_empty() {
                        String::new()
                    } else {
                        format!("[{}] ", ev.project)
                    };
                    let files = if ev.touched_files.is_empty() {
                        String::new()
                    } else {
                        format!(" | files: {}", ev.touched_files.join(", "))
                    };
                    let out_s = if ev.ai_outcome.is_empty() {
                        String::new()
                    } else {
                        format!(" -> {}", ev.ai_outcome)
                    };
                    format!("- [{}] {p}{}{out_s}{files}", ev.ts, ev.summary)
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    // 4. L3 原始对话摘录（支持直接 FTS 命中 + 由 timeline.turn_id 反查）
    if include_raw {
        let mut raws = Vec::new();
        let mut seen_r = HashSet::new();
        for fq in [fts_strict.as_deref(), fts_broad.as_deref()]
            .into_iter()
            .flatten()
        {
            if raws.len() >= 4 {
                break;
            }
            for r in query_raw_turns_fts_conn(conn, &agent.id, fq, scope, date_range.as_ref(), 4) {
                if seen_r.insert(r.turn_id.clone()) {
                    raws.push(r);
                }
                if raws.len() >= 4 {
                    break;
                }
            }
        }
        for ev in &timelines {
            if raws.len() >= 4 {
                break;
            }
            if !ev.turn_id.is_empty() && seen_r.insert(ev.turn_id.clone()) {
                if let Ok(Some(r)) = load_raw_turn_conn(conn, &ev.turn_id) {
                    raws.push(r);
                }
            }
        }
        if !raws.is_empty() {
            let blocks: Vec<String> = raws
                .into_iter()
                .map(|r| {
                    let u: String = redact_secrets(&r.user_text).chars().take(700).collect();
                    let a: String = redact_secrets(&r.ai_text).chars().take(1200).collect();
                    format!(
                        "--- [{}] [{}] ---\nUser: {u}\nAssistant: {a}",
                        r.ts, r.project
                    )
                })
                .collect();
            out.push(format!(
                "## L3 Verbatim Transcripts\n{}",
                blocks.join("\n\n")
            ));
        }
    }

    if out.len() == 1 {
        let mut proj_names: Vec<String> = known_projects
            .into_iter()
            .filter(|kp| !kp.name.is_empty())
            .map(|kp| kp.name)
            .collect();
        proj_names.sort();
        proj_names.dedup();
        out.push(format!(
            "No matching memory records found in this scope. Known projects for this agent: [{}]. Try project=\"*\" or broader keywords.",
            proj_names.join(", ")
        ));
    }

    let joined = out.join("\n\n");
    if joined.chars().count() > MAX_RECALL_RESULT_CHARS {
        Ok(format!(
            "{}\n...(truncated)",
            joined
                .chars()
                .take(MAX_RECALL_RESULT_CHARS)
                .collect::<String>()
        ))
    } else {
        Ok(joined)
    }
}
