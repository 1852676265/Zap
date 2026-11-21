//! `memory_recall` 只读工具：走独立 WAL 只读连接，支持 5 抽屉按需筛选、路牌精确直达、
//! 分列加权 BM25 + 词林重排、作废历史追溯、1-hop 图谱邻居与 Team 跨端进度透视。

use std::collections::HashSet;
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;

use async_trait::async_trait;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};
use zap_kernel::tool::{Tool, ToolContext, ToolResult};

use super::redact::redact_secrets;
use super::retrieval::{
    expand_top_node_neighbors_conn, query_nodes_ranked_conn, query_profiles_ranked_conn,
    query_raw_turns_cold_conn, query_tasks_ranked_conn, query_timeline_ranked_conn,
};
use super::search::{analyze_query, parse_temporal_range, ProjectScope};
use super::store::{
    list_known_projects_conn, load_open_tasks_scoped_conn, load_profiles_scoped_conn,
    load_raw_turn_conn, map_timeline_row, record_memory_hits_conn, resolve_agent_for_session_conn,
    with_read, with_write,
};
use super::team_bridge::{
    format_team_overview_line, list_recent_team_runs_for_project, lookup_team_run_overview,
};
use super::types::{
    normalize_project_key, normalize_project_key_str, normalize_project_name,
    MAX_RECALL_RESULT_CHARS,
};

#[derive(Debug, Default, Deserialize)]
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
    /// 可选抽屉筛选：`"all"` | `"rules"` | `"tasks"` | `"knowledge"` | `"journal"` | `"raw"`
    #[serde(default)]
    pub(crate) drawer: String,
    /// 是否包含已标作废的历史结论（默认 false）
    #[serde(default)]
    pub(crate) include_superseded: Option<bool>,
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
        "Search the active neural memory agent's 5 drawers (rules/preferences, requirements/study plans, knowledge cards & facts, daily milestone journal, and cold verbatim transcripts) using exact anchors (files/tables/symbols/error codes), weighted search, and linked Agent Teams status. Use project=\"*\" for cross-project search or project=\"<name>\" for a specific project."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, file names, table names, symbols, error codes, or natural-language topic"
                },
                "project": {
                    "type": "string",
                    "description": "Project filter: omit or empty for current working directory; \"*\" for all projects; or a specific project name/path (e.g. \"tsm-server\")"
                },
                "date_range": {
                    "type": "string",
                    "description": "Optional time filter such as \"today\", \"yesterday\", \"this week\", \"last 3 days\", \"2026-11-15\", etc."
                },
                "drawer": {
                    "type": "string",
                    "description": "Optional drawer filter: \"all\" (default), \"rules\", \"tasks\", \"knowledge\", \"journal\", or \"raw\""
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": "Whether to also include superseded/obsolete historical rules or card facts (default false)"
                },
                "include_raw": {
                    "type": "boolean",
                    "description": "Whether to also return verbatim cold-storage User/Assistant transcripts for matched turns (default false)"
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
        let started = Instant::now();

        let res = tokio::task::spawn_blocking(move || {
            with_read(|conn, _| execute_memory_recall_inner(conn, &sid, &working_dir, parsed))
        })
        .await;

        match res {
            Ok(Ok((text, hit_nodes, hit_profiles))) => {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                tracing::debug!(
                    elapsed_ms = format_args!("{elapsed_ms:.2}"),
                    hit_nodes = hit_nodes.len(),
                    hit_profiles = hit_profiles.len(),
                    "neural_memory: memory_recall completed"
                );
                if (!hit_nodes.is_empty() || !hit_profiles.is_empty()) && !cfg!(test) {
                    tokio::task::spawn_blocking(move || {
                        let _ = with_write(|store| {
                            record_memory_hits_conn(&store.conn, &hit_nodes, &hit_profiles);
                            Ok(())
                        });
                    });
                }
                ToolResult {
                    call_id: String::new(),
                    content: text,
                    is_error: false,
                    images: Vec::new(),
                }
            }
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

#[cfg(test)]
pub(crate) fn execute_memory_recall_conn(
    conn: &Connection,
    session_id: &str,
    working_dir: &Path,
    args: MemoryRecallArgs,
) -> Result<String, String> {
    let (text, _, _) = execute_memory_recall_inner(conn, session_id, working_dir, args)?;
    Ok(text)
}

fn execute_memory_recall_inner(
    conn: &Connection,
    session_id: &str,
    working_dir: &Path,
    args: MemoryRecallArgs,
) -> Result<(String, Vec<String>, Vec<String>), String> {
    let Some(agent) = resolve_agent_for_session_conn(conn, session_id) else {
        return Ok((
            "No neural memory agent is currently active for this session (activeAgent = none)."
                .to_string(),
            Vec::new(),
            Vec::new(),
        ));
    };

    let limit = args.limit.unwrap_or(6).clamp(1, 15);
    let drawer = args.drawer.trim().to_lowercase();
    let drawer_allows = |name: &str| drawer.is_empty() || drawer == "all" || drawer == name;
    let include_raw = args.include_raw.unwrap_or(false) || drawer == "raw";

    let current_proj_name = normalize_project_name(working_dir);
    let current_proj_key = normalize_project_key(working_dir);
    let raw_proj = args.project.trim();

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
    let mut plan = analyze_query(args.query.trim(), &[]);
    if !args.date_range.trim().is_empty() {
        plan.date_range = parse_temporal_range(args.date_range.trim());
    }
    let include_superseded = args.include_superseded.unwrap_or(plan.wants_superseded);
    let has_query = !args.query.trim().is_empty();

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
    let mut hit_nodes: Vec<String> = Vec::new();
    let mut hit_profiles: Vec<String> = Vec::new();

    let (s_key, s_name) = scope.map(|s| (s.key, s.name)).unwrap_or(("", ""));

    // 抽屉 1：长期规矩与习惯
    if drawer_allows("rules") {
        let matched_profiles = if has_query {
            let mut ranked =
                query_profiles_ranked_conn(conn, &agent.id, &plan, scope, include_superseded, 8);
            if ranked.is_empty() {
                let q_low = args.query.trim().to_lowercase();
                let all_p = load_profiles_scoped_conn(conn, &agent.id, s_key, s_name, 64)?;
                all_p
                    .into_iter()
                    .filter(|p| p.text.to_lowercase().contains(&q_low) || p.pinned)
                    .take(8)
                    .collect()
            } else {
                ranked.drain(..).map(|r| r.profile).collect()
            }
        } else {
            load_profiles_scoped_conn(conn, &agent.id, s_key, s_name, 8)?
        };
        if !matched_profiles.is_empty() {
            for p in &matched_profiles {
                hit_profiles.push(p.id.clone());
            }
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
                        let st = if p.is_active() {
                            String::new()
                        } else {
                            format!("[superseded {}] ", p.invalid_at)
                        };
                        format!("- {st}{tag}{}", p.text)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
    }

    // 抽屉 2：需求与学习进度本 + Team 跨端状态
    let mut team_run_ids: Vec<String> = Vec::new();
    if drawer_allows("tasks") {
        let mut tasks = load_open_tasks_scoped_conn(conn, &agent.id, s_key, s_name, limit)?;
        if has_query {
            let mut seen_t: HashSet<String> = tasks.iter().map(|t| t.id.clone()).collect();
            for rt in query_tasks_ranked_conn(conn, &agent.id, &plan, scope, limit) {
                if seen_t.insert(rt.task.id.clone()) {
                    tasks.push(rt.task);
                }
            }
            tasks.truncate(limit);
        }
        if !tasks.is_empty() {
            for t in &tasks {
                if !t.team_run_id.trim().is_empty() && !team_run_ids.contains(&t.team_run_id) {
                    team_run_ids.push(t.team_run_id.clone());
                }
            }
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
                        let next = if !t.next_step.trim().is_empty() {
                            format!(" | next: {}", t.next_step.trim())
                        } else if !t.remaining_note.trim().is_empty() {
                            format!(" | remaining: {}", t.remaining_note.trim())
                        } else {
                            String::new()
                        };
                        let team = if t.team_run_id.is_empty() {
                            String::new()
                        } else {
                            format!(" [team={}]", t.team_run_id)
                        };
                        format!(
                            "- ({}/{}, id={}) {p}{}{next}{team}",
                            t.item_kind, t.status, t.id, t.title
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        let mut team_lines = Vec::new();
        let mut seen_runs = HashSet::new();
        for rid in &team_run_ids {
            if seen_runs.insert(rid.clone()) {
                if let Some(ov) = lookup_team_run_overview(rid) {
                    team_lines.push(format_team_overview_line(&ov));
                }
            }
        }
        if plan.wants_progress && team_lines.len() < 3 {
            for ov in list_recent_team_runs_for_project(working_dir, 3) {
                if seen_runs.insert(ov.run_id.clone()) {
                    team_lines.push(format_team_overview_line(&ov));
                }
            }
        }
        if !team_lines.is_empty() {
            out.push(format!(
                "## Linked Agent Teams Progress\n{}",
                team_lines.join("\n")
            ));
        }
    }

    // 抽屉 3：知识与经验活卡片（含独立要点 + 1-hop 脑图关联邻居）
    if drawer_allows("knowledge") {
        let ranked_nodes = query_nodes_ranked_conn(conn, &agent.id, &plan, scope, limit);
        if !ranked_nodes.is_empty() {
            let mut lines = Vec::new();
            for rn in &ranked_nodes {
                hit_nodes.push(rn.node.id.clone());
                let p = if rn.node.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", rn.node.project)
                };
                let mut item = format!(
                    "- {p}{} ({}%): {}",
                    rn.node.name, rn.node.mastery, rn.node.summary
                );
                let fact_lines: Vec<String> = rn
                    .node
                    .facts
                    .iter()
                    .filter(|f| include_superseded || f.is_active())
                    .filter(|f| f.text.trim() != rn.node.summary.trim())
                    .take(5)
                    .map(|f| {
                        let st = if f.is_active() {
                            String::new()
                        } else {
                            format!("[superseded {}] ", f.invalid_at)
                        };
                        format!("  * {st}[{}] {}", f.kind, f.text)
                    })
                    .collect();
                if !fact_lines.is_empty() {
                    item.push('\n');
                    item.push_str(&fact_lines.join("\n"));
                }
                lines.push(item);
            }
            for (nb, rel) in expand_top_node_neighbors_conn(conn, &agent.id, &ranked_nodes, 2) {
                let p = if nb.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", nb.project)
                };
                lines.push(format!(
                    "- [linked:{rel}] {p}{} ({}%): {}",
                    nb.name, nb.mastery, nb.summary
                ));
            }
            out.push(format!("## L2 Knowledge Nodes\n{}", lines.join("\n")));
        }
    }

    // 抽屉 4：每日精华大事记
    let mut timelines = Vec::new();
    if drawer_allows("journal") || include_raw {
        let ranked_tl = query_timeline_ranked_conn(
            conn,
            &agent.id,
            &plan,
            scope,
            plan.date_range.as_ref(),
            limit,
        );
        timelines = ranked_tl.into_iter().map(|r| r.event).collect();

        if timelines.is_empty() && (!has_query || plan.date_range.is_some()) {
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
                        "((project_key != '' AND project_key = ?{pk}) OR (project_key = '' AND LOWER(project) = LOWER(?{pn})) OR (project_key = '' AND project = ''))"
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
            if let Some(ref dr) = plan.date_range {
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
                        user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by,
                        kind, team_run_id, node_ids_json
                 FROM timeline WHERE {} ORDER BY ts DESC LIMIT ?{pidx}",
                conds.join(" AND ")
            );
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Ok(rows) =
                    stmt.query_map(rusqlite::params_from_iter(pvals), map_timeline_row)
                {
                    timelines = rows.flatten().collect();
                }
            }
        }

        if drawer_allows("journal") && !timelines.is_empty() {
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
    }

    // 抽屉 5：原始对话冷仓库（仅按需溯源，无 FTS 膨胀）
    if include_raw {
        let mut raws = Vec::new();
        let mut seen_r = HashSet::new();
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
        if raws.len() < 4 {
            let mut terms = plan.anchors.clone();
            terms.extend(plan.core_terms.iter().take(3).cloned());
            for r in query_raw_turns_cold_conn(
                conn,
                &agent.id,
                &terms,
                scope,
                plan.date_range.as_ref(),
                4,
            ) {
                if seen_r.insert(r.turn_id.clone()) {
                    raws.push(r);
                    if raws.len() >= 4 {
                        break;
                    }
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
    let final_text = if joined.chars().count() > MAX_RECALL_RESULT_CHARS {
        format!(
            "{}\n...(truncated)",
            joined
                .chars()
                .take(MAX_RECALL_RESULT_CHARS)
                .collect::<String>()
        )
    } else {
        joined
    };
    Ok((final_text, hit_nodes, hit_profiles))
}
