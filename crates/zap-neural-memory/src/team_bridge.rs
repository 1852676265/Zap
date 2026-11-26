//! 只读 Team 协作回流桥：无侵入读取 `agent-teams/teams.db`，实现两项核心联动：
//! 1. **完结精准回流**：跳过 Team 执行/返工中途的噪音，只在子任务验收通过或任务单结束/暂停时，
//!    按角色（SQL / 后端 / 前端）将最终定案的 DDL、接口契约、文件路牌与需求进度幂等写入对应智能体的 5 个抽屉；
//! 2. **跨端总进度透视**：平时单智能体严格隔离，但在查询挂有 `team_run_id` 的需求进度时，
//!    只读查一眼该 Team 单的各端实时状态（前端/后端/SQL 哪端完成、哪端待返工），杜绝信息盲区。

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, UNIX_EPOCH};

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

use super::search::normalize_node_key;
use super::store::{
    insert_timeline_conn, load_nodes_conn, load_tasks_conn, upsert_node_conn, upsert_task_conn,
};
use super::types::{
    default_windows, new_id, normalize_project_key, normalize_project_key_str,
    normalize_project_name, now_ts_minute, NeuralNode, NeuralTask, NodeEvent, NodeFact,
    TimelineEvent,
};

#[derive(Debug, Clone, Default)]
pub struct TeamTaskDelivery {
    pub run_id: String,
    pub run_title: String,
    pub run_mode: String,
    pub run_status: String,
    pub task_id: String,
    pub domain: String,
    pub task_title: String,
    pub task_status: String,
    pub agent_id: String,
    pub working_dir: String,
    pub session_id: String,
    pub sql_schema: String,
    pub interface_spec: String,
    pub contract_summary: String,
    pub verify_feedback: String,
    pub synthesis_report: String,
    pub repair_count: u32,
    pub touched_files: Vec<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub(crate) struct TeamRunOverview {
    pub(crate) run_id: String,
    pub(crate) title: String,
    pub(crate) status: String,
    pub(crate) stage: String,
    pub(crate) updated_at: String,
    pub(crate) tasks: Vec<TeamTaskBrief>,
}

#[derive(Debug, Clone)]
pub(crate) struct TeamTaskBrief {
    pub(crate) domain: String,
    pub(crate) title: String,
    pub(crate) status: String,
    pub(crate) agent_label: String,
}

/// 识别 `agent_teams` 管理的子会话（执行/返工期跳过常规逐轮草稿写入，避免过程噪音污染）。
pub fn is_team_managed_session(session_id: &str) -> bool {
    session_id.trim().starts_with("team_")
}

fn teams_db_path() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    let dir = if let Ok(custom) = std::env::var("ZAP_AGENT_TEAMS_DIR") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            PathBuf::from(trimmed)
        } else {
            zap_config::Config::config_dir().join("agent-teams")
        }
    } else {
        zap_config::Config::config_dir().join("agent-teams")
    };
    let db = dir.join("teams.db");
    db.is_file().then_some(db)
}

fn open_teams_readonly(db_path: &Path) -> Option<Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(db_path, flags).ok()?;
    let _ = conn.busy_timeout(Duration::from_millis(800));
    let _ = conn.execute_batch("PRAGMA query_only = ON;");
    Some(conn)
}

fn file_stamp(p: &Path) -> Option<(u128, u64)> {
    let meta = std::fs::metadata(p).ok()?;
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Some((mtime_nanos, meta.len()))
}

fn wal_path_for(db_path: &Path) -> PathBuf {
    let mut os = db_path.as_os_str().to_os_string();
    os.push("-wal");
    PathBuf::from(os)
}

/// 同时采样主库 `teams.db` 与 `teams.db-wal` 的 `(mtime_nanos, len)` 组合哈希。
/// SQLite WAL 模式下未 checkpoint 前主库 mtime 不变，仅采样主库会导致漏归档。
pub(crate) fn teams_db_fingerprint(db_path: &Path) -> u64 {
    let Some(main_stamp) = file_stamp(db_path) else {
        return 0;
    };
    let wal_stamp = file_stamp(&wal_path_for(db_path));
    let mut hasher = DefaultHasher::new();
    main_stamp.hash(&mut hasher);
    wal_stamp.hash(&mut hasher);
    let fp = hasher.finish();
    if fp == 0 {
        1
    } else {
        fp
    }
}

/// 优先提取 Markdown 中的 ```sql 代码块作为核心 DDL 摘要，避免长篇说明挤掉表名锚点。
fn extract_sql_ddl_summary(raw: &str, max_chars: usize) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut blocks = Vec::new();
    let lower = trimmed.to_ascii_lowercase();
    let mut cursor = 0usize;
    while let Some(rel) = lower[cursor..].find("```sql") {
        let start = cursor + rel + 6;
        if start >= trimmed.len() {
            break;
        }
        if let Some(end_rel) = trimmed[start..].find("```") {
            let block = trimmed[start..start + end_rel].trim();
            if !block.is_empty() {
                blocks.push(block);
            }
            cursor = start + end_rel + 3;
        } else {
            break;
        }
    }
    if blocks.is_empty() {
        truncate_chars(trimmed, max_chars)
    } else {
        truncate_chars(&blocks.join("\n"), max_chars)
    }
}

/// 提取同任务单契约/决策要点的槽位前缀（如 `[Team r1] 前后端锁定契约：`），
/// 防止同一轮交付中 DDL 与接口契约互相覆盖，同时确保二期修订准确作废同类旧契约。
fn fact_slot_prefix<'a>(text: &'a str, run_prefix: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(run_prefix)?;
    let colon_pos = rest.find('：')?;
    let end = run_prefix.len() + colon_pos + '：'.len_utf8();
    Some(&text[..end])
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        format!("{}...", trimmed.chars().take(max_chars).collect::<String>())
    }
}

/// 供 `agent_teams` 直接推送或由只读桥接调用：将单个已验收/收尾的 Team 子任务成果幂等写入对应智能体的 5 抽屉。
pub(crate) fn apply_team_delivery_conn(
    conn: &Connection,
    d: &TeamTaskDelivery,
) -> Result<bool, String> {
    let agent_id = d.agent_id.trim();
    if agent_id.is_empty() || agent_id == "none" {
        return Ok(false);
    }

    // 幂等检查：相同 (team_run_id, team_task_id, updated_at) 跳过重复写入
    let prev_ts: Option<String> = conn
        .prepare_cached(
            "SELECT synced_updated_at FROM team_sync_log WHERE team_run_id = ?1 AND team_task_id = ?2",
        )
        .map_err(|e| e.to_string())?
        .query_row(params![d.run_id, d.task_id], |r| r.get(0))
        .optional()
        .map_err(|e| e.to_string())?;
    if prev_ts.as_deref() == Some(d.updated_at.as_str()) {
        return Ok(false);
    }

    let ts = if d.updated_at.trim().is_empty() {
        now_ts_minute()
    } else {
        d.updated_at.chars().take(16).collect()
    };
    let wd_path = Path::new(d.working_dir.trim());
    let project = normalize_project_name(wd_path);
    let project_key = normalize_project_key_str(d.working_dir.trim());

    let is_done = d.task_status == "completed";
    let task_status = if is_done { "done" } else { "in_progress" };
    let task_title = if d.run_title.trim().is_empty() {
        d.task_title.trim().to_string()
    } else {
        format!("{} · {}", d.run_title.trim(), d.task_title.trim())
    };
    let progress_note = if is_done {
        if d.verify_feedback.trim().is_empty() {
            format!("Team 子任务（{}）验收通过并完成交付", d.domain)
        } else {
            truncate_chars(&d.verify_feedback, 260)
        }
    } else {
        format!("Team 子任务当前状态：{}", d.task_status)
    };
    let remaining_note = if is_done {
        String::new()
    } else if !d.verify_feedback.trim().is_empty() {
        truncate_chars(&d.verify_feedback, 220)
    } else {
        "等待在 Team 协作单中继续收尾或修复".to_string()
    };

    let real_session_id = {
        let sid = d.session_id.trim();
        if !sid.is_empty() {
            sid.to_string()
        } else {
            format!("team_{}_{}", d.run_id, d.task_id)
        }
    };

    // 1. 抽屉 2（需求进度本）：按 (team_run_id, team_task_id) 幂等更新或新建进度卡
    let existing_tasks = load_tasks_conn(conn, Some(agent_id))?;
    let existing_task = existing_tasks
        .into_iter()
        .find(|t| t.team_run_id == d.run_id && t.team_task_id == d.task_id);
    let task = match existing_task {
        Some(mut old) => {
            old.title = task_title.clone();
            old.status = task_status.to_string();
            old.progress_note = progress_note.clone();
            old.remaining_note = remaining_note.clone();
            if !d.session_id.trim().is_empty() {
                old.session_id = real_session_id.clone();
            }
            old.next_step = if is_done {
                String::new()
            } else {
                "在 Team 中检查该子任务状态并继续推进".into()
            };
            for f in &d.touched_files {
                if !old.touched_files.contains(f) {
                    old.touched_files.push(f.clone());
                }
            }
            old.updated_at = ts.clone();
            if is_done {
                old.done_at = ts.clone();
            }
            old
        }
        None => NeuralTask {
            id: new_id("task"),
            agent: agent_id.to_string(),
            project: project.clone(),
            project_key: project_key.clone(),
            working_dir: d.working_dir.clone(),
            title: task_title.clone(),
            progress_note: progress_note.clone(),
            remaining_note: remaining_note.clone(),
            touched_files: d.touched_files.clone(),
            session_id: real_session_id.clone(),
            source_turn_ids: Vec::new(),
            status: task_status.to_string(),
            created_at: ts.clone(),
            updated_at: ts.clone(),
            done_at: if is_done { ts.clone() } else { String::new() },
            item_kind: "requirement".into(),
            next_step: if is_done {
                String::new()
            } else {
                "在 Team 中检查该子任务状态并继续推进".into()
            },
            team_run_id: d.run_id.clone(),
            team_task_id: d.task_id.clone(),
        },
    };
    upsert_task_conn(conn, &task)?;

    // 2. 抽屉 3（知识与经验活卡片）：按角色提取定案成果（DDL / 接口契约 / 会诊结论 / 避坑经验）
    let mut distilled_facts: Vec<NodeFact> = Vec::new();
    let domain_low = d.domain.trim().to_lowercase();
    if matches!(domain_low.as_str(), "sql" | "backend") && !d.sql_schema.trim().is_empty() {
        let ddl_summary = extract_sql_ddl_summary(&d.sql_schema, 420);
        if !ddl_summary.is_empty() {
            distilled_facts.push(NodeFact {
                id: new_id("nf"),
                kind: "contract".into(),
                text: format!("[Team {}] 确认表结构/DDL：{}", d.run_id, ddl_summary),
                status: "active".into(),
                updated_at: ts.clone(),
                ..Default::default()
            });
        }
    }
    if !d.interface_spec.trim().is_empty()
        && matches!(
            domain_low.as_str(),
            "backend" | "frontend" | "mobile" | "qa"
        )
    {
        distilled_facts.push(NodeFact {
            id: new_id("nf"),
            kind: "contract".into(),
            text: format!(
                "[Team {}] 前后端锁定契约：{}",
                d.run_id,
                truncate_chars(&d.interface_spec, 420)
            ),
            status: "active".into(),
            updated_at: ts.clone(),
            ..Default::default()
        });
    }
    if !d.contract_summary.trim().is_empty() {
        distilled_facts.push(NodeFact {
            id: new_id("nf"),
            kind: "decision".into(),
            text: format!(
                "[Team {}] {} 交付契约：{}",
                d.run_id,
                d.task_title.trim(),
                truncate_chars(&d.contract_summary, 360)
            ),
            status: "active".into(),
            updated_at: ts.clone(),
            ..Default::default()
        });
    }
    if !d.verify_feedback.trim().is_empty() && is_done {
        distilled_facts.push(NodeFact {
            id: new_id("nf"),
            kind: "fact".into(),
            text: format!(
                "[Team {}] 验收交付：{}",
                d.run_id,
                truncate_chars(&d.verify_feedback, 240)
            ),
            status: "active".into(),
            updated_at: ts.clone(),
            ..Default::default()
        });
    }
    if d.run_mode.trim().eq_ignore_ascii_case("triage") && !d.synthesis_report.trim().is_empty() {
        distilled_facts.push(NodeFact {
            id: new_id("nf"),
            kind: "fact".into(),
            text: format!(
                "[Team {}] 联合会诊结论：{}",
                d.run_id,
                truncate_chars(&d.synthesis_report, 420)
            ),
            status: "active".into(),
            updated_at: ts.clone(),
            ..Default::default()
        });
    }
    if d.repair_count > 0 && !d.verify_feedback.trim().is_empty() {
        distilled_facts.push(NodeFact {
            id: new_id("nf"),
            kind: "pitfall".into(),
            text: format!(
                "[Team {}] 返工避坑（返工 {} 次）：{}",
                d.run_id,
                d.repair_count,
                truncate_chars(&d.verify_feedback, 260)
            ),
            status: "active".into(),
            updated_at: ts.clone(),
            ..Default::default()
        });
    }

    let mut linked_node_ids = Vec::new();
    if !distilled_facts.is_empty() {
        let card_name = if !d.run_title.trim().is_empty() {
            truncate_chars(d.run_title.trim(), 32)
        } else {
            truncate_chars(d.task_title.trim(), 32)
        };
        let norm_target = normalize_node_key(&card_name);
        let all_nodes = load_nodes_conn(conn, Some(agent_id))?;
        let existing_node = all_nodes.into_iter().find(|n| {
            let same_proj = project_key.is_empty()
                || n.project_key == project_key
                || (n.project_key.is_empty() && n.project.eq_ignore_ascii_case(&project));
            same_proj
                && (normalize_node_key(&n.name) == norm_target
                    || n.aliases
                        .iter()
                        .any(|a| normalize_node_key(a) == norm_target))
        });

        let mut node = existing_node.unwrap_or_else(|| NeuralNode {
            id: new_id(agent_id),
            name: card_name.clone(),
            cat: agent_id.to_string(),
            project: project.clone(),
            project_key: project_key.clone(),
            status: if is_done { "solid" } else { "active" }.into(),
            mastery: if is_done { 88 } else { 70 },
            tier: 0,
            windows: default_windows(),
            summary: String::new(),
            events: Vec::new(),
            updated_at: ts.clone(),
            ..Default::default()
        });

        let run_prefix = format!("[Team {}]", d.run_id);
        let mut inserted_ids: HashSet<String> = HashSet::new();
        for fact in distilled_facts {
            let dup = node
                .facts
                .iter()
                .any(|f| f.is_active() && f.text == fact.text);
            if !dup {
                if matches!(fact.kind.as_str(), "contract" | "decision") {
                    let new_slot = fact_slot_prefix(&fact.text, &run_prefix);
                    for old_fact in &mut node.facts {
                        if !old_fact.is_active()
                            || inserted_ids.contains(&old_fact.id)
                            || old_fact.kind != fact.kind
                            || !old_fact.text.starts_with(&run_prefix)
                        {
                            continue;
                        }
                        let old_slot = fact_slot_prefix(&old_fact.text, &run_prefix);
                        let same_slot = match (new_slot, old_slot) {
                            (Some(ns), Some(os)) => ns == os,
                            _ => true,
                        };
                        if same_slot && old_fact.text != fact.text {
                            old_fact.status = "superseded".into();
                            old_fact.superseded_by = fact.id.clone();
                            old_fact.invalid_at = ts.clone();
                        }
                    }
                }
                inserted_ids.insert(fact.id.clone());
                node.facts.insert(0, fact);
            }
        }
        node.facts.truncate(24);

        let mut tags: HashSet<String> = node.search_tags.iter().cloned().collect();
        tags.insert(d.run_id.clone());
        tags.insert(d.domain.clone());
        if !d.task_title.trim().is_empty() {
            tags.insert(truncate_chars(d.task_title.trim(), 24));
        }
        node.search_tags = tags.into_iter().take(16).collect();

        // 综合当前生效要点生成摘要（保留已有摘要不被单句覆写丢弃）
        let top_active: Vec<String> = node
            .facts
            .iter()
            .filter(|f| f.is_active())
            .take(3)
            .map(|f| truncate_chars(&f.text, 120))
            .collect();
        if !top_active.is_empty() {
            node.summary = top_active.join("；");
        }
        node.updated_at = ts.clone();
        if is_done {
            node.status = "solid".into();
            node.mastery = node.mastery.max(85);
        }
        let ev_text = format!("[Team {}] {}", d.run_id, progress_note);
        if node.events.first().map(|e| e.text.as_str()) != Some(ev_text.as_str()) {
            node.events.insert(
                0,
                NodeEvent {
                    ts: ts.clone(),
                    text: ev_text,
                },
            );
            node.events.truncate(12);
        }
        upsert_node_conn(conn, &node)?;
        linked_node_ids.push(node.id);
    }

    // 3. 抽屉 4（每日精华大事记）：每个完结子任务仅记 1 条定案交付大事记
    insert_timeline_conn(
        conn,
        &TimelineEvent {
            id: format!("tl_team_{}_{}", d.run_id, d.task_id),
            ts: ts.clone(),
            agent: agent_id.to_string(),
            project,
            project_key,
            working_dir: d.working_dir.clone(),
            session_id: real_session_id,
            turn_id: String::new(),
            raw_file: "team".into(),
            user_goal: task_title.clone(),
            summary: format!("[Team 协作交付] {task_title}"),
            ai_outcome: progress_note,
            keywords: vec![d.domain.clone(), d.run_id.clone(), "Team交付".into()],
            touched_files: d.touched_files.clone(),
            distilled_by: "team_bridge".into(),
            kind: "team".into(),
            team_run_id: d.run_id.clone(),
            node_ids: linked_node_ids,
        },
    )?;

    conn.execute(
        "INSERT OR REPLACE INTO team_sync_log (team_run_id, team_task_id, synced_updated_at, agent_id)
         VALUES (?1, ?2, ?3, ?4)",
        params![d.run_id, d.task_id, d.updated_at, agent_id],
    )
    .map_err(|e| e.to_string())?;

    Ok(true)
}

/// 惰性增量同步 `teams.db` 中已验收通过或已暂停/失败的 Team 子任务成果到对应智能体记忆。
/// 若 `teams.db` 不存在或主库与 `-wal` 组合指纹未变化，直接 <0.1ms 短路返回。
pub(crate) fn lazy_sync_team_deliveries_if_changed() -> usize {
    static LAST_FINGERPRINT: AtomicU64 = AtomicU64::new(0);
    let Some(db_path) = teams_db_path() else {
        return 0;
    };
    let fp = teams_db_fingerprint(&db_path);
    if fp == 0 {
        return 0;
    }
    let prev = LAST_FINGERPRINT.load(Ordering::Relaxed);
    if prev == fp {
        return 0;
    }

    let Some(tconn) = open_teams_readonly(&db_path) else {
        return 0;
    };
    let deliveries = read_finalized_deliveries_from_teams_conn(&tconn);
    drop(tconn);

    if deliveries.is_empty() {
        LAST_FINGERPRINT.store(fp, Ordering::Relaxed);
        return 0;
    }

    match super::store::with_write(|store| {
        let mut count = 0usize;
        for d in &deliveries {
            if apply_team_delivery_conn(&store.conn, d).unwrap_or(false) {
                count += 1;
            }
        }
        Ok(count)
    }) {
        Ok(count) => {
            LAST_FINGERPRINT.store(fp, Ordering::Relaxed);
            count
        }
        Err(_) => 0,
    }
}

pub(crate) fn read_finalized_deliveries_from_teams_conn(
    tconn: &Connection,
) -> Vec<TeamTaskDelivery> {
    let mut deliveries: Vec<TeamTaskDelivery> = Vec::new();
    let sql = "
        SELECT r.id, r.title, r.mode, r.status, r.sql_schema_draft, r.interface_spec,
               r.delivery_sql, r.synthesis_report, r.finished_at,
               t.id, t.domain, t.title, t.status, t.agent_id, t.working_dir,
               t.session_id, t.repair_count, t.contract_summary, t.verify_feedback,
               t.touched_files_json, t.updated_at
        FROM team_tasks t
        JOIN team_runs r ON r.id = t.run_id
        WHERE t.agent_id != '' AND t.agent_id != 'none'
          AND (
            t.status = 'completed'
            OR (t.status IN ('failed', 'cancelled') AND r.status IN ('completed', 'paused', 'failed'))
          )
        ORDER BY t.updated_at DESC
        LIMIT 80
    ";
    if let Ok(mut stmt) = tconn.prepare(sql) {
        if let Ok(rows) = stmt.query_map([], |row| {
            let sql_draft: String = row.get::<_, String>(4).unwrap_or_default();
            let delivery_sql: String = row.get::<_, String>(6).unwrap_or_default();
            let run_finished_at: String = row.get::<_, String>(8).unwrap_or_default();
            let tf_str: String = row.get::<_, String>(19).unwrap_or_else(|_| "[]".into());
            let task_updated_at: String = row.get::<_, String>(20).unwrap_or_default();
            // 整单完结后 r.finished_at 会晚于单个子任务的 t.updated_at，据此触发二段同步以补全会诊结论与最终 delivery_sql
            let effective_updated_at =
                if !run_finished_at.trim().is_empty() && run_finished_at > task_updated_at {
                    run_finished_at
                } else {
                    task_updated_at
                };
            Ok(TeamTaskDelivery {
                run_id: row.get(0)?,
                run_title: row.get(1)?,
                run_mode: row.get::<_, String>(2).unwrap_or_default(),
                run_status: row.get(3)?,
                sql_schema: if !delivery_sql.trim().is_empty() {
                    delivery_sql
                } else {
                    sql_draft
                },
                interface_spec: row.get::<_, String>(5).unwrap_or_default(),
                synthesis_report: row.get::<_, String>(7).unwrap_or_default(),
                task_id: row.get(9)?,
                domain: row.get(10)?,
                task_title: row.get(11)?,
                task_status: row.get(12)?,
                agent_id: row.get(13)?,
                working_dir: row.get(14)?,
                session_id: row.get::<_, String>(15).unwrap_or_default(),
                repair_count: row.get::<_, i64>(16).unwrap_or(0).max(0) as u32,
                contract_summary: row.get::<_, String>(17).unwrap_or_default(),
                verify_feedback: row.get::<_, String>(18).unwrap_or_default(),
                touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
                updated_at: effective_updated_at,
            })
        }) {
            deliveries.extend(rows.flatten());
        }
    }

    // 常规全栈流水线不会往 team_tasks 写入 domain = 'sql'，补充扫描 team_runs 中的 SQL 架构师成果
    let sql_runs_query = "
        SELECT r.id, r.title, r.mode, r.status, r.stage,
               r.sql_schema_draft, r.interface_spec, r.delivery_sql, r.synthesis_report,
               r.sql_agent_id, r.sql_working_dir, r.sql_session_id, r.updated_at
        FROM team_runs r
        WHERE r.sql_agent_id != '' AND r.sql_agent_id != 'none'
          AND (TRIM(r.delivery_sql) != '' OR TRIM(r.sql_schema_draft) != '')
          AND (
            r.status IN ('completed', 'paused', 'failed')
            OR r.stage IN ('contracting', 'executing', 'completed')
            OR TRIM(r.delivery_sql) != ''
          )
        ORDER BY r.updated_at DESC
        LIMIT 40
    ";
    if let Ok(mut stmt) = tconn.prepare(sql_runs_query) {
        if let Ok(rows) = stmt.query_map([], |row| {
            let run_id: String = row.get(0)?;
            let run_title: String = row.get(1)?;
            let run_mode: String = row.get::<_, String>(2).unwrap_or_default();
            let run_status: String = row.get(3)?;
            let run_stage: String = row.get::<_, String>(4).unwrap_or_default();
            let sql_draft: String = row.get::<_, String>(5).unwrap_or_default();
            let interface_spec: String = row.get::<_, String>(6).unwrap_or_default();
            let delivery_sql: String = row.get::<_, String>(7).unwrap_or_default();
            let synthesis_report: String = row.get::<_, String>(8).unwrap_or_default();
            let sql_agent_id: String = row.get(9)?;
            let sql_working_dir: String = row.get::<_, String>(10).unwrap_or_default();
            let sql_session_id: String = row.get::<_, String>(11).unwrap_or_default();
            let updated_at: String = row.get::<_, String>(12).unwrap_or_default();
            let task_status = if !delivery_sql.trim().is_empty()
                || run_status == "completed"
                || matches!(
                    run_stage.as_str(),
                    "contracting" | "executing" | "completed"
                ) {
                "completed".to_string()
            } else {
                run_status.clone()
            };
            Ok(TeamTaskDelivery {
                run_id,
                run_title,
                run_mode,
                run_status,
                task_id: "__sql__".into(),
                domain: "sql".into(),
                task_title: "表结构与 DDL 设计".into(),
                task_status,
                agent_id: sql_agent_id,
                working_dir: sql_working_dir,
                session_id: sql_session_id,
                sql_schema: if !delivery_sql.trim().is_empty() {
                    delivery_sql
                } else {
                    sql_draft
                },
                interface_spec,
                contract_summary: String::new(),
                verify_feedback: String::new(),
                synthesis_report,
                repair_count: 0,
                touched_files: Vec::new(),
                updated_at,
            })
        }) {
            for sql_delivery in rows.flatten() {
                let has_explicit_sql_task = deliveries.iter().any(|d| {
                    d.run_id == sql_delivery.run_id
                        && d.domain.eq_ignore_ascii_case("sql")
                        && d.agent_id == sql_delivery.agent_id
                });
                if !has_explicit_sql_task {
                    deliveries.push(sql_delivery);
                }
            }
        }
    }

    deliveries
}

/// 只读透视指定 Team 需求单的跨端总进度（用于单聊回答“这个需求改好没？还剩哪端？”）。
pub(crate) fn lookup_team_run_overview(team_run_id: &str) -> Option<TeamRunOverview> {
    let rid = team_run_id.trim();
    if rid.is_empty() {
        return None;
    }
    let db_path = teams_db_path()?;
    let tconn = open_teams_readonly(&db_path)?;
    lookup_team_run_overview_conn(&tconn, rid)
}

pub(crate) fn lookup_team_run_overview_conn(
    tconn: &Connection,
    run_id: &str,
) -> Option<TeamRunOverview> {
    let (title, status, stage, updated_at): (String, String, String, String) = tconn
        .query_row(
            "SELECT title, status, stage, updated_at FROM team_runs WHERE id = ?1",
            params![run_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()
        .ok()
        .flatten()?;

    let mut tasks = Vec::new();
    if let Ok(mut stmt) = tconn.prepare(
        "SELECT domain, title, status, agent_label FROM team_tasks WHERE run_id = ?1 ORDER BY wave ASC, sort_order ASC",
    ) {
        if let Ok(rows) = stmt.query_map(params![run_id], |r| {
            Ok(TeamTaskBrief {
                domain: r.get(0)?,
                title: r.get(1)?,
                status: r.get(2)?,
                agent_label: r.get(3)?,
            })
        }) {
            tasks = rows.flatten().collect();
        }
    }

    Some(TeamRunOverview {
        run_id: run_id.to_string(),
        title,
        status,
        stage,
        updated_at,
        tasks,
    })
}

/// 只读列出与当前工作目录相关的最近活跃/未完 Team 协作需求单（供进度问答兜底透视）。
pub(crate) fn list_recent_team_runs_for_project(
    working_dir: &Path,
    limit: usize,
) -> Vec<TeamRunOverview> {
    let Some(db_path) = teams_db_path() else {
        return Vec::new();
    };
    let Some(tconn) = open_teams_readonly(&db_path) else {
        return Vec::new();
    };
    let pkey = normalize_project_key(working_dir);
    if pkey.is_empty() {
        return Vec::new();
    }

    let run_ids: Vec<String> = {
        let Ok(mut stmt) = tconn.prepare(
            "SELECT DISTINCT r.id
             FROM team_runs r
             JOIN team_tasks t ON t.run_id = r.id
             WHERE LOWER(REPLACE(RTRIM(t.working_dir, '/\\'), '\\', '/')) = ?1
             ORDER BY r.updated_at DESC
             LIMIT ?2",
        ) else {
            return Vec::new();
        };
        stmt.query_map(params![pkey, limit as i64], |r| r.get::<_, String>(0))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
    };

    run_ids
        .into_iter()
        .filter_map(|rid| lookup_team_run_overview_conn(&tconn, &rid))
        .collect()
}

pub(crate) fn format_team_overview_line(ov: &TeamRunOverview) -> String {
    let parts: Vec<String> = ov
        .tasks
        .iter()
        .map(|t| {
            let who = if t.agent_label.trim().is_empty() {
                t.domain.as_str()
            } else {
                t.agent_label.as_str()
            };
            format!("{who}:{}:{}", t.title, t.status)
        })
        .collect();
    format!(
        "- [Team {}] {} (status={}, stage={}, updated={}) -> {}",
        ov.run_id,
        ov.title,
        ov.status,
        ov.stage,
        ov.updated_at.chars().take(16).collect::<String>(),
        parts.join(" | ")
    )
}
