//! `zap-agent-teams` SQLite WAL 持久化存储、CAS 并发门禁、attempt_id 栅栏与进程重启孤儿自愈。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use rusqlite::{params, Connection, OptionalExtension};

#[cfg(test)]
use super::schema::init_schema;
use super::schema::open_connection;
use super::types::{
    clean_path_str, compute_and_assign_waves, new_team_id, normalize_domain, now_ts,
    truncate_chars_safe, RoleBindingPreset, TeamContract, TeamEvent, TeamGlobalConfig, TeamImage,
    TeamProjectPreset, TeamRun, TeamRunDetail, TeamTask, HARD_CAP_MAX_REPAIRS, MAX_CONTRACT_CHARS,
    MAX_EXCERPT_CHARS,
};

pub(crate) struct TeamStoreInner {
    #[allow(dead_code)]
    pub(crate) data_dir: PathBuf,
    pub(crate) conn: Connection,
}

static TEAM_STORE: OnceLock<Mutex<Option<TeamStoreInner>>> = OnceLock::new();

pub(crate) fn default_data_dir() -> PathBuf {
    #[cfg(test)]
    {
        static TEST_DIR: OnceLock<PathBuf> = OnceLock::new();
        return TEST_DIR
            .get_or_init(|| {
                let d = std::env::temp_dir()
                    .join(format!("zap-agent-teams-test-{}", std::process::id()));
                let _ = std::fs::create_dir_all(&d);
                d
            })
            .clone();
    }
    #[cfg(not(test))]
    {
        if let Ok(custom) = std::env::var("ZAP_AGENT_TEAMS_DIR") {
            let trimmed = custom.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
        zap_config::Config::config_dir().join("agent-teams")
    }
}

pub(crate) fn with_team_store<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut TeamStoreInner) -> Result<R, String>,
{
    let lock = TEAM_STORE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let dir = default_data_dir();
        let conn = open_connection(&dir)?;
        reconcile_interrupted_runs_on_boot(&conn);
        *guard = Some(TeamStoreInner {
            data_dir: dir,
            conn,
        });
    }
    let inner = guard.as_mut().expect("team store initialized");
    f(inner)
}

#[cfg(test)]
pub(crate) fn setup_mem_team_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    init_schema(&conn).unwrap();
    conn
}

/// 进程启动孤儿自愈：上次宿主退出时若仍有 `running` 态任务单，平滑收敛为 `paused`，允许用户一键续跑或重新生成设计。
fn reconcile_interrupted_runs_on_boot(conn: &Connection) {
    let now = now_ts();
    let mut stmt = match conn.prepare("SELECT id, stage FROM team_runs WHERE status = 'running'") {
        Ok(s) => s,
        Err(_) => return,
    };
    let rows: Vec<(String, String)> = match stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))) {
        Ok(rows) => rows.flatten().collect(),
        Err(_) => Vec::new(),
    };
    for (run_id, stage) in rows {
        let next_status = if matches!(stage.as_str(), "planning" | "sql_designing") {
            "paused"
        } else {
            "paused"
        };
        let _ = conn.execute(
            "UPDATE team_runs SET status = ?1, revision = revision + 1, updated_at = ?2 WHERE id = ?3",
            params![next_status, now, run_id],
        );
        let _ = conn.execute(
            "UPDATE team_tasks
             SET status = 'ready', attempt_id = '', updated_at = ?1
             WHERE run_id = ?2 AND status IN ('running', 'verifying', 'repairing')",
            params![now, run_id],
        );
        let _ = append_event_conn(
            conn,
            &run_id,
            "run_paused_recovery",
            "",
            "system",
            "检测到宿主重启，已自动保护现场并暂停当前任务，点击继续即可无缝恢复。",
            "",
        );
    }
}

// ---------------------------------------------------------------------------
// Global Config & Role Binding Presets
// ---------------------------------------------------------------------------

pub(crate) fn load_config_conn(conn: &Connection) -> TeamGlobalConfig {
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'team_config'",
            [],
            |r| r.get(0),
        )
        .optional()
        .ok()
        .flatten();
    let mut cfg = raw
        .and_then(|s| serde_json::from_str::<TeamGlobalConfig>(&s).ok())
        .unwrap_or_default();
    cfg.max_repairs = cfg.max_repairs.min(HARD_CAP_MAX_REPAIRS);
    cfg.max_parallel_per_wave = cfg.max_parallel_per_wave.clamp(1, 6);
    cfg
}

pub(crate) fn save_config_conn(conn: &Connection, cfg: &TeamGlobalConfig) -> Result<(), String> {
    let mut sanitized = cfg.clone();
    sanitized.max_repairs = sanitized.max_repairs.min(HARD_CAP_MAX_REPAIRS);
    sanitized.max_parallel_per_wave = sanitized.max_parallel_per_wave.clamp(1, 6);
    sanitized
        .prompt_templates
        .retain(|k, v| !k.trim().is_empty() && !v.trim().is_empty());
    let raw = serde_json::to_string(&sanitized).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('team_config', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![raw],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn load_role_presets_conn(conn: &Connection) -> Result<Vec<RoleBindingPreset>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT domain, working_dir, agent_id, model, updated_at
             FROM role_presets
             ORDER BY domain ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let wd: String = r.get(1)?;
            Ok(RoleBindingPreset {
                domain: r.get(0)?,
                working_dir: clean_path_str(&wd),
                agent_id: r.get(2)?,
                model: r.get(3)?,
                updated_at: r.get(4)?,
            })
        })
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn delete_role_preset_conn(conn: &Connection, domain: &str) -> Result<(), String> {
    let norm = normalize_domain(domain);
    conn.execute("DELETE FROM role_presets WHERE domain = ?1", params![norm])
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn upsert_role_preset_conn(
    conn: &Connection,
    preset: &RoleBindingPreset,
) -> Result<(), String> {
    let domain = normalize_domain(&preset.domain);
    let now = now_ts();
    let clean_wd = clean_path_str(&preset.working_dir);
    conn.execute(
        "INSERT INTO role_presets (domain, working_dir, agent_id, model, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(domain) DO UPDATE SET
             working_dir = excluded.working_dir,
             agent_id = excluded.agent_id,
             model = excluded.model,
             updated_at = excluded.updated_at",
        params![
            domain,
            clean_wd,
            if preset.agent_id.trim().is_empty() {
                "none"
            } else {
                preset.agent_id.trim()
            },
            preset.model.trim(),
            now
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn load_custom_projects_conn(
    conn: &Connection,
) -> Result<Vec<TeamProjectPreset>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, name, domain, working_dir, agent_id, avatar_id, model, notes, sort_order, updated_at
             FROM team_projects
             ORDER BY sort_order ASC, updated_at DESC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let sort_i64: i64 = r.get(8)?;
            Ok(TeamProjectPreset {
                id: r.get(0)?,
                name: r.get(1)?,
                domain: r.get(2)?,
                working_dir: r.get(3)?,
                agent_id: r.get(4)?,
                avatar_id: r.get(5)?,
                model: r.get(6)?,
                notes: r.get(7)?,
                sort_order: usize::try_from(sort_i64.max(0)).unwrap_or(0),
                updated_at: r.get(9)?,
            })
        })
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn upsert_custom_project_conn(
    conn: &Connection,
    proj: &TeamProjectPreset,
) -> Result<TeamProjectPreset, String> {
    let clean_name = proj.name.trim();
    if clean_name.is_empty() {
        return Err("请输入项目名称".to_string());
    }
    let id = if proj.id.trim().is_empty() {
        new_team_id("prj")
    } else {
        proj.id.trim().to_string()
    };
    let domain = normalize_domain(&proj.domain);
    let working_dir = clean_path_str(&proj.working_dir);
    let agent_id = if proj.agent_id.trim().is_empty() {
        "none".to_string()
    } else {
        proj.agent_id.trim().to_string()
    };
    let avatar_id = proj.avatar_id.trim().to_string();
    let model = proj.model.trim().to_string();
    let notes = proj.notes.trim().to_string();
    let now = now_ts();
    let sort_order = proj.sort_order;

    conn.execute(
        "INSERT INTO team_projects (
            id, name, domain, working_dir, agent_id, avatar_id, model, notes, sort_order, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            domain = excluded.domain,
            working_dir = excluded.working_dir,
            agent_id = excluded.agent_id,
            avatar_id = excluded.avatar_id,
            model = excluded.model,
            notes = excluded.notes,
            sort_order = excluded.sort_order,
            updated_at = excluded.updated_at",
        params![
            id,
            clean_name,
            domain,
            working_dir,
            agent_id,
            avatar_id,
            model,
            notes,
            sort_order as i64,
            now
        ],
    )
    .map_err(|e| e.to_string())?;

    Ok(TeamProjectPreset {
        id,
        name: clean_name.to_string(),
        domain,
        working_dir,
        agent_id,
        avatar_id,
        model,
        notes,
        sort_order,
        updated_at: now,
    })
}

pub(crate) fn delete_custom_project_conn(conn: &Connection, id: &str) -> Result<(), String> {
    conn.execute(
        "DELETE FROM team_projects WHERE id = ?1",
        params![id.trim()],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Runs, Tasks, Contracts & Events CRUD + CAS / Fencing Primitives
// ---------------------------------------------------------------------------

fn row_to_run(r: &rusqlite::Row<'_>) -> rusqlite::Result<TeamRun> {
    let rev: i64 = r.get(5)?;
    let auto_verify: i64 = r.get(6)?;
    let max_repairs: i64 = r.get(7)?;
    let requires_db_i64: i64 = r.get(15)?;
    let images_json: String = r.get(20)?;
    let goal_images: Vec<TeamImage> = serde_json::from_str(&images_json).unwrap_or_default();
    let sql_wd_raw: String = r.get(21)?;
    let overrides_json: String = r.get(25)?;
    let prompt_overrides: BTreeMap<String, String> =
        serde_json::from_str(&overrides_json).unwrap_or_default();
    Ok(TeamRun {
        id: r.get(0)?,
        title: r.get(1)?,
        user_goal: r.get(2)?,
        mode: r.get(3)?,
        status: r.get(4)?,
        stage: r.get(14)?,
        requires_db: requires_db_i64 != 0,
        execution_plan: r.get(16)?,
        sql_schema_draft: r.get(17)?,
        interface_spec: r.get(18)?,
        delivery_sql: r.get(19)?,
        sql_working_dir: clean_path_str(&sql_wd_raw),
        sql_agent_id: r.get(22)?,
        sql_model: r.get(23)?,
        sql_session_id: r.get(24)?,
        prompt_overrides,
        goal_images,
        revision: u64::try_from(rev.max(0)).unwrap_or(1),
        auto_verify: auto_verify != 0,
        max_repairs: u32::try_from(max_repairs.max(0))
            .unwrap_or(2)
            .min(HARD_CAP_MAX_REPAIRS),
        lead_model: r.get(8)?,
        source_triage_id: r.get(9)?,
        synthesis_report: r.get(10)?,
        created_at: r.get(11)?,
        updated_at: r.get(12)?,
        finished_at: r.get(13)?,
    })
}

fn row_to_task(r: &rusqlite::Row<'_>) -> rusqlite::Result<TeamTask> {
    let deps_json: String = r.get(6)?;
    let wave_i64: i64 = r.get(7)?;
    let sort_i64: i64 = r.get(8)?;
    let attempt_i64: i64 = r.get(15)?;
    let repair_i64: i64 = r.get(17)?;
    let files_json: String = r.get(22)?;
    let images_json: String = r.get(31)?;
    Ok(TeamTask {
        id: r.get(0)?,
        run_id: r.get(1)?,
        domain: r.get(2)?,
        title: r.get(3)?,
        description: r.get(4)?,
        acceptance_criteria: r.get(5)?,
        depends_on: serde_json::from_str(&deps_json).unwrap_or_default(),
        wave: usize::try_from(wave_i64.max(1)).unwrap_or(1),
        sort_order: usize::try_from(sort_i64.max(0)).unwrap_or(0),
        project_slot_id: r.get(27)?,
        project_name: r.get(28)?,
        avatar_id: r.get(29)?,
        working_dir: r.get(9)?,
        project_hash: r.get(10)?,
        agent_id: r.get(11)?,
        agent_label: r.get(12)?,
        model: r.get(13)?,
        session_id: r.get(14)?,
        custom_prompt: r.get(30)?,
        images: serde_json::from_str(&images_json).unwrap_or_default(),
        revision_note: r.get(32)?,
        status: r.get(23)?,
        attempt: u32::try_from(attempt_i64.max(0)).unwrap_or(0),
        attempt_id: r.get(16)?,
        repair_count: u32::try_from(repair_i64.max(0)).unwrap_or(0),
        contract_summary: r.get(18)?,
        last_output_excerpt: r.get(19)?,
        verify_verdict: r.get(20)?,
        verify_feedback: r.get(21)?,
        touched_files: serde_json::from_str(&files_json).unwrap_or_default(),
        started_at: r.get(24)?,
        updated_at: r.get(25)?,
        finished_at: r.get(26)?,
    })
}

pub(crate) fn list_runs_conn(conn: &Connection, limit: usize) -> Result<Vec<TeamRun>, String> {
    let lim = limit.clamp(1, 200) as i64;
    let mut stmt = conn
        .prepare(
            "SELECT id, title, user_goal, mode, status, revision, auto_verify, max_repairs,
                    lead_model, source_triage_id, synthesis_report, created_at, updated_at, finished_at,
                    stage, requires_db, execution_plan, sql_schema_draft, interface_spec, delivery_sql, goal_images_json,
                    sql_working_dir, sql_agent_id, sql_model, sql_session_id, prompt_overrides_json
             FROM team_runs
             ORDER BY updated_at DESC
             LIMIT ?1",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![lim], row_to_run)
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn load_run_conn(conn: &Connection, run_id: &str) -> Result<Option<TeamRun>, String> {
    conn.query_row(
        "SELECT id, title, user_goal, mode, status, revision, auto_verify, max_repairs,
                lead_model, source_triage_id, synthesis_report, created_at, updated_at, finished_at,
                stage, requires_db, execution_plan, sql_schema_draft, interface_spec, delivery_sql, goal_images_json,
                sql_working_dir, sql_agent_id, sql_model, sql_session_id, prompt_overrides_json
         FROM team_runs
         WHERE id = ?1",
        params![run_id],
        row_to_run,
    )
    .optional()
    .map_err(|e| e.to_string())
}

pub(crate) fn load_tasks_for_run_conn(
    conn: &Connection,
    run_id: &str,
) -> Result<Vec<TeamTask>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, run_id, domain, title, description, acceptance_criteria,
                    depends_on_json, wave, sort_order, working_dir, project_hash,
                    agent_id, agent_label, model, session_id, attempt, attempt_id,
                    repair_count, contract_summary, last_output_excerpt, verify_verdict,
                    verify_feedback, touched_files_json, status, started_at, updated_at, finished_at,
                    project_slot_id, project_name, avatar_id, custom_prompt, images_json, revision_note
             FROM team_tasks
             WHERE run_id = ?1
             ORDER BY wave ASC, sort_order ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![run_id], row_to_task)
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn load_task_conn(conn: &Connection, task_id: &str) -> Result<Option<TeamTask>, String> {
    conn.query_row(
        "SELECT id, run_id, domain, title, description, acceptance_criteria,
                depends_on_json, wave, sort_order, working_dir, project_hash,
                agent_id, agent_label, model, session_id, attempt, attempt_id,
                repair_count, contract_summary, last_output_excerpt, verify_verdict,
                verify_feedback, touched_files_json, status, started_at, updated_at, finished_at,
                project_slot_id, project_name, avatar_id, custom_prompt, images_json, revision_note
         FROM team_tasks
         WHERE id = ?1",
        params![task_id],
        row_to_task,
    )
    .optional()
    .map_err(|e| e.to_string())
}

pub(crate) fn load_contracts_for_run_conn(
    conn: &Connection,
    run_id: &str,
) -> Result<Vec<TeamContract>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, run_id, from_task_id, domain, title, schema_or_api, notes, created_at
             FROM team_contracts
             WHERE run_id = ?1
             ORDER BY created_at ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![run_id], |r| {
            Ok(TeamContract {
                id: r.get(0)?,
                run_id: r.get(1)?,
                from_task_id: r.get(2)?,
                domain: r.get(3)?,
                title: r.get(4)?,
                schema_or_api: r.get(5)?,
                notes: r.get(6)?,
                created_at: r.get(7)?,
            })
        })
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn load_task_statuses_by_run_conn(
    conn: &Connection,
    run_ids: &[String],
) -> Result<std::collections::HashMap<String, Vec<String>>, String> {
    use std::collections::HashMap;
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    if run_ids.is_empty() {
        return Ok(map);
    }
    let placeholders = (1..=run_ids.len())
        .map(|idx| format!("?{idx}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT run_id, status
         FROM team_tasks
         WHERE run_id IN ({placeholders})
         ORDER BY wave ASC, sort_order ASC"
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let params = rusqlite::params_from_iter(run_ids.iter());
    let rows = stmt
        .query_map(params, |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })
        .map_err(|e| e.to_string())?;
    for row in rows.flatten() {
        map.entry(row.0).or_default().push(row.1);
    }
    Ok(map)
}

pub(crate) fn append_task_note_conn(
    conn: &Connection,
    run_id: &str,
    task_id: &str,
    note: &str,
) -> Result<(), String> {
    let now = now_ts();
    let cur = load_task_conn(conn, task_id)?.ok_or_else(|| format!("未找到子任务: {task_id}"))?;
    if cur.run_id != run_id {
        return Err("子任务与当前任务单不匹配".to_string());
    }
    let trimmed = note.trim();
    if trimmed.is_empty() {
        return Err("补充指令不能为空".to_string());
    }
    let next_desc = if cur.description.trim().is_empty() {
        format!("[补充指令 {now}] {trimmed}")
    } else {
        format!("{}\n[补充指令 {now}] {trimmed}", cur.description.trim_end())
    };
    conn.execute(
        "UPDATE team_tasks SET description = ?1, updated_at = ?2 WHERE id = ?3",
        params![next_desc, now, task_id],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE team_runs SET revision = revision + 1, updated_at = ?1 WHERE id = ?2",
        params![now, run_id],
    )
    .ok();
    Ok(())
}

pub(crate) fn update_task_followup_result_conn(
    conn: &Connection,
    task_id: &str,
    excerpt: &str,
    touched_files: &[String],
) -> Result<(), String> {
    let now = now_ts();
    let clean_excerpt = truncate_chars_safe(excerpt, MAX_EXCERPT_CHARS);
    let files_json = serde_json::to_string(touched_files).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "UPDATE team_tasks
         SET last_output_excerpt = ?1, touched_files_json = ?2, updated_at = ?3
         WHERE id = ?4",
        params![clean_excerpt, files_json, now, task_id],
    )
    .map_err(|e| e.to_string())?;
    if let Ok(Some(t)) = load_task_conn(conn, task_id) {
        let _ = conn.execute(
            "UPDATE team_runs SET revision = revision + 1, updated_at = ?1 WHERE id = ?2",
            params![now, t.run_id],
        );
    }
    Ok(())
}

pub(crate) fn load_events_for_run_conn(
    conn: &Connection,
    run_id: &str,
    after_seq: i64,
    limit: usize,
) -> Result<Vec<TeamEvent>, String> {
    let lim = limit.clamp(1, 500) as i64;
    let mut stmt = conn
        .prepare(
            "SELECT seq, run_id, ts, kind, task_id, actor, summary, detail
             FROM team_events
             WHERE run_id = ?1 AND seq > ?2
             ORDER BY seq ASC
             LIMIT ?3",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(params![run_id, after_seq, lim], |r| {
            Ok(TeamEvent {
                seq: r.get(0)?,
                run_id: r.get(1)?,
                ts: r.get(2)?,
                kind: r.get(3)?,
                task_id: r.get(4)?,
                actor: r.get(5)?,
                summary: r.get(6)?,
                detail: r.get(7)?,
            })
        })
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn load_run_detail_conn(
    conn: &Connection,
    run_id: &str,
    after_seq: i64,
) -> Result<Option<TeamRunDetail>, String> {
    let Some(run) = load_run_conn(conn, run_id)? else {
        return Ok(None);
    };
    let tasks = load_tasks_for_run_conn(conn, run_id)?;
    let contracts = load_contracts_for_run_conn(conn, run_id)?;
    let events = load_events_for_run_conn(conn, run_id, after_seq, 300)?;
    Ok(Some(TeamRunDetail {
        run,
        tasks,
        contracts,
        events,
    }))
}

pub(crate) fn create_run_with_tasks_conn(
    conn: &mut Connection,
    run: &TeamRun,
    tasks: &[TeamTask],
) -> Result<(), String> {
    let images_json = serde_json::to_string(&run.goal_images).unwrap_or_else(|_| "[]".into());
    let mut sanitized_overrides = run.prompt_overrides.clone();
    sanitized_overrides.retain(|k, v| !k.trim().is_empty() && !v.trim().is_empty());
    let overrides_json =
        serde_json::to_string(&sanitized_overrides).unwrap_or_else(|_| "{}".into());
    let clean_sql_wd = clean_path_str(&run.sql_working_dir);
    let clean_sql_agent = if run.sql_agent_id.trim().is_empty() {
        "none"
    } else {
        run.sql_agent_id.trim()
    };
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "INSERT INTO team_runs (
            id, title, user_goal, mode, status, revision, auto_verify, max_repairs,
            lead_model, source_triage_id, synthesis_report, created_at, updated_at, finished_at,
            stage, requires_db, execution_plan, sql_schema_draft, interface_spec, delivery_sql, goal_images_json,
            sql_working_dir, sql_agent_id, sql_model, sql_session_id, prompt_overrides_json
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
        params![
            run.id,
            run.title,
            run.user_goal,
            run.mode,
            run.status,
            run.revision.max(1) as i64,
            i64::from(run.auto_verify),
            run.max_repairs.min(HARD_CAP_MAX_REPAIRS) as i64,
            run.lead_model,
            run.source_triage_id,
            run.synthesis_report,
            run.created_at,
            run.updated_at,
            run.finished_at,
            run.stage,
            i64::from(run.requires_db),
            run.execution_plan,
            run.sql_schema_draft,
            run.interface_spec,
            run.delivery_sql,
            images_json,
            clean_sql_wd,
            clean_sql_agent,
            run.sql_model.trim(),
            run.sql_session_id.trim(),
            overrides_json,
        ],
    )
    .map_err(|e| e.to_string())?;

    for task in tasks {
        insert_task_row(&tx, task)?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

fn insert_task_row(conn: &Connection, task: &TeamTask) -> Result<(), String> {
    let deps_json = serde_json::to_string(&task.depends_on).unwrap_or_else(|_| "[]".into());
    let files_json = serde_json::to_string(&task.touched_files).unwrap_or_else(|_| "[]".into());
    let images_json = serde_json::to_string(&task.images).unwrap_or_else(|_| "[]".into());
    conn.execute(
        "INSERT INTO team_tasks (
            id, run_id, domain, title, description, acceptance_criteria,
            depends_on_json, wave, sort_order, working_dir, project_hash,
            agent_id, agent_label, model, session_id, status, attempt, attempt_id,
            repair_count, contract_summary, last_output_excerpt, verify_verdict,
            verify_feedback, touched_files_json, started_at, updated_at, finished_at,
            project_slot_id, project_name, avatar_id, custom_prompt, images_json, revision_note
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6,
            ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16, ?17, ?18,
            ?19, ?20, ?21, ?22,
            ?23, ?24, ?25, ?26, ?27,
            ?28, ?29, ?30, ?31, ?32, ?33
         )",
        params![
            task.id,
            task.run_id,
            normalize_domain(&task.domain),
            task.title,
            task.description,
            task.acceptance_criteria,
            deps_json,
            task.wave as i64,
            task.sort_order as i64,
            task.working_dir,
            task.project_hash,
            if task.agent_id.trim().is_empty() {
                "none"
            } else {
                task.agent_id.trim()
            },
            task.agent_label,
            task.model,
            task.session_id,
            task.status,
            task.attempt as i64,
            task.attempt_id,
            task.repair_count as i64,
            task.contract_summary,
            task.last_output_excerpt,
            task.verify_verdict,
            task.verify_feedback,
            files_json,
            task.started_at,
            task.updated_at,
            task.finished_at,
            task.project_slot_id,
            task.project_name,
            task.avatar_id,
            task.custom_prompt,
            images_json,
            task.revision_note,
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn binding_dirs_equal(a: &str, b: &str) -> bool {
    let a = clean_path_str(a);
    let b = clean_path_str(b);
    if a.is_empty() || b.is_empty() {
        return a == b;
    }
    super::sessions::same_project(std::path::Path::new(&a), std::path::Path::new(&b))
}

// Only editable fields come from the client. Runtime and attempt fencing belong to the DB.
fn task_for_draft(client: &TeamTask, old: Option<&TeamTask>, run_id: &str, now: &str) -> TeamTask {
    let mut task = client.clone();
    task.run_id = run_id.to_string();
    task.working_dir = clean_path_str(&client.working_dir);
    let retained = old.filter(|old| binding_dirs_equal(&old.working_dir, &task.working_dir));
    task.session_id = retained.map(|t| t.session_id.clone()).unwrap_or_default();
    task.project_hash = retained.map(|t| t.project_hash.clone()).unwrap_or_default();
    task.status = retained
        .map(|t| t.status.clone())
        .unwrap_or_else(|| "pending".into());
    task.attempt = retained.map(|t| t.attempt).unwrap_or_default();
    task.attempt_id = retained.map(|t| t.attempt_id.clone()).unwrap_or_default();
    task.repair_count = retained.map(|t| t.repair_count).unwrap_or_default();
    task.contract_summary = retained
        .map(|t| t.contract_summary.clone())
        .unwrap_or_default();
    task.last_output_excerpt = retained
        .map(|t| t.last_output_excerpt.clone())
        .unwrap_or_default();
    task.verify_verdict = retained
        .map(|t| t.verify_verdict.clone())
        .unwrap_or_default();
    task.verify_feedback = retained
        .map(|t| t.verify_feedback.clone())
        .unwrap_or_default();
    task.touched_files = retained
        .map(|t| t.touched_files.clone())
        .unwrap_or_default();
    task.revision_note = if client.revision_note.trim().is_empty() {
        retained
            .map(|t| t.revision_note.clone())
            .unwrap_or_default()
    } else {
        client.revision_note.clone()
    };
    task.images = client.images.iter().take(4).cloned().collect();
    task.started_at = retained.map(|t| t.started_at.clone()).unwrap_or_default();
    task.finished_at = retained.map(|t| t.finished_at.clone()).unwrap_or_default();
    task.updated_at = now.to_string();
    task
}

#[derive(Debug, Default)]
pub(crate) struct RunDraftExtras<'a> {
    pub(crate) sql_working_dir: Option<&'a str>,
    pub(crate) sql_agent_id: Option<&'a str>,
    pub(crate) sql_model: Option<&'a str>,
    pub(crate) interface_spec: Option<&'a str>,
    pub(crate) prompt_overrides: Option<&'a BTreeMap<String, String>>,
}

/// CAS 更新任务草稿（校验 `expected_revision`，仅允许在非 `running` 态下整单编辑任务拓扑）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_run_draft_cas_conn(
    conn: &mut Connection,
    run_id: &str,
    expected_revision: Option<u64>,
    title: &str,
    auto_verify: bool,
    max_repairs: u32,
    lead_model: &str,
    requires_db: Option<bool>,
    tasks: &[TeamTask],
    extras: &RunDraftExtras<'_>,
) -> Result<u64, String> {
    let cur = load_run_conn(conn, run_id)?.ok_or_else(|| format!("未找到任务: {run_id}"))?;
    if cur.status == "running" {
        return Err("任务正在执行中，请先暂停后再调整任务结构".to_string());
    }
    if let Some(exp) = expected_revision {
        if exp != cur.revision {
            return Err(format!(
                "任务版本已变更（本地 v{exp} != 最新 v{}），已刷新最新状态",
                cur.revision
            ));
        }
    }

    let next_rev = cur.revision.saturating_add(1);
    let now = now_ts();
    let effective_title = if title.trim().is_empty() {
        cur.title.as_str()
    } else {
        title.trim()
    };
    let req_db_val = requires_db.map(i64::from);
    let sql_wd_val = extras.sql_working_dir.map(clean_path_str);
    let sql_aid_val = extras.sql_agent_id.map(|s| {
        let t = s.trim();
        if t.is_empty() {
            "none".to_string()
        } else {
            t.to_string()
        }
    });
    let sql_mdl_val = extras.sql_model.map(|s| s.trim().to_string());
    let spec_val = extras.interface_spec.map(str::to_string);
    let overrides_val = extras.prompt_overrides.map(|m| {
        let mut sanitized = m.clone();
        sanitized.retain(|k, v| !k.trim().is_empty() && !v.trim().is_empty());
        serde_json::to_string(&sanitized).unwrap_or_else(|_| "{}".into())
    });
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let old_tasks = load_tasks_for_run_conn(&tx, run_id)?;
    let sql_dir_changed = sql_wd_val
        .as_ref()
        .map_or(false, |wd| !binding_dirs_equal(wd, &cur.sql_working_dir));
    let changed = tx
        .execute(
            "UPDATE team_runs
             SET title = ?1, auto_verify = ?2, max_repairs = ?3, lead_model = ?4,
                 requires_db = COALESCE(?5, requires_db),
                 sql_working_dir = COALESCE(?6, sql_working_dir),
                 sql_agent_id = COALESCE(?7, sql_agent_id),
                 sql_model = COALESCE(?8, sql_model),
                 prompt_overrides_json = COALESCE(?9, prompt_overrides_json),
                 sql_session_id = CASE WHEN ?14 THEN '' ELSE sql_session_id END,
                 interface_spec = COALESCE(?15, interface_spec),
                 revision = ?10, updated_at = ?11
             WHERE id = ?12 AND revision = ?13",
            params![
                effective_title,
                i64::from(auto_verify),
                max_repairs.min(HARD_CAP_MAX_REPAIRS) as i64,
                lead_model.trim(),
                req_db_val,
                sql_wd_val,
                sql_aid_val,
                sql_mdl_val,
                overrides_val,
                next_rev as i64,
                now,
                run_id,
                cur.revision as i64,
                sql_dir_changed,
                spec_val,
            ],
        )
        .map_err(|e| e.to_string())?;
    if changed == 0 {
        return Err("并发保存冲突（CAS revision mismatch），请重试".to_string());
    }

    tx.execute("DELETE FROM team_tasks WHERE run_id = ?1", params![run_id])
        .map_err(|e| e.to_string())?;
    for task in tasks {
        let old = old_tasks.iter().find(|old| old.id == task.id);
        let sanitized = task_for_draft(task, old, run_id, &now);
        if old.map_or(false, |old| {
            !binding_dirs_equal(&old.working_dir, &task.working_dir)
        }) {
            tx.execute(
                "DELETE FROM team_contracts WHERE run_id = ?1 AND from_task_id = ?2",
                params![run_id, task.id],
            )
            .map_err(|e| e.to_string())?;
        }
        insert_task_row(&tx, &sanitized)?;
    }
    if let Some(ref sp) = spec_val {
        let _ = tx.execute(
            "UPDATE team_contracts SET schema_or_api = ?1 WHERE run_id = ?2 AND from_task_id = 'lead_contract'",
            params![truncate_chars_safe(sp, MAX_CONTRACT_CHARS), run_id],
        );
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(next_rev)
}

pub(crate) fn set_run_status_conn(
    conn: &Connection,
    run_id: &str,
    status: &str,
    synthesis_report: Option<&str>,
) -> Result<(), String> {
    let now = now_ts();
    let finished = if matches!(status, "completed" | "failed" | "cancelled") {
        now.as_str()
    } else {
        ""
    };
    if let Some(report) = synthesis_report {
        conn.execute(
            "UPDATE team_runs
             SET status = ?1, synthesis_report = ?2, revision = revision + 1,
                 updated_at = ?3, finished_at = CASE WHEN ?4 != '' THEN ?4 ELSE finished_at END
             WHERE id = ?5",
            params![status, report, now, finished, run_id],
        )
        .map_err(|e| e.to_string())?;
    } else {
        conn.execute(
            "UPDATE team_runs
             SET status = ?1, revision = revision + 1,
                 updated_at = ?2, finished_at = CASE WHEN ?3 != '' THEN ?3 ELSE finished_at END
             WHERE id = ?4",
            params![status, now, finished, run_id],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn update_run_design_conn(
    conn: &mut Connection,
    run_id: &str,
    title: Option<&str>,
    status: &str,
    stage: &str,
    requires_db: bool,
    execution_plan: &str,
    sql_schema_draft: &str,
    tasks: Option<&[TeamTask]>,
) -> Result<(), String> {
    let now = now_ts();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    if let Some(t) = title.map(str::trim).filter(|s| !s.is_empty()) {
        tx.execute(
            "UPDATE team_runs
             SET title = ?1, status = ?2, stage = ?3, requires_db = ?4,
                 execution_plan = ?5, sql_schema_draft = ?6,
                 revision = revision + 1, updated_at = ?7
             WHERE id = ?8",
            params![
                t,
                status,
                stage,
                i64::from(requires_db),
                execution_plan,
                sql_schema_draft,
                now,
                run_id,
            ],
        )
        .map_err(|e| e.to_string())?;
    } else {
        tx.execute(
            "UPDATE team_runs
             SET status = ?1, stage = ?2, requires_db = ?3,
                 execution_plan = ?4, sql_schema_draft = ?5,
                 revision = revision + 1, updated_at = ?6
             WHERE id = ?7",
            params![
                status,
                stage,
                i64::from(requires_db),
                execution_plan,
                sql_schema_draft,
                now,
                run_id,
            ],
        )
        .map_err(|e| e.to_string())?;
    }
    if let Some(new_tasks) = tasks {
        tx.execute("DELETE FROM team_tasks WHERE run_id = ?1", params![run_id])
            .map_err(|e| e.to_string())?;
        for task in new_tasks {
            insert_task_row(&tx, task)?;
        }
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn reset_run_to_team_binding_conn(
    conn: &Connection,
    run_id: &str,
) -> Result<(), String> {
    transition_run_stage_conn(conn, run_id, "draft", "team_binding", None)
}

pub(crate) fn transition_run_stage_conn(
    conn: &Connection,
    run_id: &str,
    status: &str,
    stage: &str,
    requires_db: Option<bool>,
) -> Result<(), String> {
    let cur = load_run_conn(conn, run_id)?.ok_or_else(|| format!("未找到任务: {run_id}"))?;
    if cur.status == "running" {
        return Err("任务正在执行中，请先暂停后再切换阶段".to_string());
    }
    let now = now_ts();
    let req_db = requires_db.unwrap_or(cur.requires_db);
    conn.execute(
        "UPDATE team_runs
         SET status = ?1, stage = ?2, requires_db = ?3, revision = revision + 1, updated_at = ?4
         WHERE id = ?5",
        params![status, stage, i64::from(req_db), now, run_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// 阶段 2 → 阶段 3：用户确认数据库表结构定稿后，进入研发团队配置阶段，并顺手用 SQL 参考目录预填未绑定目录的后端智能体。
pub(crate) fn confirm_sql_to_team_binding_conn(
    conn: &Connection,
    run_id: &str,
    execution_plan: Option<&str>,
    sql_schema_draft: Option<&str>,
) -> Result<(), String> {
    let cur = load_run_conn(conn, run_id)?.ok_or_else(|| format!("未找到任务: {run_id}"))?;
    if cur.status == "running" {
        return Err("任务正在执行中，请先暂停后再操作".to_string());
    }
    update_run_design_texts_conn(conn, run_id, execution_plan, sql_schema_draft)?;
    transition_run_stage_conn(conn, run_id, "draft", "team_binding", Some(true))?;

    // 若 SQL 阶段绑定的目录确实包含应用源码（而非纯 SQL 脚本目录或空目录），且后端智能体目录为空，自动预填避免重复选择
    let candidate_sql_wd = if !clean_path_str(&cur.sql_working_dir).is_empty() {
        clean_path_str(&cur.sql_working_dir)
    } else if let Ok(presets) = load_role_presets_conn(conn) {
        presets
            .into_iter()
            .find(|p| {
                normalize_domain(&p.domain) == "sql" && !clean_path_str(&p.working_dir).is_empty()
            })
            .map(|p| clean_path_str(&p.working_dir))
            .unwrap_or_default()
    } else {
        String::new()
    };
    if !candidate_sql_wd.is_empty()
        && super::prompts::inspect_dir_profile(std::path::Path::new(&candidate_sql_wd)).kind
            == super::prompts::DirProfileKind::Codebase
    {
        let _ = conn.execute(
            "UPDATE team_tasks
             SET working_dir = ?1, updated_at = ?2
             WHERE run_id = ?3 AND domain = 'backend' AND TRIM(working_dir) = ''",
            params![candidate_sql_wd, now_ts(), run_id],
        );
    }
    Ok(())
}

pub(crate) fn update_run_sql_binding_conn(
    conn: &Connection,
    run_id: &str,
    sql_working_dir: Option<&str>,
    sql_agent_id: Option<&str>,
    sql_model: Option<&str>,
    sql_session_id: Option<&str>,
) -> Result<(), String> {
    let now = now_ts();
    let wd_val = sql_working_dir.map(clean_path_str);
    let aid_val = sql_agent_id.map(|s| {
        let t = s.trim();
        if t.is_empty() {
            "none".to_string()
        } else {
            t.to_string()
        }
    });
    let mdl_val = sql_model.map(|s| s.trim().to_string());
    let sid_val = sql_session_id.map(|s| s.trim().to_string());
    let dir_changed = if let Some(wd) = &wd_val {
        let cur = load_run_conn(conn, run_id)?.ok_or_else(|| format!("未找到任务: {run_id}"))?;
        !binding_dirs_equal(wd, &cur.sql_working_dir)
    } else {
        false
    };
    conn.execute(
        "UPDATE team_runs
         SET sql_working_dir = COALESCE(?1, sql_working_dir),
             sql_agent_id = COALESCE(?2, sql_agent_id),
             sql_model = COALESCE(?3, sql_model),
             sql_session_id = CASE
                 WHEN ?1 IS NOT NULL AND ?7 THEN ''
                 ELSE COALESCE(?4, sql_session_id) END,
             updated_at = ?5
         WHERE id = ?6",
        params![wd_val, aid_val, mdl_val, sid_val, now, run_id, dir_changed],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn update_run_design_texts_conn(
    conn: &Connection,
    run_id: &str,
    execution_plan: Option<&str>,
    sql_schema_draft: Option<&str>,
) -> Result<(), String> {
    let now = now_ts();
    if let Some(plan) = execution_plan {
        conn.execute(
            "UPDATE team_runs SET execution_plan = ?1, updated_at = ?2 WHERE id = ?3",
            params![plan, now, run_id],
        )
        .map_err(|e| e.to_string())?;
    }
    if let Some(sql) = sql_schema_draft {
        conn.execute(
            "UPDATE team_runs SET sql_schema_draft = ?1, updated_at = ?2 WHERE id = ?3",
            params![sql, now, run_id],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 在全流程任意阶段（包括运行中途、暂停、已完成）向任务单安全追加新成员与可选接口契约补充，绝不重置或丢失已有成员状态。
pub(crate) fn append_tasks_to_run_conn(
    conn: &mut Connection,
    run_id: &str,
    new_tasks: &[TeamTask],
    interface_spec_addition: Option<&str>,
    replace_interface_spec: Option<&str>,
) -> Result<u64, String> {
    if new_tasks.is_empty()
        && interface_spec_addition
            .map(|s| s.trim().is_empty())
            .unwrap_or(true)
        && replace_interface_spec.is_none()
    {
        return Err("没有需要插入的新成员或契约内容".to_string());
    }
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let cur = load_run_conn(&tx, run_id)?.ok_or_else(|| format!("未找到任务: {run_id}"))?;
    if cur.status == "running" && matches!(cur.stage.as_str(), "planning" | "sql_designing") {
        return Err("当前正处于主控拆单/SQL设计阶段，请等待该阶段完成后再插入成员".to_string());
    }

    let now = now_ts();
    let existing = load_tasks_for_run_conn(&tx, run_id)?;
    let existing_ids: std::collections::HashSet<String> =
        existing.iter().map(|t| t.id.clone()).collect();
    let mut combined = existing.clone();
    let mut added_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for raw in new_tasks {
        let mut t = raw.clone();
        if t.id.trim().is_empty() || existing_ids.contains(&t.id) || added_ids.contains(&t.id) {
            t.id = new_team_id("tsk");
        }
        added_ids.insert(t.id.clone());
        t.run_id = run_id.to_string();
        t.domain = normalize_domain(&t.domain);
        t.working_dir = clean_path_str(&t.working_dir);
        t.images.truncate(4);
        t.revision_note.clear();
        t.status = "pending".to_string();
        t.attempt = 0;
        t.attempt_id.clear();
        t.repair_count = 0;
        t.session_id.clear();
        t.contract_summary.clear();
        t.last_output_excerpt.clear();
        t.verify_verdict.clear();
        t.verify_feedback.clear();
        t.touched_files.clear();
        t.started_at.clear();
        t.finished_at.clear();
        t.updated_at = now.clone();
        combined.push(t);
    }

    let _ = compute_and_assign_waves(&mut combined)?;

    // 仅更新既有任务的 wave 与 sort_order，保留全部运行态字段与已完成结果
    for task in &combined {
        if existing_ids.contains(&task.id) {
            tx.execute(
                "UPDATE team_tasks SET wave = ?1, sort_order = ?2 WHERE id = ?3",
                params![task.wave as i64, task.sort_order as i64, task.id],
            )
            .map_err(|e| e.to_string())?;
        } else {
            insert_task_row(&tx, task)?;
        }
    }

    let next_spec = if let Some(full) = replace_interface_spec {
        full.to_string()
    } else if let Some(add) = interface_spec_addition
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if cur.interface_spec.trim().is_empty() {
            add.to_string()
        } else {
            format!("{}\n\n{}", cur.interface_spec.trim_end(), add)
        }
    } else {
        cur.interface_spec.clone()
    };

    let (next_status, next_stage) =
        if !new_tasks.is_empty() && matches!(cur.status.as_str(), "completed" | "cancelled") {
            (
                "draft",
                if cur.mode == "triage" {
                    "triage"
                } else {
                    "team_binding"
                },
            )
        } else {
            (cur.status.as_str(), cur.stage.as_str())
        };

    let next_rev = cur.revision.saturating_add(1);
    tx.execute(
        "UPDATE team_runs
         SET status = ?1, stage = ?2, interface_spec = ?3, revision = ?4, updated_at = ?5
         WHERE id = ?6",
        params![
            next_status,
            next_stage,
            next_spec,
            next_rev as i64,
            now,
            run_id
        ],
    )
    .map_err(|e| e.to_string())?;
    if next_spec != cur.interface_spec {
        let _ = tx.execute(
            "UPDATE team_contracts SET schema_or_api = ?1 WHERE run_id = ?2 AND from_task_id = 'lead_contract'",
            params![truncate_chars_safe(&next_spec, MAX_CONTRACT_CHARS), run_id],
        );
    }

    if !new_tasks.is_empty() {
        let names: Vec<String> = combined
            .iter()
            .filter(|t| added_ids.contains(&t.id))
            .map(|t| {
                if t.project_name.trim().is_empty() {
                    format!("{} (Wave {})", t.title, t.wave)
                } else {
                    format!("{} · {} (Wave {})", t.project_name, t.title, t.wave)
                }
            })
            .collect();
        let _ = append_event_conn(
            &tx,
            run_id,
            "task_inserted",
            "",
            "user",
            &format!("插入 {} 位协同成员：{}", added_ids.len(), names.join("、")),
            interface_spec_addition.unwrap_or(""),
        );
    }

    tx.commit().map_err(|e| e.to_string())?;
    Ok(next_rev)
}

#[derive(Debug, Clone)]
pub(crate) struct RevisionTaskTarget {
    pub(crate) task_id: String,
    pub(crate) note: String,
    pub(crate) images: Vec<TeamImage>,
    pub(crate) replace_images: bool,
    pub(crate) extra_depends_on: Vec<String>,
}

/// 安排单成员定向改造或跨成员联动变更：
/// - 仅重置本次受影响成员的状态为 `pending` 并写入 `revision_note`（与可选附图、追加前置依赖）；
/// - 其余未勾选成员的 `completed` 状态与既有成果完整保留；
/// - 校验 DAG 无环并自动重算波次，确保下游联动成员等上游改完再开工。
pub(crate) fn schedule_revision_conn(
    conn: &mut Connection,
    run_id: &str,
    targets: &[RevisionTaskTarget],
    interface_spec_addition: Option<&str>,
    replace_interface_spec: Option<&str>,
    auto_start: bool,
) -> Result<u64, String> {
    if targets.is_empty() {
        return Err("请至少选择一位需要修改或联动的成员".to_string());
    }
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let cur = load_run_conn(&tx, run_id)?.ok_or_else(|| format!("未找到任务: {run_id}"))?;
    if cur.status == "running" && matches!(cur.stage.as_str(), "planning" | "sql_designing") {
        return Err("当前正处于主控拆单/SQL设计阶段，请稍候再发起变更".to_string());
    }

    let mut all_tasks = load_tasks_for_run_conn(&tx, run_id)?;
    let valid_ids: std::collections::HashSet<String> =
        all_tasks.iter().map(|t| t.id.clone()).collect();
    let now = now_ts();
    let mut updated_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut target_titles: Vec<String> = Vec::new();
    let mut summary_details: Vec<String> = Vec::new();

    for spec in targets {
        let tid = spec.task_id.trim();
        let note = spec.note.trim();
        if tid.is_empty() || note.is_empty() {
            return Err("每个需要改造的成员都必须指定 taskId 与修改说明".to_string());
        }
        let task = all_tasks
            .iter_mut()
            .find(|t| t.id == tid)
            .ok_or_else(|| format!("未找到子任务: {tid}"))?;
        if matches!(task.status.as_str(), "running" | "verifying" | "repairing") {
            return Err(format!(
                "成员「{}」正在执行中，请等待其完成或暂停后再发起定向改造",
                task.title
            ));
        }
        if !updated_ids.insert(task.id.clone()) {
            return Err(format!("成员「{}」重复出现在改造列表中", task.title));
        }
        task.status = "pending".to_string();
        task.attempt_id.clear();
        task.repair_count = 0;
        task.verify_verdict.clear();
        task.verify_feedback.clear();
        task.revision_note = note.to_string();
        task.updated_at = now.clone();
        if spec.replace_images {
            task.images = spec.images.iter().take(4).cloned().collect();
        } else if !spec.images.is_empty() {
            for im in &spec.images {
                if task.images.len() >= 4 {
                    break;
                }
                task.images.push(im.clone());
            }
        }
        for dep in &spec.extra_depends_on {
            let dep_id = dep.trim();
            if !dep_id.is_empty()
                && dep_id != task.id
                && valid_ids.contains(dep_id)
                && !task.depends_on.iter().any(|d| d == dep_id)
            {
                task.depends_on.push(dep_id.to_string());
            }
        }
        let label = if task.project_name.trim().is_empty() {
            task.title.clone()
        } else {
            format!("{} · {}", task.project_name, task.title)
        };
        target_titles.push(label.clone());
        summary_details.push(format!("- {label}: {}", truncate_chars_safe(note, 260)));
    }

    let _ = compute_and_assign_waves(&mut all_tasks)?;

    for task in &all_tasks {
        if updated_ids.contains(&task.id) {
            let deps_json = serde_json::to_string(&task.depends_on).unwrap_or_else(|_| "[]".into());
            let images_json = serde_json::to_string(&task.images).unwrap_or_else(|_| "[]".into());
            tx.execute(
                "UPDATE team_tasks
                 SET depends_on_json = ?1, wave = ?2, sort_order = ?3,
                     status = 'pending', attempt_id = '', repair_count = 0,
                     verify_verdict = '', verify_feedback = '',
                     revision_note = ?4, images_json = ?5, updated_at = ?6
                 WHERE id = ?7",
                params![
                    deps_json,
                    task.wave as i64,
                    task.sort_order as i64,
                    task.revision_note,
                    images_json,
                    now,
                    task.id,
                ],
            )
            .map_err(|e| e.to_string())?;
        } else {
            tx.execute(
                "UPDATE team_tasks SET wave = ?1, sort_order = ?2 WHERE id = ?3",
                params![task.wave as i64, task.sort_order as i64, task.id],
            )
            .map_err(|e| e.to_string())?;
        }
    }

    let next_spec = if let Some(full) = replace_interface_spec {
        full.to_string()
    } else if let Some(add) = interface_spec_addition
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        if cur.interface_spec.trim().is_empty() {
            add.to_string()
        } else {
            format!("{}\n\n{}", cur.interface_spec.trim_end(), add)
        }
    } else {
        cur.interface_spec.clone()
    };

    let next_status = if auto_start { "running" } else { "draft" };
    let next_stage = if cur.mode == "triage" {
        "triage"
    } else if auto_start {
        "executing"
    } else {
        "team_binding"
    };
    let next_rev = cur.revision.saturating_add(1);

    tx.execute(
        "UPDATE team_runs
         SET status = ?1, stage = ?2, interface_spec = ?3, revision = ?4, updated_at = ?5
         WHERE id = ?6",
        params![
            next_status,
            next_stage,
            next_spec,
            next_rev as i64,
            now,
            run_id
        ],
    )
    .map_err(|e| e.to_string())?;

    if next_spec != cur.interface_spec {
        let _ = tx.execute(
            "UPDATE team_contracts SET schema_or_api = ?1 WHERE run_id = ?2 AND from_task_id = 'lead_contract'",
            params![truncate_chars_safe(&next_spec, MAX_CONTRACT_CHARS), run_id],
        );
    }

    let _ = append_event_conn(
        &tx,
        run_id,
        "revision_scheduled",
        if targets.len() == 1 {
            targets[0].task_id.as_str()
        } else {
            ""
        },
        "user",
        &format!(
            "发起{}（涉及 {} 位：{}）",
            if targets.len() > 1 {
                "跨成员联动改造"
            } else {
                "单成员定向改造"
            },
            targets.len(),
            target_titles.join(" → ")
        ),
        &summary_details.join("\n"),
    );

    tx.commit().map_err(|e| e.to_string())?;
    Ok(next_rev)
}

pub(crate) fn update_run_stage_conn(
    conn: &Connection,
    run_id: &str,
    status: Option<&str>,
    stage: &str,
) -> Result<(), String> {
    let now = now_ts();
    if let Some(st) = status {
        conn.execute(
            "UPDATE team_runs
             SET status = ?1, stage = ?2, revision = revision + 1, updated_at = ?3
             WHERE id = ?4",
            params![st, stage, now, run_id],
        )
        .map_err(|e| e.to_string())?;
    } else {
        conn.execute(
            "UPDATE team_runs
             SET stage = ?1, revision = revision + 1, updated_at = ?2
             WHERE id = ?3",
            params![stage, now, run_id],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn update_run_interface_spec_conn(
    conn: &Connection,
    run_id: &str,
    stage: &str,
    interface_spec: &str,
) -> Result<(), String> {
    let now = now_ts();
    conn.execute(
        "UPDATE team_runs
         SET stage = ?1, interface_spec = ?2, revision = revision + 1, updated_at = ?3
         WHERE id = ?4",
        params![stage, interface_spec, now, run_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn update_run_delivery_conn(
    conn: &Connection,
    run_id: &str,
    status: &str,
    stage: &str,
    delivery_sql: &str,
    synthesis_report: &str,
) -> Result<(), String> {
    let now = now_ts();
    let finished = if matches!(status, "completed" | "failed" | "cancelled") {
        now.as_str()
    } else {
        ""
    };
    conn.execute(
        "UPDATE team_runs
         SET status = ?1, stage = ?2, delivery_sql = ?3, synthesis_report = ?4,
             revision = revision + 1, updated_at = ?5,
             finished_at = CASE WHEN ?6 != '' THEN ?6 ELSE finished_at END
         WHERE id = ?7",
        params![
            status,
            stage,
            delivery_sql,
            synthesis_report,
            now,
            finished,
            run_id
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// 认领子任务并生成全新 `attempt_id` 栅栏令牌。
pub(crate) fn claim_task_attempt_conn(
    conn: &Connection,
    task_id: &str,
    session_id: &str,
    project_hash: &str,
    status: &str,
) -> Result<(u32, String), String> {
    let cur = load_task_conn(conn, task_id)?.ok_or_else(|| format!("Task not found: {task_id}"))?;
    let next_attempt = cur.attempt.saturating_add(1);
    let attempt_id = new_team_id(&format!("att{next_attempt}"));
    let now = now_ts();
    let started = if cur.started_at.is_empty() {
        now.clone()
    } else {
        cur.started_at
    };
    conn.execute(
        "UPDATE team_tasks
         SET status = ?1, session_id = ?2, project_hash = ?3,
             attempt = ?4, attempt_id = ?5, started_at = ?6, updated_at = ?7
         WHERE id = ?8",
        params![
            status,
            session_id,
            project_hash,
            next_attempt as i64,
            attempt_id,
            started,
            now,
            task_id
        ],
    )
    .map_err(|e| e.to_string())?;
    conn.execute(
        "UPDATE team_runs SET revision = revision + 1, updated_at = ?1 WHERE id = ?2",
        params![now, cur.run_id],
    )
    .ok();
    Ok((next_attempt, attempt_id))
}

/// 带 `attempt_id` 栅栏校验的子任务状态写回：若后台已被取消、重试或新 attempt 接管，旧结果自动丢弃不污染。
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_task_fenced_conn(
    conn: &Connection,
    task_id: &str,
    expected_attempt_id: &str,
    status: &str,
    excerpt: &str,
    contract_summary: Option<&str>,
    verify_verdict: &str,
    verify_feedback: &str,
    repair_count: u32,
    touched_files: &[String],
) -> Result<bool, String> {
    let now = now_ts();
    let finished = if matches!(status, "completed" | "failed" | "cancelled") {
        now.as_str()
    } else {
        ""
    };
    let clean_excerpt = truncate_chars_safe(excerpt, MAX_EXCERPT_CHARS);
    let files_json = serde_json::to_string(touched_files).unwrap_or_else(|_| "[]".into());
    let (next_desc, clear_revision) = if status == "completed" {
        if let Ok(Some(cur)) = load_task_conn(conn, task_id) {
            if !cur.revision_note.trim().is_empty() {
                let archive = format!(
                    "[已完成改造 {now}] {}",
                    truncate_chars_safe(&cur.revision_note, 400)
                );
                let combined = if cur.description.trim().is_empty() {
                    archive
                } else {
                    format!("{}\n{archive}", cur.description.trim_end())
                };
                (Some(truncate_chars_safe(&combined, 6000)), true)
            } else {
                (None, true)
            }
        } else {
            (None, false)
        }
    } else {
        (None, false)
    };
    let changed = if let Some(contract) = contract_summary {
        let clean_contract = truncate_chars_safe(contract, MAX_CONTRACT_CHARS);
        conn.execute(
            "UPDATE team_tasks
             SET status = ?1, last_output_excerpt = ?2, contract_summary = ?3,
                 verify_verdict = ?4, verify_feedback = ?5, repair_count = ?6,
                 touched_files_json = ?7, updated_at = ?8,
                 finished_at = CASE WHEN ?9 != '' THEN ?9 ELSE finished_at END,
                 description = COALESCE(?12, description),
                 revision_note = CASE WHEN ?13 THEN '' ELSE revision_note END
             WHERE id = ?10 AND attempt_id = ?11",
            params![
                status,
                clean_excerpt,
                clean_contract,
                verify_verdict,
                verify_feedback,
                repair_count as i64,
                files_json,
                now,
                finished,
                task_id,
                expected_attempt_id,
                next_desc,
                clear_revision,
            ],
        )
        .map_err(|e| e.to_string())?
    } else {
        conn.execute(
            "UPDATE team_tasks
             SET status = ?1, last_output_excerpt = ?2,
                 verify_verdict = ?3, verify_feedback = ?4, repair_count = ?5,
                 touched_files_json = ?6, updated_at = ?7,
                 finished_at = CASE WHEN ?8 != '' THEN ?8 ELSE finished_at END,
                 description = COALESCE(?11, description),
                 revision_note = CASE WHEN ?12 THEN '' ELSE revision_note END
             WHERE id = ?9 AND attempt_id = ?10",
            params![
                status,
                clean_excerpt,
                verify_verdict,
                verify_feedback,
                repair_count as i64,
                files_json,
                now,
                finished,
                task_id,
                expected_attempt_id,
                next_desc,
                clear_revision,
            ],
        )
        .map_err(|e| e.to_string())?
    };

    if changed > 0 {
        if let Ok(Some(t)) = load_task_conn(conn, task_id) {
            let _ = conn.execute(
                "UPDATE team_runs SET revision = revision + 1, updated_at = ?1 WHERE id = ?2",
                params![now, t.run_id],
            );
        }
    }
    Ok(changed > 0)
}

pub(crate) fn set_task_status_fenced_conn(
    conn: &Connection,
    task_id: &str,
    expected_attempt_id: &str,
    status: &str,
) -> Result<bool, String> {
    let now = now_ts();
    let changed = conn
        .execute(
            "UPDATE team_tasks SET status = ?1, updated_at = ?2 WHERE id = ?3 AND attempt_id = ?4",
            params![status, now, task_id, expected_attempt_id],
        )
        .map_err(|e| e.to_string())?;
    Ok(changed > 0)
}

pub(crate) fn upsert_contract_conn(
    conn: &Connection,
    run_id: &str,
    from_task_id: &str,
    domain: &str,
    title: &str,
    schema_or_api: &str,
    notes: &str,
) -> Result<TeamContract, String> {
    let now = now_ts();
    let clean_schema = truncate_chars_safe(schema_or_api, MAX_CONTRACT_CHARS);
    let clean_notes = truncate_chars_safe(notes, 1200);
    let existing_id: Option<String> = conn
        .query_row(
            "SELECT id FROM team_contracts WHERE run_id = ?1 AND from_task_id = ?2",
            params![run_id, from_task_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let id = existing_id.unwrap_or_else(|| new_team_id("ctr"));
    let norm_domain = normalize_domain(domain);

    conn.execute(
        "INSERT INTO team_contracts (id, run_id, from_task_id, domain, title, schema_or_api, notes, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(id) DO UPDATE SET
             domain = excluded.domain,
             title = excluded.title,
             schema_or_api = excluded.schema_or_api,
             notes = excluded.notes,
             created_at = excluded.created_at",
        params![
            id,
            run_id,
            from_task_id,
            norm_domain,
            title.trim(),
            clean_schema,
            clean_notes,
            now
        ],
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "UPDATE team_tasks SET contract_summary = ?1, updated_at = ?2 WHERE id = ?3",
        params![
            format!("【{}】\n{}", title.trim(), clean_schema),
            now,
            from_task_id
        ],
    )
    .ok();
    conn.execute(
        "UPDATE team_runs SET revision = revision + 1, updated_at = ?1 WHERE id = ?2",
        params![now, run_id],
    )
    .ok();

    Ok(TeamContract {
        id,
        run_id: run_id.to_string(),
        from_task_id: from_task_id.to_string(),
        domain: norm_domain,
        title: title.trim().to_string(),
        schema_or_api: clean_schema,
        notes: clean_notes,
        created_at: now,
    })
}

pub(crate) fn append_event_conn(
    conn: &Connection,
    run_id: &str,
    kind: &str,
    task_id: &str,
    actor: &str,
    summary: &str,
    detail: &str,
) -> Result<i64, String> {
    let now = now_ts();
    let clean_detail = truncate_chars_safe(detail, 2400);
    conn.execute(
        "INSERT INTO team_events (run_id, ts, kind, task_id, actor, summary, detail)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            run_id,
            now,
            kind,
            task_id,
            actor,
            summary.trim(),
            clean_detail
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(conn.last_insert_rowid())
}

pub(crate) fn delete_run_conn(conn: &mut Connection, run_id: &str) -> Result<(), String> {
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute("DELETE FROM team_events WHERE run_id = ?1", params![run_id])
        .map_err(|e| e.to_string())?;
    tx.execute(
        "DELETE FROM team_contracts WHERE run_id = ?1",
        params![run_id],
    )
    .map_err(|e| e.to_string())?;
    tx.execute("DELETE FROM team_tasks WHERE run_id = ?1", params![run_id])
        .map_err(|e| e.to_string())?;
    tx.execute("DELETE FROM team_runs WHERE id = ?1", params![run_id])
        .map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}
