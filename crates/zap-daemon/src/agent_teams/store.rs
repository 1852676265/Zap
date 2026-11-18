//! `zap-agent-teams` SQLite WAL 持久化存储、CAS 并发门禁、attempt_id 栅栏与进程重启孤儿自愈。

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use rusqlite::{params, Connection, OptionalExtension};

#[cfg(test)]
use super::schema::init_schema;
use super::schema::open_connection;
use super::types::{
    new_team_id, normalize_domain, now_ts, truncate_chars_safe, RoleBindingPreset, TeamContract,
    TeamEvent, TeamGlobalConfig, TeamRun, TeamRunDetail, TeamTask, HARD_CAP_MAX_REPAIRS,
    MAX_CONTRACT_CHARS, MAX_EXCERPT_CHARS,
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

/// 进程启动孤儿自愈：上次宿主退出时若仍有 `running` 态编排单，平滑收敛为 `paused`，允许用户一键续跑。
fn reconcile_interrupted_runs_on_boot(conn: &Connection) {
    let now = now_ts();
    let mut stmt = match conn.prepare("SELECT id FROM team_runs WHERE status = 'running'") {
        Ok(s) => s,
        Err(_) => return,
    };
    let ids: Vec<String> = match stmt.query_map([], |r| r.get(0)) {
        Ok(rows) => rows.flatten().collect(),
        Err(_) => Vec::new(),
    };
    for run_id in ids {
        let _ = conn.execute(
            "UPDATE team_runs SET status = 'paused', revision = revision + 1, updated_at = ?1 WHERE id = ?2",
            params![now, run_id],
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
            "检测到宿主重启，已自动保护现场并暂停流水线，点击「继续执行」即可从断点无缝恢复。",
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
            Ok(RoleBindingPreset {
                domain: r.get(0)?,
                working_dir: r.get(1)?,
                agent_id: r.get(2)?,
                model: r.get(3)?,
                updated_at: r.get(4)?,
            })
        })
        .map_err(|e| e.to_string())?;
    Ok(rows.flatten().collect())
}

pub(crate) fn upsert_role_preset_conn(
    conn: &Connection,
    preset: &RoleBindingPreset,
) -> Result<(), String> {
    let domain = normalize_domain(&preset.domain);
    let now = now_ts();
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
            preset.working_dir.trim(),
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

// ---------------------------------------------------------------------------
// Runs, Tasks, Contracts & Events CRUD + CAS / Fencing Primitives
// ---------------------------------------------------------------------------

fn row_to_run(r: &rusqlite::Row<'_>) -> rusqlite::Result<TeamRun> {
    let rev: i64 = r.get(5)?;
    let auto_verify: i64 = r.get(6)?;
    let max_repairs: i64 = r.get(7)?;
    Ok(TeamRun {
        id: r.get(0)?,
        title: r.get(1)?,
        user_goal: r.get(2)?,
        mode: r.get(3)?,
        status: r.get(4)?,
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
        working_dir: r.get(9)?,
        project_hash: r.get(10)?,
        agent_id: r.get(11)?,
        agent_label: r.get(12)?,
        model: r.get(13)?,
        session_id: r.get(14)?,
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
                    lead_model, source_triage_id, synthesis_report, created_at, updated_at, finished_at
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
                lead_model, source_triage_id, synthesis_report, created_at, updated_at, finished_at
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
                    verify_feedback, touched_files_json, status, started_at, updated_at, finished_at
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
                verify_feedback, touched_files_json, status, started_at, updated_at, finished_at
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
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    tx.execute(
        "INSERT INTO team_runs (
            id, title, user_goal, mode, status, revision, auto_verify, max_repairs,
            lead_model, source_triage_id, synthesis_report, created_at, updated_at, finished_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
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
    conn.execute(
        "INSERT INTO team_tasks (
            id, run_id, domain, title, description, acceptance_criteria,
            depends_on_json, wave, sort_order, working_dir, project_hash,
            agent_id, agent_label, model, session_id, status, attempt, attempt_id,
            repair_count, contract_summary, last_output_excerpt, verify_verdict,
            verify_feedback, touched_files_json, started_at, updated_at, finished_at
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6,
            ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16, ?17, ?18,
            ?19, ?20, ?21, ?22,
            ?23, ?24, ?25, ?26, ?27
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
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// CAS 更新派工单草稿（校验 `expected_revision`，仅允许在非 `running` 态下整单编辑任务拓扑）。
pub(crate) fn update_run_draft_cas_conn(
    conn: &mut Connection,
    run_id: &str,
    expected_revision: Option<u64>,
    title: &str,
    auto_verify: bool,
    max_repairs: u32,
    lead_model: &str,
    tasks: &[TeamTask],
) -> Result<u64, String> {
    let cur = load_run_conn(conn, run_id)?.ok_or_else(|| format!("未找到派工单: {run_id}"))?;
    if cur.status == "running" {
        return Err("派工单正在执行中，请先暂停后再调整任务结构".to_string());
    }
    if let Some(exp) = expected_revision {
        if exp != cur.revision {
            return Err(format!(
                "派工单版本已变更（本地 v{exp} != 最新 v{}），已刷新最新状态",
                cur.revision
            ));
        }
    }

    let next_rev = cur.revision.saturating_add(1);
    let now = now_ts();
    let tx = conn.transaction().map_err(|e| e.to_string())?;
    let changed = tx
        .execute(
            "UPDATE team_runs
             SET title = ?1, auto_verify = ?2, max_repairs = ?3, lead_model = ?4,
                 revision = ?5, updated_at = ?6
             WHERE id = ?7 AND revision = ?8",
            params![
                title.trim(),
                i64::from(auto_verify),
                max_repairs.min(HARD_CAP_MAX_REPAIRS) as i64,
                lead_model.trim(),
                next_rev as i64,
                now,
                run_id,
                cur.revision as i64,
            ],
        )
        .map_err(|e| e.to_string())?;
    if changed == 0 {
        return Err("并发保存冲突（CAS revision mismatch），请重试".to_string());
    }

    tx.execute("DELETE FROM team_tasks WHERE run_id = ?1", params![run_id])
        .map_err(|e| e.to_string())?;
    for task in tasks {
        insert_task_row(&tx, task)?;
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
    let changed = if let Some(contract) = contract_summary {
        let clean_contract = truncate_chars_safe(contract, MAX_CONTRACT_CHARS);
        conn.execute(
            "UPDATE team_tasks
             SET status = ?1, last_output_excerpt = ?2, contract_summary = ?3,
                 verify_verdict = ?4, verify_feedback = ?5, repair_count = ?6,
                 touched_files_json = ?7, updated_at = ?8,
                 finished_at = CASE WHEN ?9 != '' THEN ?9 ELSE finished_at END
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
            ],
        )
        .map_err(|e| e.to_string())?
    } else {
        conn.execute(
            "UPDATE team_tasks
             SET status = ?1, last_output_excerpt = ?2,
                 verify_verdict = ?3, verify_feedback = ?4, repair_count = ?5,
                 touched_files_json = ?6, updated_at = ?7,
                 finished_at = CASE WHEN ?8 != '' THEN ?8 ELSE finished_at END
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
