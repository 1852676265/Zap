//! `zap-agent-teams` SQLite WAL 表结构定义与版本迁移。

use std::path::Path;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

pub(crate) const CURRENT_SCHEMA_VERSION: i64 = 5;

fn format_sqlite_err(err: &rusqlite::Error) -> String {
    if let rusqlite::Error::SqliteFailure(ffi_err, msg) = err {
        let detail = msg.as_deref().unwrap_or("");
        if detail.is_empty() {
            format!(
                "{err} (code={:?}, ext={})",
                ffi_err.code, ffi_err.extended_code
            )
        } else {
            format!(
                "{err} (code={:?}, ext={}, detail={detail})",
                ffi_err.code, ffi_err.extended_code
            )
        }
    } else {
        err.to_string()
    }
}

pub(crate) fn open_connection(data_dir: &Path) -> Result<Connection, String> {
    std::fs::create_dir_all(data_dir).map_err(|e| {
        format!(
            "Failed to create agent-teams dir {}: {e}",
            data_dir.display()
        )
    })?;
    let db_path = data_dir.join("teams.db");
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;

    let mut last_conn = None;
    for attempt in 0..5 {
        let conn = Connection::open_with_flags(&db_path, flags).map_err(|e| {
            format!(
                "Failed to open teams.db at {}: {}",
                db_path.display(),
                format_sqlite_err(&e)
            )
        })?;
        let _ = conn.busy_timeout(Duration::from_millis(5000));
        if !conn
            .is_readonly(rusqlite::DatabaseName::Main)
            .unwrap_or(false)
        {
            last_conn = Some(conn);
            break;
        }
        if attempt < 4 {
            std::thread::sleep(Duration::from_millis(80 * (attempt + 1)));
        } else {
            let os_probe = match std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&db_path)
            {
                Ok(_) => {
                    "os_rw_open=ok (sqlite opened readonly due to concurrent lock/shm)".to_string()
                }
                Err(e) => format!("os_rw_open_err={e}"),
            };
            return Err(format!(
                "teams.db at {} opened in read-only mode ({os_probe}; please ensure no stale zap.exe holds an exclusive lock)",
                db_path.display()
            ));
        }
    }
    let conn = last_conn.expect("connection opened");
    init_schema(&conn).map_err(|e| format!("{e} [path={}]", db_path.display()))?;
    Ok(conn)
}

pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
    let _ = conn.busy_timeout(Duration::from_millis(5000));

    let cur_mode: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap_or_default();
    if !cur_mode.eq_ignore_ascii_case("wal") {
        // 多进程同时访问时若 WAL/shm 切换偶发只读/锁冲突，不阻断正常建表
        let _ = conn.execute_batch("PRAGMA journal_mode = WAL;");
    }

    conn.execute_batch(
        "
        PRAGMA synchronous = NORMAL;
        PRAGMA busy_timeout = 5000;
        PRAGMA foreign_keys = OFF;

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS role_presets (
            domain TEXT PRIMARY KEY,
            working_dir TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT 'none',
            model TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'general',
            working_dir TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT 'none',
            avatar_id TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            sort_order INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_runs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            user_goal TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'pipeline',
            status TEXT NOT NULL DEFAULT 'draft',
            revision INTEGER NOT NULL DEFAULT 1,
            auto_verify INTEGER NOT NULL DEFAULT 1,
            max_repairs INTEGER NOT NULL DEFAULT 2,
            lead_model TEXT NOT NULL DEFAULT '',
            source_triage_id TEXT NOT NULL DEFAULT '',
            synthesis_report TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS team_tasks (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'general',
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            acceptance_criteria TEXT NOT NULL DEFAULT '',
            depends_on_json TEXT NOT NULL DEFAULT '[]',
            wave INTEGER NOT NULL DEFAULT 1,
            sort_order INTEGER NOT NULL DEFAULT 0,
            working_dir TEXT NOT NULL DEFAULT '',
            project_hash TEXT NOT NULL DEFAULT '',
            agent_id TEXT NOT NULL DEFAULT 'none',
            agent_label TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            attempt INTEGER NOT NULL DEFAULT 0,
            attempt_id TEXT NOT NULL DEFAULT '',
            repair_count INTEGER NOT NULL DEFAULT 0,
            contract_summary TEXT NOT NULL DEFAULT '',
            last_output_excerpt TEXT NOT NULL DEFAULT '',
            verify_verdict TEXT NOT NULL DEFAULT '',
            verify_feedback TEXT NOT NULL DEFAULT '',
            touched_files_json TEXT NOT NULL DEFAULT '[]',
            started_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS team_contracts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            from_task_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            title TEXT NOT NULL,
            schema_or_api TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            task_id TEXT NOT NULL DEFAULT '',
            actor TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_team_runs_updated ON team_runs(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_team_tasks_run ON team_tasks(run_id, wave ASC, sort_order ASC);
        CREATE INDEX IF NOT EXISTS idx_team_contracts_run ON team_contracts(run_id, created_at ASC);
        CREATE INDEX IF NOT EXISTS idx_team_events_run_seq ON team_events(run_id, seq ASC);
        ",
    )
    .map_err(|e| format!("Failed to initialize agent_teams schema: {}", format_sqlite_err(&e)))?;

    ensure_column(
        conn,
        "team_tasks",
        "project_slot_id",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_tasks",
        "project_name",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(conn, "team_tasks", "avatar_id", "TEXT NOT NULL DEFAULT ''")?;

    ensure_column(conn, "team_runs", "stage", "TEXT NOT NULL DEFAULT ''")?;
    ensure_column(
        conn,
        "team_runs",
        "requires_db",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "execution_plan",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "sql_schema_draft",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "interface_spec",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "delivery_sql",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "goal_images_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "sql_working_dir",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "sql_agent_id",
        "TEXT NOT NULL DEFAULT 'none'",
    )?;
    ensure_column(conn, "team_runs", "sql_model", "TEXT NOT NULL DEFAULT ''")?;
    ensure_column(
        conn,
        "team_runs",
        "sql_session_id",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_runs",
        "prompt_overrides_json",
        "TEXT NOT NULL DEFAULT '{}'",
    )?;
    ensure_column(
        conn,
        "team_tasks",
        "custom_prompt",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "team_tasks",
        "images_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;
    ensure_column(
        conn,
        "team_tasks",
        "revision_note",
        "TEXT NOT NULL DEFAULT ''",
    )?;

    let ver: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap_or(0);
    if ver < CURRENT_SCHEMA_VERSION {
        conn.execute_batch(&format!("PRAGMA user_version = {CURRENT_SCHEMA_VERSION};"))
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn ensure_column(conn: &Connection, table: &str, col: &str, ddl: &str) -> Result<(), String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|e| e.to_string())?;
    let exists = stmt
        .query_map([], |r| r.get::<_, String>(1))
        .map_err(|e| e.to_string())?
        .flatten()
        .any(|name| name == col);
    if !exists {
        conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {col} {ddl}"), [])
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
