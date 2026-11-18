//! `zap-agent-teams` SQLite WAL 表结构定义与版本迁移。

use std::path::Path;

use rusqlite::Connection;

pub(crate) const CURRENT_SCHEMA_VERSION: i64 = 1;

pub(crate) fn open_connection(data_dir: &Path) -> Result<Connection, String> {
    std::fs::create_dir_all(data_dir).map_err(|e| {
        format!(
            "Failed to create agent-teams dir {}: {e}",
            data_dir.display()
        )
    })?;
    let db_path = data_dir.join("teams.db");
    let conn = Connection::open(&db_path)
        .map_err(|e| format!("Failed to open teams.db at {}: {e}", db_path.display()))?;
    init_schema(&conn)?;
    Ok(conn)
}

pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
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
    .map_err(|e| format!("Failed to initialize agent_teams schema: {e}"))?;

    let ver: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap_or(0);
    if ver < CURRENT_SCHEMA_VERSION {
        conn.execute_batch(&format!("PRAGMA user_version = {CURRENT_SCHEMA_VERSION};"))
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}
