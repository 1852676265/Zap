//! SQLite Schema 初始化、版本管理（PRAGMA user_version）与平滑迁移。

use std::path::Path;

use rusqlite::Connection;

use super::types::normalize_project_key_str;

pub(crate) const CURRENT_SCHEMA_VERSION: i64 = 2;

fn table_has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, String> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&sql).map_err(|e| e.to_string())?;
    let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
    while let Some(row) = rows.next().map_err(|e| e.to_string())? {
        let name: String = row.get(1).map_err(|e| e.to_string())?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    col_def: &str,
) -> Result<(), String> {
    if !table_has_column(conn, table, column)? {
        let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {col_def}");
        conn.execute(&sql, []).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 初始化并迁移 SQLite 表结构（同时供生产连接与单元测试内存库调用）。
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

        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            icon TEXT NOT NULL,
            name TEXT NOT NULL,
            label TEXT NOT NULL,
            color TEXT NOT NULL,
            hint TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS session_agents (
            session_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            source_turn_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            working_dir TEXT NOT NULL,
            title TEXT NOT NULL,
            progress_note TEXT NOT NULL,
            remaining_note TEXT NOT NULL,
            touched_files_json TEXT NOT NULL,
            session_id TEXT NOT NULL,
            source_turn_ids_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            done_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            project TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            mastery INTEGER NOT NULL,
            tier INTEGER NOT NULL,
            windows_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            events_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS links (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            PRIMARY KEY (source, target)
        );

        CREATE TABLE IF NOT EXISTS timeline (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            working_dir TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            raw_file TEXT NOT NULL,
            user_goal TEXT NOT NULL,
            summary TEXT NOT NULL,
            ai_outcome TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            touched_files_json TEXT NOT NULL,
            distilled_by TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS raw_turns (
            turn_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            working_dir TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_text TEXT NOT NULL,
            ai_text TEXT NOT NULL,
            touched_files_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            node_id UNINDEXED,
            agent_id UNINDEXED,
            search_text
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS timeline_fts USING fts5(
            event_id UNINDEXED,
            agent_id UNINDEXED,
            search_text
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS raw_turns_fts USING fts5(
            turn_id UNINDEXED,
            agent_id UNINDEXED,
            search_text
        );
        ",
    )
    .map_err(|e| format!("Failed to initialize SQLite schema: {e}"))?;

    migrate_schema(conn)?;

    conn.execute_batch(
        "
        CREATE INDEX IF NOT EXISTS idx_profiles_agent ON profiles(agent_id, pinned DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_profiles_scope ON profiles(agent_id, project_key, pinned DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_status ON tasks(agent_id, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_scope ON tasks(agent_id, project_key, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_nodes_agent ON nodes(agent_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(agent_id, project_key, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_timeline_agent_ts ON timeline(agent_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_timeline_scope ON timeline(agent_id, project_key, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_raw_turns_agent_ts ON raw_turns(agent_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_raw_turns_scope ON raw_turns(agent_id, project_key, ts DESC);
        ",
    )
    .map_err(|e| format!("Failed to create SQLite indexes: {e}"))?;

    Ok(())
}

fn migrate_schema(conn: &Connection) -> Result<(), String> {
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap_or(0);

    if version < 2 {
        for table in ["profiles", "tasks", "nodes", "timeline", "raw_turns"] {
            ensure_column(conn, table, "project_key", "TEXT NOT NULL DEFAULT ''")?;
        }
        // 为历史带有 working_dir 的行回填规范化 project_key
        for table in ["tasks", "timeline", "raw_turns"] {
            let select_sql = format!(
                "SELECT rowid, working_dir FROM {table} WHERE project_key = '' AND working_dir != ''"
            );
            let mut stmt = conn.prepare(&select_sql).map_err(|e| e.to_string())?;
            let rows: Vec<(i64, String)> = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .map_err(|e| e.to_string())?
                .flatten()
                .collect();
            drop(stmt);
            let update_sql = format!("UPDATE {table} SET project_key = ?1 WHERE rowid = ?2");
            for (rowid, wd) in rows {
                let key = normalize_project_key_str(&wd);
                if !key.is_empty() {
                    conn.execute(&update_sql, rusqlite::params![key, rowid])
                        .ok();
                }
            }
        }
        conn.execute_batch(&format!("PRAGMA user_version = {CURRENT_SCHEMA_VERSION};"))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn open_connection(data_dir: &Path) -> Result<Connection, String> {
    std::fs::create_dir_all(data_dir)
        .map_err(|e| format!("Failed to create neural memory directory: {e}"))?;
    let db_path = data_dir.join("neural_memory.db");
    let conn = Connection::open(&db_path)
        .map_err(|e| format!("Failed to open neural memory SQLite: {e}"))?;
    init_schema(&conn)?;
    Ok(conn)
}
