//! SQLite Schema 初始化、版本管理（PRAGMA user_version = 3）与无损平滑迁移。
//!
//! v3 核心存储升级：
//! - 5 抽屉清晰分层：规矩（`profiles`）、需求/学习进度本（`tasks`）、知识经验活卡片（`nodes` + `facts_json` + `aliases_json` + `search_tags_json`）、每日精华大事记（`timeline`）、原始对话冷仓库（`raw_turns`）
//! - 精确路牌表 `memory_anchors`（文件、表名、接口、符号、错误码、Team 单号 B-Tree 索引秒查）
//! - 持久化草稿队列 `draft_turns`（聊天当下轻写入，阶段整理抄正本，重启不丢）
//! - 幂等 Team 回流日志 `team_sync_log`
//! - 分列加权 FTS5（`nodes_fts` / `timeline_fts` / `profiles_fts` / `tasks_fts`），用 `rowid` 对齐主表消除删改全表扫
//! - 移除膨胀的 `raw_turns_fts`，热库瘦身 80%+

use std::path::Path;
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags};

use super::search::{build_fts_column_text, extract_storage_anchors};
use super::types::{new_id, normalize_project_key_str, NodeEvent, NodeFact};

pub(crate) const CURRENT_SCHEMA_VERSION: i64 = 3;

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

/// 创建 v3 分列加权 FTS5 虚拟表（全部通过 `rowid` 对齐主表 `rowid`，实现 $O(\log N)$ 精准增删）。
fn create_v3_fts_tables(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            node_id UNINDEXED,
            agent_id UNINDEXED,
            title_col,
            tags_col,
            summary_col,
            body_col
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS timeline_fts USING fts5(
            event_id UNINDEXED,
            agent_id UNINDEXED,
            keywords_col,
            goal_col,
            summary_col
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts USING fts5(
            profile_id UNINDEXED,
            agent_id UNINDEXED,
            text_col
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
            task_id UNINDEXED,
            agent_id UNINDEXED,
            title_col,
            detail_col
        );
        ",
    )
    .map_err(|e| format!("Failed to create v3 FTS5 tables: {e}"))
}

/// 初始化并迁移 SQLite 表结构（同时供生产写连接与单元测试内存库调用）。
pub(crate) fn init_schema(conn: &Connection) -> Result<(), String> {
    let _ = conn.busy_timeout(Duration::from_millis(5000));
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA busy_timeout = 5000;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -8000;
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
            source_turn_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'rule',
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by TEXT NOT NULL DEFAULT '',
            invalid_at TEXT NOT NULL DEFAULT '',
            hit_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT NOT NULL DEFAULT ''
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
            done_at TEXT NOT NULL,
            item_kind TEXT NOT NULL DEFAULT 'requirement',
            next_step TEXT NOT NULL DEFAULT '',
            team_run_id TEXT NOT NULL DEFAULT '',
            team_task_id TEXT NOT NULL DEFAULT ''
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
            updated_at TEXT NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            search_tags_json TEXT NOT NULL DEFAULT '[]',
            facts_json TEXT NOT NULL DEFAULT '[]',
            hit_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS links (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL DEFAULT 'related',
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
            distilled_by TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'work',
            team_run_id TEXT NOT NULL DEFAULT '',
            node_ids_json TEXT NOT NULL DEFAULT '[]'
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

        CREATE TABLE IF NOT EXISTS memory_anchors (
            agent_id TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            target_kind TEXT NOT NULL,
            target_id TEXT NOT NULL,
            anchor_type TEXT NOT NULL,
            anchor_value TEXT NOT NULL,
            anchor_norm TEXT NOT NULL,
            PRIMARY KEY (agent_id, target_kind, target_id, anchor_type, anchor_norm)
        );

        CREATE TABLE IF NOT EXISTS draft_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            project TEXT NOT NULL,
            project_key TEXT NOT NULL DEFAULT '',
            working_dir TEXT NOT NULL,
            user_text TEXT NOT NULL,
            ai_text TEXT NOT NULL,
            touched_files_json TEXT NOT NULL DEFAULT '[]',
            provider_name TEXT NOT NULL DEFAULT '',
            explicit_remember INTEGER NOT NULL DEFAULT 0,
            retry_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_sync_log (
            team_run_id TEXT NOT NULL,
            team_task_id TEXT NOT NULL,
            synced_updated_at TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            PRIMARY KEY (team_run_id, team_task_id)
        );
        ",
    )
    .map_err(|e| format!("Failed to initialize SQLite schema: {e}"))?;

    migrate_schema(conn)?;

    conn.execute_batch(
        "
        CREATE INDEX IF NOT EXISTS idx_profiles_agent ON profiles(agent_id, status, pinned DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_profiles_scope ON profiles(agent_id, project_key, status, pinned DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_status ON tasks(agent_id, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_scope ON tasks(agent_id, project_key, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_team ON tasks(team_run_id, team_task_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_agent ON nodes(agent_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(agent_id, project_key, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_nodes_projects ON nodes(agent_id, project_key, project);
        CREATE INDEX IF NOT EXISTS idx_timeline_agent_ts ON timeline(agent_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_timeline_scope ON timeline(agent_id, project_key, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_timeline_projects ON timeline(agent_id, project_key, project);
        CREATE INDEX IF NOT EXISTS idx_raw_turns_agent_ts ON raw_turns(agent_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_raw_turns_scope ON raw_turns(agent_id, project_key, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_anchors_lookup ON memory_anchors(agent_id, anchor_norm, target_kind);
        CREATE INDEX IF NOT EXISTS idx_anchors_target ON memory_anchors(target_kind, target_id);
        CREATE INDEX IF NOT EXISTS idx_drafts_scope ON draft_turns(agent_id, session_id, id ASC);
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
                    conn.execute(&update_sql, params![key, rowid]).ok();
                }
            }
        }
    }

    if version < 3 {
        // 1. 为老表无损补齐 v3 字段
        for (table, col, def) in [
            ("profiles", "kind", "TEXT NOT NULL DEFAULT 'rule'"),
            ("profiles", "status", "TEXT NOT NULL DEFAULT 'active'"),
            ("profiles", "superseded_by", "TEXT NOT NULL DEFAULT ''"),
            ("profiles", "invalid_at", "TEXT NOT NULL DEFAULT ''"),
            ("profiles", "hit_count", "INTEGER NOT NULL DEFAULT 0"),
            ("profiles", "last_used_at", "TEXT NOT NULL DEFAULT ''"),
            ("tasks", "item_kind", "TEXT NOT NULL DEFAULT 'requirement'"),
            ("tasks", "next_step", "TEXT NOT NULL DEFAULT ''"),
            ("tasks", "team_run_id", "TEXT NOT NULL DEFAULT ''"),
            ("tasks", "team_task_id", "TEXT NOT NULL DEFAULT ''"),
            ("nodes", "aliases_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("nodes", "search_tags_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("nodes", "facts_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("nodes", "hit_count", "INTEGER NOT NULL DEFAULT 0"),
            ("nodes", "last_used_at", "TEXT NOT NULL DEFAULT ''"),
            ("links", "relation", "TEXT NOT NULL DEFAULT 'related'"),
            ("timeline", "kind", "TEXT NOT NULL DEFAULT 'work'"),
            ("timeline", "team_run_id", "TEXT NOT NULL DEFAULT ''"),
            ("timeline", "node_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
        ] {
            ensure_column(conn, table, col, def)?;
        }

        // 2. 重建分列加权 FTS5，并彻底移除膨胀的 raw_turns_fts
        conn.execute_batch(
            "
            DROP TABLE IF EXISTS raw_turns_fts;
            DROP TABLE IF EXISTS nodes_fts;
            DROP TABLE IF EXISTS timeline_fts;
            DROP TABLE IF EXISTS profiles_fts;
            DROP TABLE IF EXISTS tasks_fts;
            ",
        )
        .map_err(|e| format!("Failed to drop legacy FTS5 tables: {e}"))?;
        create_v3_fts_tables(conn)?;

        // 3. 回填老数据节点要点、分列索引与精确路牌
        rebuild_all_v3_indexes_conn(conn)?;

        conn.execute_batch(&format!("PRAGMA user_version = {CURRENT_SCHEMA_VERSION};"))
            .map_err(|e| e.to_string())?;
        // 迁移后立即截断一次 WAL，释放旧版残留的几 MB WAL 体积
        let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
    } else {
        create_v3_fts_tables(conn)?;
    }

    Ok(())
}

fn insert_anchors_batch(conn: &Connection, anchors: &[super::types::MemoryAnchor]) {
    if anchors.is_empty() {
        return;
    }
    let Ok(mut stmt) = conn.prepare_cached(
        "INSERT OR IGNORE INTO memory_anchors
         (agent_id, project_key, target_kind, target_id, anchor_type, anchor_value, anchor_norm)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    ) else {
        return;
    };
    for a in anchors {
        let _ = stmt.execute(params![
            a.agent_id,
            a.project_key,
            a.target_kind,
            a.target_id,
            a.anchor_type,
            a.anchor_value,
            a.anchor_norm
        ]);
    }
}

/// 全量重建 v3 分列 FTS5 索引与 `memory_anchors`（用于 v2->v3 无损迁移及老库维护修复）。
pub(crate) fn rebuild_all_v3_indexes_conn(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
        DELETE FROM nodes_fts;
        DELETE FROM timeline_fts;
        DELETE FROM profiles_fts;
        DELETE FROM tasks_fts;
        DELETE FROM memory_anchors;
        ",
    )
    .map_err(|e| e.to_string())?;

    // 1. nodes
    let mut node_rows = Vec::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT rowid, id, agent_id, name, project, project_key, summary, events_json,
                        aliases_json, search_tags_json, facts_json, updated_at
                 FROM nodes",
            )
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
        while let Some(r) = rows.next().map_err(|e| e.to_string())? {
            node_rows.push((
                r.get::<_, i64>(0).unwrap_or(0),
                r.get::<_, String>(1).unwrap_or_default(),
                r.get::<_, String>(2).unwrap_or_default(),
                r.get::<_, String>(3).unwrap_or_default(),
                r.get::<_, String>(4).unwrap_or_default(),
                r.get::<_, String>(5).unwrap_or_default(),
                r.get::<_, String>(6).unwrap_or_default(),
                r.get::<_, String>(7).unwrap_or_default(),
                r.get::<_, String>(8).unwrap_or_default(),
                r.get::<_, String>(9).unwrap_or_default(),
                r.get::<_, String>(10).unwrap_or_default(),
                r.get::<_, String>(11).unwrap_or_default(),
            ));
        }
    }
    for (
        rowid,
        id,
        agent_id,
        name,
        project,
        project_key,
        summary,
        events_json,
        aliases_json,
        search_tags_json,
        facts_json,
        updated_at,
    ) in node_rows
    {
        let aliases: Vec<String> = serde_json::from_str(&aliases_json).unwrap_or_default();
        let search_tags: Vec<String> = serde_json::from_str(&search_tags_json).unwrap_or_default();
        let events: Vec<NodeEvent> = serde_json::from_str(&events_json).unwrap_or_default();
        let mut facts: Vec<NodeFact> = serde_json::from_str(&facts_json).unwrap_or_default();
        if facts.is_empty() && !summary.trim().is_empty() {
            facts.push(NodeFact {
                id: new_id("nf"),
                kind: "fact".into(),
                text: summary.trim().to_string(),
                status: "active".into(),
                updated_at,
                ..Default::default()
            });
            if let Ok(fj) = serde_json::to_string(&facts) {
                let _ = conn.execute(
                    "UPDATE nodes SET facts_json = ?1 WHERE id = ?2",
                    params![fj, id],
                );
            }
        }
        let alias_j = aliases.join(" ");
        let tags_j = search_tags.join(" ");
        let facts_j = facts
            .iter()
            .filter(|f| f.is_active())
            .map(|f| f.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let events_j = events
            .iter()
            .take(6)
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let anchors = extract_storage_anchors(
            &agent_id,
            &project_key,
            "node",
            &id,
            &[&name, &alias_j, &tags_j, &summary, &facts_j],
            &[],
            "",
        );
        let anchor_j = anchors
            .iter()
            .map(|a| a.anchor_value.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let _ = conn.execute(
            "INSERT OR REPLACE INTO nodes_fts (rowid, node_id, agent_id, title_col, tags_col, summary_col, body_col)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                rowid,
                id,
                agent_id,
                build_fts_column_text(&[&name, &alias_j]),
                build_fts_column_text(&[&tags_j, &project, &anchor_j]),
                build_fts_column_text(&[&summary, &facts_j]),
                build_fts_column_text(&[&events_j]),
            ],
        );
        insert_anchors_batch(conn, &anchors);
    }

    // 2. timeline
    let mut tl_rows = Vec::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT rowid, id, agent_id, project, project_key, user_goal, summary, ai_outcome,
                        keywords_json, touched_files_json, team_run_id
                 FROM timeline",
            )
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
        while let Some(r) = rows.next().map_err(|e| e.to_string())? {
            tl_rows.push((
                r.get::<_, i64>(0).unwrap_or(0),
                r.get::<_, String>(1).unwrap_or_default(),
                r.get::<_, String>(2).unwrap_or_default(),
                r.get::<_, String>(3).unwrap_or_default(),
                r.get::<_, String>(4).unwrap_or_default(),
                r.get::<_, String>(5).unwrap_or_default(),
                r.get::<_, String>(6).unwrap_or_default(),
                r.get::<_, String>(7).unwrap_or_default(),
                r.get::<_, String>(8).unwrap_or_default(),
                r.get::<_, String>(9).unwrap_or_default(),
                r.get::<_, String>(10).unwrap_or_default(),
            ));
        }
    }
    for (
        rowid,
        id,
        agent_id,
        project,
        project_key,
        goal,
        summary,
        outcome,
        kw_j,
        tf_j,
        team_run_id,
    ) in tl_rows
    {
        let kws: Vec<String> = serde_json::from_str(&kw_j).unwrap_or_default();
        let files: Vec<String> = serde_json::from_str(&tf_j).unwrap_or_default();
        let kw_s = kws.join(" ");
        let tf_s = files.join(" ");
        let _ = conn.execute(
            "INSERT OR REPLACE INTO timeline_fts (rowid, event_id, agent_id, keywords_col, goal_col, summary_col)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                rowid,
                id,
                agent_id,
                build_fts_column_text(&[&kw_s, &tf_s]),
                build_fts_column_text(&[&goal, &project]),
                build_fts_column_text(&[&summary, &outcome]),
            ],
        );
        let anchors = extract_storage_anchors(
            &agent_id,
            &project_key,
            "timeline",
            &id,
            &[&goal, &summary, &outcome, &kw_s],
            &files,
            &team_run_id,
        );
        insert_anchors_batch(conn, &anchors);
    }

    // 3. profiles
    let mut prof_rows = Vec::new();
    {
        let mut stmt = conn
            .prepare("SELECT rowid, id, agent_id, project, project_key, text FROM profiles")
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
        while let Some(r) = rows.next().map_err(|e| e.to_string())? {
            prof_rows.push((
                r.get::<_, i64>(0).unwrap_or(0),
                r.get::<_, String>(1).unwrap_or_default(),
                r.get::<_, String>(2).unwrap_or_default(),
                r.get::<_, String>(3).unwrap_or_default(),
                r.get::<_, String>(4).unwrap_or_default(),
                r.get::<_, String>(5).unwrap_or_default(),
            ));
        }
    }
    for (rowid, id, agent_id, project, project_key, text) in prof_rows {
        let _ = conn.execute(
            "INSERT OR REPLACE INTO profiles_fts (rowid, profile_id, agent_id, text_col)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                rowid,
                id,
                agent_id,
                build_fts_column_text(&[&text, &project])
            ],
        );
        let anchors =
            extract_storage_anchors(&agent_id, &project_key, "profile", &id, &[&text], &[], "");
        insert_anchors_batch(conn, &anchors);
    }

    // 4. tasks
    let mut task_rows = Vec::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT rowid, id, agent_id, project, project_key, title, progress_note,
                        remaining_note, next_step, touched_files_json, team_run_id
                 FROM tasks",
            )
            .map_err(|e| e.to_string())?;
        let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
        while let Some(r) = rows.next().map_err(|e| e.to_string())? {
            task_rows.push((
                r.get::<_, i64>(0).unwrap_or(0),
                r.get::<_, String>(1).unwrap_or_default(),
                r.get::<_, String>(2).unwrap_or_default(),
                r.get::<_, String>(3).unwrap_or_default(),
                r.get::<_, String>(4).unwrap_or_default(),
                r.get::<_, String>(5).unwrap_or_default(),
                r.get::<_, String>(6).unwrap_or_default(),
                r.get::<_, String>(7).unwrap_or_default(),
                r.get::<_, String>(8).unwrap_or_default(),
                r.get::<_, String>(9).unwrap_or_default(),
                r.get::<_, String>(10).unwrap_or_default(),
            ));
        }
    }
    for (rowid, id, agent_id, project, project_key, title, prog, rem, next, tf_j, team_run_id) in
        task_rows
    {
        let files: Vec<String> = serde_json::from_str(&tf_j).unwrap_or_default();
        let tf_s = files.join(" ");
        let _ = conn.execute(
            "INSERT OR REPLACE INTO tasks_fts (rowid, task_id, agent_id, title_col, detail_col)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                rowid,
                id,
                agent_id,
                build_fts_column_text(&[&title, &project]),
                build_fts_column_text(&[&prog, &rem, &next, &tf_s]),
            ],
        );
        let anchors = extract_storage_anchors(
            &agent_id,
            &project_key,
            "task",
            &id,
            &[&title, &prog, &rem, &next],
            &files,
            &team_run_id,
        );
        insert_anchors_batch(conn, &anchors);
    }

    Ok(())
}

/// 打开主写连接并初始化/迁移 Schema。
pub(crate) fn open_connection(data_dir: &Path) -> Result<Connection, String> {
    std::fs::create_dir_all(data_dir)
        .map_err(|e| format!("Failed to create neural memory directory: {e}"))?;
    let db_path = data_dir.join("neural_memory.db");
    let conn = Connection::open(&db_path)
        .map_err(|e| format!("Failed to open neural memory SQLite: {e}"))?;
    init_schema(&conn)?;
    Ok(conn)
}

/// 打开独立只读连接（用于 WAL 读写并发，零阻塞后台写入）。
pub(crate) fn open_read_connection(data_dir: &Path) -> Result<Connection, String> {
    let db_path = data_dir.join("neural_memory.db");
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(&db_path, flags)
        .map_err(|e| format!("Failed to open read-only neural memory SQLite: {e}"))?;
    let _ = conn.busy_timeout(Duration::from_millis(3000));
    let _ = conn.execute_batch(
        "
        PRAGMA query_only = ON;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -8000;
        ",
    );
    Ok(conn)
}
