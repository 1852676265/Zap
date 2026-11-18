//! SQLite 存储层（WAL + 可恢复连接 + 下推查询 + 留存淘汰 + 旧版迁移）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::redact::redact_raw_turn;
#[cfg(test)]
use super::schema::init_schema;
use super::schema::open_connection;
use super::search::{build_fts_search_text, is_noise_node_name};
use super::types::{
    default_windows, new_id, normalize_project_key_str, now_iso, now_ts_minute, NeuralAgent,
    NeuralConfig, NeuralLink, NeuralNode, NeuralTask, ProfileFact, RawTurn, TimelineEvent,
    MAX_NODES_PER_AGENT, MAX_NON_PINNED_PROFILES, MAX_RAW_TURNS_PER_AGENT, MAX_TIMELINE_PER_AGENT,
};

pub(crate) struct StoreInner {
    pub(crate) data_dir: PathBuf,
    pub(crate) conn: Connection,
}

static STORE: OnceLock<Mutex<Option<StoreInner>>> = OnceLock::new();

#[derive(Serialize, Deserialize, Default)]
struct LocationPointer {
    #[serde(default)]
    data_dir: String,
}

fn pointer_file_path() -> PathBuf {
    zap_config::Config::config_dir()
        .join("neural-memory")
        .join("location.json")
}

pub(crate) fn default_data_dir() -> PathBuf {
    zap_config::Config::config_dir().join("neural-memory")
}

#[cfg(not(test))]
fn resolve_configured_data_dir() -> PathBuf {
    let ptr = pointer_file_path();
    if let Ok(raw) = std::fs::read_to_string(&ptr) {
        if let Ok(parsed) = serde_json::from_str::<LocationPointer>(&raw) {
            let trimmed = parsed.data_dir.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
    }
    default_data_dir()
}

#[cfg(test)]
fn resolve_configured_data_dir() -> PathBuf {
    static TEST_DIR: OnceLock<PathBuf> = OnceLock::new();
    TEST_DIR
        .get_or_init(|| {
            let dir =
                std::env::temp_dir().join(format!("zap-neural-memory-test-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            dir
        })
        .clone()
}

pub(crate) fn save_data_dir_pointer(dir: &Path) -> Result<(), String> {
    let ptr = pointer_file_path();
    if let Some(parent) = ptr.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let payload = serde_json::to_string_pretty(&LocationPointer {
        data_dir: dir.to_string_lossy().into_owned(),
    })
    .map_err(|e| e.to_string())?;
    std::fs::write(&ptr, payload).map_err(|e| e.to_string())
}

/// 获取全局 SQLite 存储连接。若打开失败不会永久退回内存库，而是返回错误并在下次调用重试。
pub(crate) fn with_store<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut StoreInner) -> Result<R, String>,
{
    let lock = STORE.get_or_init(|| Mutex::new(None));
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let dir = resolve_configured_data_dir();
        let mut conn = open_connection(&dir)?;
        bootstrap_initial_data(&mut conn, &dir);
        *guard = Some(StoreInner {
            data_dir: dir,
            conn,
        });
    }
    let inner = guard.as_mut().expect("store initialized above");
    f(inner)
}

pub(crate) fn default_agents() -> Vec<NeuralAgent> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// 自动迁移旧版数据（移除硬编码外部开发路径）
// ---------------------------------------------------------------------------

fn find_legacy_data_dirs() -> Vec<PathBuf> {
    if cfg!(test) {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let plugins_root = zap_config::Config::config_dir().join("plugins");
    if let Ok(entries) = std::fs::read_dir(&plugins_root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.contains("neural") {
                    candidates.push(p.join("data"));
                }
                if let Ok(sub) = std::fs::read_dir(&p) {
                    for s in sub.flatten() {
                        if s.path().is_dir()
                            && s.file_name()
                                .to_string_lossy()
                                .to_lowercase()
                                .contains("neural")
                        {
                            candidates.push(s.path().join("data"));
                        }
                    }
                }
            }
        }
    }
    candidates
}

fn find_legacy_storage_json() -> Option<PathBuf> {
    let p = zap_config::Config::config_dir()
        .join("plugin-storage")
        .join("_zap_neural-agents")
        .join("storage.json");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

pub(crate) fn bootstrap_initial_data(conn: &mut Connection, _data_dir: &Path) {
    let already_migrated: bool = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'migration_done'",
            [],
            |r| r.get::<_, String>(0),
        )
        .optional()
        .ok()
        .flatten()
        .is_some();
    if already_migrated {
        return;
    }

    for (idx, a) in default_agents().iter().enumerate() {
        let _ = upsert_agent_conn(conn, a, idx as i64);
    }

    let mut imported_any = false;
    for dir in find_legacy_data_dirs() {
        if dir.is_dir() && dir.join("agents.json").is_file() {
            if let Err(err) = import_legacy_dir_conn(conn, &dir) {
                tracing::warn!(dir = %dir.display(), error = %err, "neural_memory: legacy import error");
            } else {
                imported_any = true;
                tracing::info!(dir = %dir.display(), "neural_memory: imported legacy plugin data");
                break;
            }
        }
    }

    if let Some(storage_path) = find_legacy_storage_json() {
        if let Ok(raw) = std::fs::read_to_string(&storage_path) {
            if let Ok(val) = serde_json::from_str::<Value>(&raw) {
                if let Some(obj) = val.as_object() {
                    let cfg = NeuralConfig {
                        active_agent: obj
                            .get("activeAgent")
                            .and_then(|v| v.as_str())
                            .unwrap_or("none")
                            .to_string(),
                        distill_mode: obj
                            .get("distillMode")
                            .and_then(|v| v.as_str())
                            .unwrap_or("llm_auto")
                            .to_string(),
                        distill_model: obj
                            .get("distillModel")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        timeline_auto_capture: obj
                            .get("timelineAutoCapture")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                        show_floating_capsule: obj
                            .get("showFloatingCapsule")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                        memory_data_dir: obj
                            .get("memoryDataDir")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    };
                    let _ = save_config_conn(conn, &cfg);
                    if let Some(smap) = obj.get("sessionAgentMap").and_then(|v| v.as_object()) {
                        for (sid, aid) in smap {
                            if let Some(a) = aid.as_str() {
                                let _ = set_session_agent_conn(conn, sid, a);
                            }
                        }
                    }
                }
            }
        }
    }

    let _ = conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('migration_done', ?1)",
        params![if imported_any { "imported" } else { "clean" }],
    );
}

pub(crate) fn import_legacy_dir_conn(conn: &mut Connection, dir: &Path) -> Result<Value, String> {
    let mut stats = json!({
        "agents": 0,
        "nodes": 0,
        "links": 0,
        "timeline": 0,
        "tasks": 0,
        "profiles": 0,
        "rawTurns": 0,
        "cleanedNoise": 0
    });
    let mut cleaned_noise: usize = 0;

    if let Ok(raw) = std::fs::read_to_string(dir.join("agents.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<NeuralAgent>>(&raw) {
            if !list.is_empty() {
                conn.execute("DELETE FROM agents", []).ok();
                for (idx, a) in list.iter().enumerate() {
                    upsert_agent_conn(conn, a, idx as i64)?;
                }
                stats["agents"] = json!(list.len());
            }
        }
    }

    if let Ok(raw) = std::fs::read_to_string(dir.join("session-map.json")) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&raw) {
            for (sid, aid) in map {
                set_session_agent_conn(conn, &sid, &aid)?;
            }
        }
    }

    let mut removed_node_ids: HashSet<String> = HashSet::new();
    if let Ok(raw) = std::fs::read_to_string(dir.join("graph.json")) {
        if let Ok(val) = serde_json::from_str::<Value>(&raw) {
            let mut node_count = 0usize;
            if let Some(nodes) = val.get("nodes").and_then(|v| v.as_array()) {
                for nv in nodes {
                    if let Ok(n) = serde_json::from_value::<NeuralNode>(nv.clone()) {
                        if is_noise_node_name(&n.name) {
                            removed_node_ids.insert(n.id);
                            cleaned_noise += 1;
                            continue;
                        }
                        upsert_node_conn(conn, &n)?;
                        node_count += 1;
                    }
                }
            }
            stats["nodes"] = json!(node_count);

            let mut link_count = 0usize;
            if let Some(links) = val.get("links").and_then(|v| v.as_array()) {
                for lv in links {
                    let src = lv.get("source").and_then(|v| v.as_str()).unwrap_or("");
                    let tgt = lv.get("target").and_then(|v| v.as_str()).unwrap_or("");
                    if !src.is_empty()
                        && !tgt.is_empty()
                        && !removed_node_ids.contains(src)
                        && !removed_node_ids.contains(tgt)
                    {
                        conn.execute(
                            "INSERT OR IGNORE INTO links (source, target) VALUES (?1, ?2)",
                            params![src, tgt],
                        )
                        .ok();
                        link_count += 1;
                    }
                }
            }
            stats["links"] = json!(link_count);
        }
    }

    if let Ok(raw) = std::fs::read_to_string(dir.join("profiles.json")) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<ProfileFact>>>(&raw) {
            let mut count = 0usize;
            for (agent_id, list) in map {
                for f in list {
                    upsert_profile_conn(conn, &agent_id, &f)?;
                    count += 1;
                }
            }
            stats["profiles"] = json!(count);
        }
    }

    if let Ok(raw) = std::fs::read_to_string(dir.join("tasks.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<NeuralTask>>(&raw) {
            for mut t in list {
                if t.project_key.is_empty() && !t.working_dir.is_empty() {
                    t.project_key = normalize_project_key_str(&t.working_dir);
                }
                upsert_task_conn(conn, &t)?;
            }
            stats["tasks"] = json!(load_tasks_conn(conn, None).map(|v| v.len()).unwrap_or(0));
        }
    }

    if let Ok(raw) = std::fs::read_to_string(dir.join("timeline.json")) {
        if let Ok(list) = serde_json::from_str::<Vec<TimelineEvent>>(&raw) {
            let mut count = 0usize;
            for mut ev in list {
                let sum = ev.summary.trim();
                let goal = ev.user_goal.trim();
                if ev.agent.trim().is_empty()
                    || ev.agent == "hubs"
                    || (sum.is_empty() && goal.is_empty())
                    || (is_noise_node_name(sum) && goal.is_empty())
                {
                    cleaned_noise += 1;
                    continue;
                }
                if ev.id.is_empty() {
                    ev.id = new_id("tl");
                }
                if ev.project_key.is_empty() && !ev.working_dir.is_empty() {
                    ev.project_key = normalize_project_key_str(&ev.working_dir);
                }
                insert_timeline_conn(conn, &ev)?;
                count += 1;
            }
            stats["timeline"] = json!(count);
        }
    }

    let mut raw_count = 0usize;
    for sub in ["raw", "archive"] {
        let sub_dir = dir.join(sub);
        if let Ok(entries) = std::fs::read_dir(&sub_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(raw) = std::fs::read_to_string(&p) {
                        if let Ok(turns) = serde_json::from_str::<Vec<RawTurn>>(&raw) {
                            for mut t in turns {
                                if t.project_key.is_empty() && !t.working_dir.is_empty() {
                                    t.project_key = normalize_project_key_str(&t.working_dir);
                                }
                                upsert_raw_turn_conn(conn, &t)?;
                                raw_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    stats["rawTurns"] = json!(raw_count);
    stats["cleanedNoise"] = json!(cleaned_noise);
    Ok(stats)
}

// ---------------------------------------------------------------------------
// 底层 SQLite CRUD 封装
// ---------------------------------------------------------------------------

pub(crate) fn upsert_agent_conn(
    conn: &Connection,
    a: &NeuralAgent,
    sort_order: i64,
) -> Result<(), String> {
    let label = if a.label.trim().is_empty() {
        format!("{} {}", a.icon, a.name)
    } else {
        a.label.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO agents (id, icon, name, label, color, hint, system_prompt, sort_order)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            a.id,
            a.icon,
            a.name,
            label,
            a.color,
            a.hint,
            a.system_prompt,
            sort_order
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn load_agents_conn(conn: &Connection) -> Result<Vec<NeuralAgent>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT id, icon, name, label, color, hint, system_prompt
             FROM agents ORDER BY sort_order ASC, id ASC",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(NeuralAgent {
                id: row.get(0)?,
                icon: row.get(1)?,
                name: row.get(2)?,
                label: row.get(3)?,
                color: row.get(4)?,
                hint: row.get(5)?,
                system_prompt: row.get(6)?,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn has_any_agents_conn(conn: &Connection) -> bool {
    conn.query_row("SELECT 1 FROM agents LIMIT 1", [], |_| Ok(true))
        .optional()
        .ok()
        .flatten()
        .unwrap_or(false)
}

pub(crate) fn get_config_conn(conn: &Connection) -> NeuralConfig {
    let mut cfg = NeuralConfig::default();
    if let Ok(mut stmt) = conn.prepare("SELECT key, value FROM meta") {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        {
            for r in rows.flatten() {
                match r.0.as_str() {
                    "cfg_active_agent" => cfg.active_agent = r.1,
                    "cfg_distill_mode" => cfg.distill_mode = r.1,
                    "cfg_distill_model" => cfg.distill_model = r.1,
                    "cfg_timeline_auto_capture" => cfg.timeline_auto_capture = r.1 != "false",
                    "cfg_show_floating_capsule" => cfg.show_floating_capsule = r.1 != "false",
                    "cfg_memory_data_dir" => cfg.memory_data_dir = r.1,
                    _ => {}
                }
            }
        }
    }
    cfg
}

pub(crate) fn save_config_conn(conn: &Connection, cfg: &NeuralConfig) -> Result<(), String> {
    let pairs = [
        ("cfg_active_agent", cfg.active_agent.as_str()),
        ("cfg_distill_mode", cfg.distill_mode.as_str()),
        ("cfg_distill_model", cfg.distill_model.as_str()),
        (
            "cfg_timeline_auto_capture",
            if cfg.timeline_auto_capture {
                "true"
            } else {
                "false"
            },
        ),
        (
            "cfg_show_floating_capsule",
            if cfg.show_floating_capsule {
                "true"
            } else {
                "false"
            },
        ),
        ("cfg_memory_data_dir", cfg.memory_data_dir.as_str()),
    ];
    for (k, v) in pairs {
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![k, v],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn set_session_agent_conn(
    conn: &Connection,
    session_id: &str,
    agent_id: &str,
) -> Result<(), String> {
    if session_id.trim().is_empty() {
        return Ok(());
    }
    conn.execute(
        "INSERT OR REPLACE INTO session_agents (session_id, agent_id, updated_at) VALUES (?1, ?2, ?3)",
        params![session_id.trim(), agent_id.trim(), now_iso()],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// 为刚创建完成的 Fresh 会话原子绑定智能体，并消费清空全局草稿态 `cfg_active_agent`。
pub(crate) fn bind_fresh_session_agent_if_unset_conn(
    conn: &Connection,
    session_id: &str,
    agent_id: &str,
) -> Result<(), String> {
    let sid = session_id.trim();
    let aid = agent_id.trim();
    if sid.is_empty() || aid.is_empty() || aid == "none" {
        return Ok(());
    }
    conn.execute(
        "INSERT OR IGNORE INTO session_agents (session_id, agent_id, updated_at) VALUES (?1, ?2, ?3)",
        params![sid, aid, now_iso()],
    )
    .map_err(|e| e.to_string())?;
    let mut cfg = get_config_conn(conn);
    if cfg.active_agent == aid {
        cfg.active_agent = "none".to_string();
        let _ = save_config_conn(conn, &cfg);
    }
    Ok(())
}

pub(crate) fn resolve_agent_for_session_conn(
    conn: &Connection,
    session_id: &str,
) -> Option<NeuralAgent> {
    let cfg = get_config_conn(conn);
    let mapped: Option<String> = if session_id.trim().is_empty() {
        None
    } else {
        conn.query_row(
            "SELECT agent_id FROM session_agents WHERE session_id = ?1",
            params![session_id.trim()],
            |r| r.get(0),
        )
        .optional()
        .ok()
        .flatten()
    };
    let agent_id = mapped.unwrap_or_else(|| cfg.active_agent.clone());
    if agent_id.trim().is_empty() || agent_id == "none" {
        return None;
    }
    load_agents_conn(conn)
        .ok()?
        .into_iter()
        .find(|a| a.id == agent_id)
}

pub(crate) fn upsert_profile_conn(
    conn: &Connection,
    agent_id: &str,
    f: &ProfileFact,
) -> Result<(), String> {
    let id = if f.id.trim().is_empty() {
        new_id("pf")
    } else {
        f.id.clone()
    };
    let updated_at = if f.updated_at.trim().is_empty() {
        now_ts_minute()
    } else {
        f.updated_at.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO profiles (id, agent_id, project, project_key, text, pinned, updated_at, source_turn_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            id,
            agent_id,
            f.project,
            f.project_key,
            f.text,
            if f.pinned { 1 } else { 0 },
            updated_at,
            f.source_turn_id
        ],
    )
    .map_err(|e| e.to_string())?;

    // 按 (agent_id, project_key, project) 分项目淘汰非置顶老事实
    conn.execute(
        "DELETE FROM profiles WHERE agent_id = ?1 AND project_key = ?2 AND project = ?3 AND pinned = 0 AND id NOT IN (
             SELECT id FROM profiles WHERE agent_id = ?1 AND project_key = ?2 AND project = ?3 AND pinned = 0
             ORDER BY updated_at DESC, rowid DESC LIMIT ?4
         )",
        params![agent_id, f.project_key, f.project, MAX_NON_PINNED_PROFILES as i64],
    )
    .ok();
    Ok(())
}

pub(crate) fn load_profiles_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<HashMap<String, Vec<ProfileFact>>, String> {
    let mut map: HashMap<String, Vec<ProfileFact>> = HashMap::new();
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, project, project_key, text, pinned, updated_at, source_turn_id
         FROM profiles WHERE agent_id = ?1 ORDER BY pinned DESC, updated_at DESC"
    } else {
        "SELECT id, agent_id, project, project_key, text, pinned, updated_at, source_turn_id
         FROM profiles ORDER BY pinned DESC, updated_at DESC"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(String, ProfileFact)> {
        let aid: String = row.get(1)?;
        let pinned_int: i64 = row.get(5)?;
        Ok((
            aid,
            ProfileFact {
                id: row.get(0)?,
                project: row.get(2)?,
                project_key: row.get(3)?,
                text: row.get(4)?,
                pinned: pinned_int != 0,
                updated_at: row.get(6)?,
                source_turn_id: row.get(7)?,
            },
        ))
    };
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            let (a, f) = r.map_err(|e| e.to_string())?;
            map.entry(a).or_default().push(f);
        }
    } else {
        let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
        for r in rows {
            let (a, f) = r.map_err(|e| e.to_string())?;
            map.entry(a).or_default().push(f);
        }
    }
    Ok(map)
}

/// SQL 下推：按项目作用域精确读取 L1 长期档案（置顶与当前项目优先）。
pub(crate) fn load_profiles_scoped_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
    limit: usize,
) -> Result<Vec<ProfileFact>, String> {
    let sql = if project_key.is_empty() && project_name.is_empty() {
        "SELECT id, project, project_key, text, pinned, updated_at, source_turn_id
         FROM profiles
         WHERE agent_id = ?1
         ORDER BY pinned DESC, updated_at DESC, rowid DESC
         LIMIT ?2"
    } else {
        "SELECT id, project, project_key, text, pinned, updated_at, source_turn_id
         FROM profiles
         WHERE agent_id = ?1
           AND (
             (project_key = '' AND project = '')
             OR (?2 != '' AND project_key = ?2)
             OR (project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3))
           )
         ORDER BY
           pinned DESC,
           CASE
             WHEN ?2 != '' AND project_key = ?2 THEN 0
             WHEN project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3) THEN 1
             ELSE 2
           END ASC,
           updated_at DESC,
           rowid DESC
         LIMIT ?4"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<ProfileFact> {
        let pinned_int: i64 = row.get(4)?;
        Ok(ProfileFact {
            id: row.get(0)?,
            project: row.get(1)?,
            project_key: row.get(2)?,
            text: row.get(3)?,
            pinned: pinned_int != 0,
            updated_at: row.get(5)?,
            source_turn_id: row.get(6)?,
        })
    };
    let rows = if project_key.is_empty() && project_name.is_empty() {
        stmt.query_map(params![agent_id, limit as i64], mapper)
    } else {
        stmt.query_map(
            params![agent_id, project_key, project_name, limit as i64],
            mapper,
        )
    }
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn upsert_task_conn(conn: &Connection, t: &NeuralTask) -> Result<(), String> {
    let touched = serde_json::to_string(&t.touched_files).unwrap_or_else(|_| "[]".into());
    let turns = serde_json::to_string(&t.source_turn_ids).unwrap_or_else(|_| "[]".into());
    let pkey = if t.project_key.is_empty() && !t.working_dir.is_empty() {
        normalize_project_key_str(&t.working_dir)
    } else {
        t.project_key.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO tasks (
            id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
            touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            t.id,
            t.agent,
            t.project,
            pkey,
            t.working_dir,
            t.title,
            t.progress_note,
            t.remaining_note,
            touched,
            t.session_id,
            turns,
            t.status,
            t.created_at,
            t.updated_at,
            t.done_at
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(crate) fn load_tasks_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<Vec<NeuralTask>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
         FROM tasks WHERE agent_id = ?1 ORDER BY updated_at DESC"
    } else {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
         FROM tasks ORDER BY updated_at DESC"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<NeuralTask> {
        let touched_str: String = row.get(8)?;
        let turns_str: String = row.get(10)?;
        Ok(NeuralTask {
            id: row.get(0)?,
            agent: row.get(1)?,
            project: row.get(2)?,
            project_key: row.get(3)?,
            working_dir: row.get(4)?,
            title: row.get(5)?,
            progress_note: row.get(6)?,
            remaining_note: row.get(7)?,
            touched_files: serde_json::from_str(&touched_str).unwrap_or_default(),
            session_id: row.get(9)?,
            source_turn_ids: serde_json::from_str(&turns_str).unwrap_or_default(),
            status: row.get(11)?,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
            done_at: row.get(14)?,
        })
    };
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

/// SQL 下推：按项目作用域读取未完成任务，避免全表读取。
pub(crate) fn load_open_tasks_scoped_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
    limit: usize,
) -> Result<Vec<NeuralTask>, String> {
    let sql = if project_key.is_empty() && project_name.is_empty() {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
         FROM tasks
         WHERE agent_id = ?1 AND status != 'done'
         ORDER BY updated_at DESC, rowid DESC
         LIMIT ?2"
    } else {
        "SELECT id, agent_id, project, project_key, working_dir, title, progress_note, remaining_note,
                touched_files_json, session_id, source_turn_ids_json, status, created_at, updated_at, done_at
         FROM tasks
         WHERE agent_id = ?1
           AND status != 'done'
           AND (
             (project_key = '' AND project = '')
             OR (?2 != '' AND project_key = ?2)
             OR (project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3))
           )
         ORDER BY
           CASE
             WHEN ?2 != '' AND project_key = ?2 THEN 0
             WHEN project_key = '' AND ?3 != '' AND LOWER(project) = LOWER(?3) THEN 1
             ELSE 2
           END ASC,
           updated_at DESC,
           rowid DESC
         LIMIT ?4"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<NeuralTask> {
        let touched_str: String = row.get(8)?;
        let turns_str: String = row.get(10)?;
        Ok(NeuralTask {
            id: row.get(0)?,
            agent: row.get(1)?,
            project: row.get(2)?,
            project_key: row.get(3)?,
            working_dir: row.get(4)?,
            title: row.get(5)?,
            progress_note: row.get(6)?,
            remaining_note: row.get(7)?,
            touched_files: serde_json::from_str(&touched_str).unwrap_or_default(),
            session_id: row.get(9)?,
            source_turn_ids: serde_json::from_str(&turns_str).unwrap_or_default(),
            status: row.get(11)?,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
            done_at: row.get(14)?,
        })
    };
    let rows = if project_key.is_empty() && project_name.is_empty() {
        stmt.query_map(params![agent_id, limit as i64], mapper)
    } else {
        stmt.query_map(
            params![agent_id, project_key, project_name, limit as i64],
            mapper,
        )
    }
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn upsert_node_conn(conn: &Connection, n: &NeuralNode) -> Result<(), String> {
    let windows_json = serde_json::to_string(&n.windows).unwrap_or_else(|_| "[\"all\"]".into());
    let events_json = serde_json::to_string(&n.events).unwrap_or_else(|_| "[]".into());
    let updated_at = if n.updated_at.trim().is_empty() {
        now_ts_minute()
    } else {
        n.updated_at.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO nodes (id, agent_id, name, project, project_key, status, mastery, tier, windows_json, summary, events_json, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            n.id,
            n.cat,
            n.name,
            n.project,
            n.project_key,
            n.status,
            n.mastery,
            n.tier,
            windows_json,
            n.summary,
            events_json,
            updated_at
        ],
    )
    .map_err(|e| e.to_string())?;

    conn.execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![n.id])
        .ok();
    let event_texts: Vec<&str> = n.events.iter().map(|e| e.text.as_str()).collect();
    let ev_joined = event_texts.join(" ");
    let search_text = build_fts_search_text(&[&n.name, &n.project, &n.summary, &ev_joined]);
    conn.execute(
        "INSERT INTO nodes_fts (node_id, agent_id, search_text) VALUES (?1, ?2, ?3)",
        params![n.id, n.cat, search_text],
    )
    .ok();
    Ok(())
}

pub(crate) fn load_nodes_conn(
    conn: &Connection,
    agent_id: Option<&str>,
) -> Result<Vec<NeuralNode>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json, summary, events_json, updated_at
         FROM nodes WHERE agent_id = ?1 ORDER BY updated_at DESC"
    } else {
        "SELECT id, agent_id, name, project, project_key, status, mastery, tier, windows_json, summary, events_json, updated_at
         FROM nodes ORDER BY updated_at DESC"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<NeuralNode> {
        let win_str: String = row.get(8)?;
        let ev_str: String = row.get(10)?;
        Ok(NeuralNode {
            id: row.get(0)?,
            cat: row.get(1)?,
            name: row.get(2)?,
            project: row.get(3)?,
            project_key: row.get(4)?,
            status: row.get(5)?,
            mastery: row.get(6)?,
            tier: row.get(7)?,
            windows: serde_json::from_str(&win_str).unwrap_or_else(|_| default_windows()),
            summary: row.get(9)?,
            events: serde_json::from_str(&ev_str).unwrap_or_default(),
            updated_at: row.get(11)?,
        })
    };
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt.query_map([], mapper).map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

/// 仅查询当前/全局范围内最近的若干节点名称（供提炼提示复用已有节点，避免全表加载）。
pub(crate) fn load_recent_node_names_conn(
    conn: &Connection,
    agent_id: &str,
    project_key: &str,
    project_name: &str,
    limit: usize,
) -> Vec<String> {
    let sql = if project_key.is_empty() && project_name.is_empty() {
        "SELECT name FROM nodes WHERE agent_id = ?1 ORDER BY updated_at DESC LIMIT ?2"
    } else {
        "SELECT name FROM nodes
         WHERE agent_id = ?1
           AND ((?2 != '' AND project_key = ?2) OR (project_key = '' AND LOWER(project) = LOWER(?3)) OR project = '')
         ORDER BY updated_at DESC LIMIT ?4"
    };
    let Ok(mut stmt) = conn.prepare(sql) else {
        return Vec::new();
    };
    if project_key.is_empty() && project_name.is_empty() {
        stmt.query_map(params![agent_id, limit as i64], |r| r.get::<_, String>(0))
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
    } else {
        stmt.query_map(
            params![agent_id, project_key, project_name, limit as i64],
            |r| r.get::<_, String>(0),
        )
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
    }
}

pub(crate) fn load_links_conn(conn: &Connection) -> Result<Vec<NeuralLink>, String> {
    let mut stmt = conn
        .prepare("SELECT source, target FROM links")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok(NeuralLink {
                source: r.get(0)?,
                target: r.get(1)?,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

pub(crate) fn insert_timeline_conn(conn: &Connection, ev: &TimelineEvent) -> Result<(), String> {
    let id = if ev.id.trim().is_empty() {
        new_id("tl")
    } else {
        ev.id.clone()
    };
    let kw_json = serde_json::to_string(&ev.keywords).unwrap_or_else(|_| "[]".into());
    let tf_json = serde_json::to_string(&ev.touched_files).unwrap_or_else(|_| "[]".into());
    let pkey = if ev.project_key.is_empty() && !ev.working_dir.is_empty() {
        normalize_project_key_str(&ev.working_dir)
    } else {
        ev.project_key.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO timeline (
            id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
            user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            id,
            ev.ts,
            ev.agent,
            ev.project,
            pkey,
            ev.working_dir,
            ev.session_id,
            ev.turn_id,
            ev.raw_file,
            ev.user_goal,
            ev.summary,
            ev.ai_outcome,
            kw_json,
            tf_json,
            ev.distilled_by
        ],
    )
    .map_err(|e| e.to_string())?;

    conn.execute("DELETE FROM timeline_fts WHERE event_id = ?1", params![id])
        .ok();
    let kw_joined = ev.keywords.join(" ");
    let tf_joined = ev.touched_files.join(" ");
    let search_text = build_fts_search_text(&[
        &ev.project,
        &ev.user_goal,
        &ev.summary,
        &ev.ai_outcome,
        &kw_joined,
        &tf_joined,
    ]);
    conn.execute(
        "INSERT INTO timeline_fts (event_id, agent_id, search_text) VALUES (?1, ?2, ?3)",
        params![id, ev.agent, search_text],
    )
    .ok();
    Ok(())
}

pub(crate) fn load_timeline_conn(
    conn: &Connection,
    agent_id: Option<&str>,
    limit: usize,
) -> Result<Vec<TimelineEvent>, String> {
    let sql = if agent_id.is_some() {
        "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
         FROM timeline WHERE agent_id = ?1 ORDER BY ts DESC LIMIT ?2"
    } else {
        "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
         FROM timeline ORDER BY ts DESC LIMIT ?1"
    };
    let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
    let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<TimelineEvent> {
        let kw_str: String = row.get(12)?;
        let tf_str: String = row.get(13)?;
        Ok(TimelineEvent {
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
    };
    let mut out = Vec::new();
    if let Some(aid) = agent_id {
        let rows = stmt
            .query_map(params![aid, limit as i64], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    } else {
        let rows = stmt
            .query_map(params![limit as i64], mapper)
            .map_err(|e| e.to_string())?;
        for r in rows {
            out.push(r.map_err(|e| e.to_string())?);
        }
    }
    Ok(out)
}

pub(crate) fn upsert_raw_turn_conn(conn: &Connection, t: &RawTurn) -> Result<(), String> {
    let clean = redact_raw_turn(t.clone());
    let tf_json = serde_json::to_string(&clean.touched_files).unwrap_or_else(|_| "[]".into());
    let pkey = if clean.project_key.is_empty() && !clean.working_dir.is_empty() {
        normalize_project_key_str(&clean.working_dir)
    } else {
        clean.project_key.clone()
    };
    conn.execute(
        "INSERT OR REPLACE INTO raw_turns (turn_id, ts, agent_id, project, project_key, working_dir, session_id, user_text, ai_text, touched_files_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            clean.turn_id,
            clean.ts,
            clean.agent,
            clean.project,
            pkey,
            clean.working_dir,
            clean.session_id,
            clean.user_text,
            clean.ai_text,
            tf_json
        ],
    )
    .map_err(|e| e.to_string())?;

    conn.execute(
        "DELETE FROM raw_turns_fts WHERE turn_id = ?1",
        params![clean.turn_id],
    )
    .ok();
    let tf_joined = clean.touched_files.join(" ");
    let search_text =
        build_fts_search_text(&[&clean.project, &clean.user_text, &clean.ai_text, &tf_joined]);
    conn.execute(
        "INSERT INTO raw_turns_fts (turn_id, agent_id, search_text) VALUES (?1, ?2, ?3)",
        params![clean.turn_id, clean.agent, search_text],
    )
    .ok();
    Ok(())
}

pub(crate) fn load_raw_turn_conn(
    conn: &Connection,
    turn_id: &str,
) -> Result<Option<RawTurn>, String> {
    conn.query_row(
        "SELECT turn_id, ts, agent_id, project, project_key, working_dir, session_id, user_text, ai_text, touched_files_json
         FROM raw_turns WHERE turn_id = ?1",
        params![turn_id],
        |row| {
            let tf_str: String = row.get(9)?;
            Ok(RawTurn {
                turn_id: row.get(0)?,
                ts: row.get(1)?,
                agent: row.get(2)?,
                project: row.get(3)?,
                project_key: row.get(4)?,
                working_dir: row.get(5)?,
                session_id: row.get(6)?,
                user_text: row.get(7)?,
                ai_text: row.get(8)?,
                touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
            })
        },
    )
    .optional()
    .map_err(|e| e.to_string())
}

#[derive(Debug, Clone)]
pub(crate) struct KnownProject {
    pub(crate) name: String,
    pub(crate) key: String,
}

/// 查询库内所有出现过的项目（同时保留展示名与规范化路径键）。
pub(crate) fn list_known_projects_conn(conn: &Connection, agent_id: &str) -> Vec<KnownProject> {
    let mut seen: HashMap<String, KnownProject> = HashMap::new();
    for sql in [
        "SELECT DISTINCT project, project_key FROM nodes WHERE agent_id = ?1 AND (project != '' OR project_key != '')",
        "SELECT DISTINCT project, project_key FROM timeline WHERE agent_id = ?1 AND (project != '' OR project_key != '')",
        "SELECT DISTINCT project, project_key FROM tasks WHERE agent_id = ?1 AND (project != '' OR project_key != '')",
    ] {
        if let Ok(mut stmt) = conn.prepare(sql) {
            if let Ok(rows) = stmt.query_map(params![agent_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            }) {
                for (name, key) in rows.flatten() {
                    let clean_name = name.trim().to_string();
                    let clean_key = key.trim().to_string();
                    if clean_name.is_empty() && clean_key.is_empty() {
                        continue;
                    }
                    let dedup_k = if !clean_key.is_empty() {
                        clean_key.clone()
                    } else {
                        clean_name.to_lowercase()
                    };
                    seen.entry(dedup_k).or_insert(KnownProject {
                        name: clean_name,
                        key: clean_key,
                    });
                }
            }
        }
    }
    let mut out: Vec<KnownProject> = seen.into_values().collect();
    out.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.key.cmp(&b.key)));
    out
}

/// 控制每个智能体的存储规模，在单一保存点事务内批量淘汰最旧的原文、时间线与冷门节点，避免频繁逐行刷盘。
pub(crate) fn prune_retention_conn(conn: &Connection, agent_id: &str) {
    if conn.execute_batch("SAVEPOINT neural_prune;").is_err() {
        return;
    }
    let res = (|| -> rusqlite::Result<()> {
        // 1. 批量淘汰超出上限的 raw_turns 及对应 FTS
        conn.execute(
            "DELETE FROM raw_turns_fts WHERE turn_id IN (
                 SELECT turn_id FROM raw_turns WHERE agent_id = ?1
                 ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?2
             )",
            params![agent_id, MAX_RAW_TURNS_PER_AGENT as i64],
        )?;
        conn.execute(
            "DELETE FROM raw_turns WHERE turn_id IN (
                 SELECT turn_id FROM raw_turns WHERE agent_id = ?1
                 ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?2
             )",
            params![agent_id, MAX_RAW_TURNS_PER_AGENT as i64],
        )?;

        // 2. 批量淘汰超出上限的 timeline 及对应 FTS
        conn.execute(
            "DELETE FROM timeline_fts WHERE event_id IN (
                 SELECT id FROM timeline WHERE agent_id = ?1
                 ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?2
             )",
            params![agent_id, MAX_TIMELINE_PER_AGENT as i64],
        )?;
        conn.execute(
            "DELETE FROM timeline WHERE id IN (
                 SELECT id FROM timeline WHERE agent_id = ?1
                 ORDER BY ts DESC, rowid DESC LIMIT -1 OFFSET ?2
             )",
            params![agent_id, MAX_TIMELINE_PER_AGENT as i64],
        )?;

        // 3. 批量淘汰超出上限的 nodes（优先保留 solid 状态、高连接度 Hub 与高掌握度节点）
        let prune_ids: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT n.id FROM nodes n
                 WHERE n.agent_id = ?1
                 ORDER BY
                   CASE WHEN n.status = 'solid' THEN 1 ELSE 0 END DESC,
                   (SELECT COUNT(*) FROM links l WHERE l.source = n.id OR l.target = n.id) DESC,
                   n.mastery DESC,
                   n.updated_at DESC,
                   n.rowid DESC
                 LIMIT -1 OFFSET ?2",
            )?;
            let rows = stmt.query_map(params![agent_id, MAX_NODES_PER_AGENT as i64], |r| {
                r.get::<_, String>(0)
            })?;
            rows.flatten().collect()
        };
        for nid in prune_ids {
            conn.execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![nid])?;
            conn.execute(
                "DELETE FROM links WHERE source = ?1 OR target = ?1",
                params![nid],
            )?;
            conn.execute("DELETE FROM nodes WHERE id = ?1", params![nid])?;
        }
        Ok(())
    })();
    if res.is_ok() {
        let _ = conn.execute_batch("RELEASE SAVEPOINT neural_prune;");
    } else {
        let _ = conn
            .execute_batch("ROLLBACK TO SAVEPOINT neural_prune; RELEASE SAVEPOINT neural_prune;");
    }
}

/// 测试辅助：创建内存 SQLite 并初始化表结构。
#[cfg(test)]
pub(crate) fn setup_mem_db() -> Connection {
    let conn = Connection::open_in_memory().expect("in-memory sqlite");
    init_schema(&conn).expect("init_schema");
    for (idx, a) in default_agents().iter().enumerate() {
        upsert_agent_conn(&conn, a, idx as i64).expect("seed agent");
    }
    let cfg = NeuralConfig {
        active_agent: "backend".into(),
        ..Default::default()
    };
    save_config_conn(&conn, &cfg).expect("save cfg");
    conn
}
