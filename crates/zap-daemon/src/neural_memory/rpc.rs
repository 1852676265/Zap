//! 前端与插件 RPC 分发器（`/api/plugins-rpc/neural-agents`）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde_json::{json, Value};

use super::redact::redact_raw_turn;
use super::schema::open_connection;
use super::store::{
    bootstrap_initial_data, default_agents, default_data_dir, get_config_conn,
    import_legacy_dir_conn, load_agents_conn, load_links_conn, load_nodes_conn, load_profiles_conn,
    load_raw_turn_conn, load_tasks_conn, load_timeline_conn, save_config_conn,
    save_data_dir_pointer, set_session_agent_conn, upsert_agent_conn, upsert_node_conn, with_store,
    StoreInner,
};
use super::types::{now_ts_minute, NeuralAgent, NeuralLink, NeuralNode, RawTurn};

pub fn handle_rpc(action: &str, payload: &Value) -> Result<Value, String> {
    match action {
        "neural:bootstrap" => with_store(|store| build_bootstrap_payload(store)),

        "neural:set-session-agent" => {
            let sid = payload
                .get("sessionId")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let aid = payload
                .get("agentId")
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            with_store(|store| {
                set_session_agent_conn(&store.conn, sid, aid)?;
                Ok(json!({ "success": true, "sessionId": sid, "agentId": aid }))
            })
        }

        "neural:update-config" => with_store(|store| {
            let mut cfg = get_config_conn(&store.conn);
            if let Some(v) = payload.get("activeAgent").and_then(|v| v.as_str()) {
                cfg.active_agent = v.to_string();
            }
            if let Some(v) = payload.get("distillMode").and_then(|v| v.as_str()) {
                cfg.distill_mode = match v.trim() {
                    "auto_silent" | "rule_only" => "rule_only".to_string(),
                    "off" => "off".to_string(),
                    _ => "llm_auto".to_string(),
                };
            }
            if let Some(v) = payload.get("distillModel").and_then(|v| v.as_str()) {
                cfg.distill_model = v.to_string();
            }
            if let Some(v) = payload.get("timelineAutoCapture").and_then(|v| v.as_bool()) {
                cfg.timeline_auto_capture = v;
            }
            if let Some(v) = payload.get("showFloatingCapsule").and_then(|v| v.as_bool()) {
                cfg.show_floating_capsule = v;
            }
            save_config_conn(&store.conn, &cfg)?;
            Ok(json!({ "success": true, "config": cfg }))
        }),

        "neural:save-agents" => {
            let agents_val = payload.get("agents").cloned().unwrap_or(Value::Null);
            let agents: Vec<NeuralAgent> =
                serde_json::from_value(agents_val).map_err(|e| format!("智能体格式错误: {e}"))?;
            with_store(|store| {
                store
                    .conn
                    .execute("DELETE FROM agents", [])
                    .map_err(|e| e.to_string())?;
                for (idx, a) in agents.iter().enumerate() {
                    upsert_agent_conn(&store.conn, a, idx as i64)?;
                }
                Ok(json!({ "success": true, "agents": load_agents_conn(&store.conn)? }))
            })
        }

        "neural:reset-agents" => with_store(|store| {
            store
                .conn
                .execute("DELETE FROM agents", [])
                .map_err(|e| e.to_string())?;
            for (idx, a) in default_agents().iter().enumerate() {
                upsert_agent_conn(&store.conn, a, idx as i64)?;
            }
            Ok(json!({ "success": true, "agents": load_agents_conn(&store.conn)? }))
        }),

        "neural:save-graph" => {
            let mut nodes: Vec<NeuralNode> =
                serde_json::from_value(payload.get("nodes").cloned().unwrap_or_else(|| json!([])))
                    .map_err(|e| format!("节点数据格式错误: {e}"))?;
            let links: Vec<NeuralLink> =
                serde_json::from_value(payload.get("links").cloned().unwrap_or_else(|| json!([])))
                    .map_err(|e| format!("连线数据格式错误: {e}"))?;
            let base_ids: Option<HashSet<String>> = payload
                .get("baseIds")
                .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
                .map(|v| v.into_iter().collect());
            with_store(|store| {
                let existing_by_id: HashMap<String, NeuralNode> =
                    load_nodes_conn(&store.conn, None)?
                        .into_iter()
                        .map(|n| (n.id.clone(), n))
                        .collect();
                for n in &mut nodes {
                    if let Some(old) = existing_by_id.get(&n.id) {
                        if n.project.trim().is_empty() && !old.project.is_empty() {
                            n.project = old.project.clone();
                        }
                        if n.project_key.trim().is_empty() && !old.project_key.is_empty() {
                            n.project_key = old.project_key.clone();
                        }
                        if n.updated_at.trim().is_empty() && !old.updated_at.is_empty() {
                            n.updated_at = old.updated_at.clone();
                        }
                    }
                }
                let incoming_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
                let tx = store
                    .conn
                    .unchecked_transaction()
                    .map_err(|e| e.to_string())?;
                if let Some(ref base) = base_ids {
                    // 仅删除前端快照中存在但本次被用户剪除的节点，绝不误删后台并发提炼出的新节点
                    for old_id in base {
                        if !incoming_ids.contains(old_id) {
                            tx.execute("DELETE FROM nodes WHERE id = ?1", params![old_id])
                                .map_err(|e| e.to_string())?;
                            tx.execute("DELETE FROM nodes_fts WHERE node_id = ?1", params![old_id])
                                .map_err(|e| e.to_string())?;
                            tx.execute(
                                "DELETE FROM links WHERE source = ?1 OR target = ?1",
                                params![old_id],
                            )
                            .map_err(|e| e.to_string())?;
                        }
                    }
                    // 仅清理当前前端视野已知节点之间的旧边，保留后台新节点连线（O(E) 一次扫描）
                    let scope_ids: HashSet<String> = base.union(&incoming_ids).cloned().collect();
                    let existing_links = load_links_conn(&tx)?;
                    for old_l in existing_links {
                        if scope_ids.contains(&old_l.source) && scope_ids.contains(&old_l.target) {
                            tx.execute(
                                "DELETE FROM links WHERE source = ?1 AND target = ?2",
                                params![old_l.source, old_l.target],
                            )
                            .map_err(|e| e.to_string())?;
                        }
                    }
                } else {
                    tx.execute("DELETE FROM nodes", [])
                        .map_err(|e| e.to_string())?;
                    tx.execute("DELETE FROM nodes_fts", [])
                        .map_err(|e| e.to_string())?;
                    tx.execute("DELETE FROM links", [])
                        .map_err(|e| e.to_string())?;
                }
                for n in &nodes {
                    upsert_node_conn(&tx, n)?;
                }
                for l in &links {
                    if l.source == l.target || l.source.is_empty() || l.target.is_empty() {
                        continue;
                    }
                    tx.execute(
                        "INSERT INTO links (source, target)
                         SELECT ?1, ?2
                         WHERE NOT EXISTS (
                             SELECT 1 FROM links
                             WHERE (source = ?1 AND target = ?2) OR (source = ?2 AND target = ?1)
                         )",
                        params![l.source, l.target],
                    )
                    .map_err(|e| e.to_string())?;
                }
                tx.commit().map_err(|e| e.to_string())?;
                Ok(json!({ "success": true }))
            })
        }

        "neural:delete-profile" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let agent = payload.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                store
                    .conn
                    .execute(
                        "DELETE FROM profiles WHERE id = ?1 AND (?2 = '' OR agent_id = ?2)",
                        params![id, agent],
                    )
                    .map_err(|e| e.to_string())?;
                build_bootstrap_payload(store)
            })
        }

        "neural:task-action" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let op = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                if op == "delete" {
                    store
                        .conn
                        .execute("DELETE FROM tasks WHERE id = ?1", params![id])
                        .map_err(|e| e.to_string())?;
                } else if op == "done" {
                    store
                        .conn
                        .execute(
                            "UPDATE tasks SET status = 'done', done_at = ?1, updated_at = ?1, remaining_note = '' WHERE id = ?2",
                            params![now_ts_minute(), id],
                        )
                        .map_err(|e| e.to_string())?;
                }
                build_bootstrap_payload(store)
            })
        }

        "neural:delete-timeline" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let ts = payload.get("ts").and_then(|v| v.as_str()).unwrap_or("");
            let agent = payload.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                if !id.is_empty() {
                    store
                        .conn
                        .execute("DELETE FROM timeline WHERE id = ?1", params![id])
                        .map_err(|e| e.to_string())?;
                    store
                        .conn
                        .execute("DELETE FROM timeline_fts WHERE event_id = ?1", params![id])
                        .ok();
                } else if !ts.is_empty() {
                    store
                        .conn
                        .execute(
                            "DELETE FROM timeline WHERE ts = ?1 AND (?2 = '' OR agent_id = ?2)",
                            params![ts, agent],
                        )
                        .map_err(|e| e.to_string())?;
                }
                build_bootstrap_payload(store)
            })
        }

        "neural:clear-memory" => {
            let agent = payload
                .get("agent")
                .and_then(|v| v.as_str())
                .unwrap_or("all");
            with_store(|store| {
                if agent == "all" || agent.is_empty() {
                    store
                        .conn
                        .execute_batch(
                            "DELETE FROM profiles; DELETE FROM tasks; DELETE FROM nodes; DELETE FROM nodes_fts;
                             DELETE FROM links; DELETE FROM timeline; DELETE FROM timeline_fts;
                             DELETE FROM raw_turns; DELETE FROM raw_turns_fts;",
                        )
                        .map_err(|e| e.to_string())?;
                } else {
                    store
                        .conn
                        .execute("DELETE FROM profiles WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM tasks WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM links WHERE source IN (SELECT id FROM nodes WHERE agent_id = ?1)
                                OR target IN (SELECT id FROM nodes WHERE agent_id = ?1)",
                            params![agent],
                        )
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM nodes WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM nodes_fts WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM timeline WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM timeline_fts WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM raw_turns WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM raw_turns_fts WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                }
                build_bootstrap_payload(store)
            })
        }

        "neural:request-raw" => {
            let turn_id = payload.get("turnId").and_then(|v| v.as_str()).unwrap_or("");
            with_store(|store| {
                let raw = load_raw_turn_conn(&store.conn, turn_id)?;
                Ok(json!({ "success": true, "raw": raw }))
            })
        }

        "neural:import-legacy" => {
            let dir = payload.get("dir").and_then(|v| v.as_str()).unwrap_or("");
            if dir.trim().is_empty() {
                return Err("请提供有效的旧版数据目录路径".into());
            }
            with_store(|store| {
                let stats = import_legacy_dir_conn(&mut store.conn, Path::new(dir.trim()))?;
                let mut boot = build_bootstrap_payload(store)?;
                boot["importStats"] = stats;
                Ok(boot)
            })
        }

        "neural:set-data-dir" => {
            let new_dir_str = payload
                .get("dir")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            let target_dir = if new_dir_str.is_empty() {
                default_data_dir()
            } else {
                PathBuf::from(new_dir_str)
            };
            with_store(|store| {
                if target_dir == store.data_dir {
                    return build_bootstrap_payload(store);
                }
                std::fs::create_dir_all(&target_dir)
                    .map_err(|e| format!("无法创建目标目录: {e}"))?;
                let target_db = target_dir.join("neural_memory.db");
                let old_db = store.data_dir.join("neural_memory.db");

                if !target_db.exists() && old_db.exists() {
                    let _ = store.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                    std::fs::copy(&old_db, &target_db)
                        .map_err(|e| format!("复制数据库到新目录失败: {e}"))?;
                }
                let mut new_conn = open_connection(&target_dir)?;
                bootstrap_initial_data(&mut new_conn, &target_dir);
                let mut cfg = get_config_conn(&new_conn);
                cfg.memory_data_dir = new_dir_str.to_string();
                save_config_conn(&new_conn, &cfg)?;
                save_data_dir_pointer(&target_dir)?;

                store.conn = new_conn;
                store.data_dir = target_dir;
                build_bootstrap_payload(store)
            })
        }

        "neural:export-backup" => with_store(|store| {
            let mut boot = build_bootstrap_payload(store)?;
            let raw_turns: Vec<RawTurn> = {
                let mut stmt = store
                    .conn
                    .prepare(
                        "SELECT turn_id, ts, agent_id, project, project_key, working_dir, session_id, user_text, ai_text, touched_files_json
                         FROM raw_turns ORDER BY ts DESC",
                    )
                    .map_err(|e| e.to_string())?;
                let rows = stmt
                    .query_map([], |row| {
                        let tf_str: String = row.get(9)?;
                        Ok(redact_raw_turn(RawTurn {
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
                        }))
                    })
                    .map_err(|e| e.to_string())?;
                rows.flatten().collect()
            };
            boot["rawTurns"] = serde_json::to_value(raw_turns).unwrap_or_else(|_| json!([]));
            Ok(boot)
        }),

        _ => Err(format!("未知 neural_memory RPC action: {action}")),
    }
}

fn build_bootstrap_payload(store: &StoreInner) -> Result<Value, String> {
    let cfg = get_config_conn(&store.conn);
    let agents = load_agents_conn(&store.conn)?;
    let profiles = load_profiles_conn(&store.conn, None)?;
    let tasks = load_tasks_conn(&store.conn, None)?;
    let nodes = load_nodes_conn(&store.conn, None)?;
    let links = load_links_conn(&store.conn)?;
    let timeline = load_timeline_conn(&store.conn, None, 200)?;

    let mut session_map: HashMap<String, String> = HashMap::new();
    if let Ok(mut stmt) = store
        .conn
        .prepare("SELECT session_id, agent_id FROM session_agents")
    {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        {
            for r in rows.flatten() {
                session_map.insert(r.0, r.1);
            }
        }
    }

    let raw_count: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM raw_turns", [], |r| r.get(0))
        .unwrap_or(0);

    Ok(json!({
        "success": true,
        "dataDir": store.data_dir.to_string_lossy(),
        "dbPath": store.data_dir.join("neural_memory.db").to_string_lossy(),
        "config": cfg,
        "agents": agents,
        "sessionMap": session_map,
        "profiles": profiles,
        "tasks": tasks,
        "nodes": nodes,
        "links": links,
        "timeline": timeline,
        "rawTurnCount": raw_count,
    }))
}
