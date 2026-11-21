//! 前端与插件 RPC 分发器（`/api/plugins-rpc/neural-agents`）。
//!
//! 读写分离优化：
//! - `neural:bootstrap` / `neural:request-raw` / `neural:export-backup` 全部走独立只读连接（`with_read`），零阻塞后台整理写库；
//! - `neural:set-session-agent` 在切换智能体前自动刷出上一任智能体的未整理草稿；
//! - 新增 `neural:flush-drafts`（手动立刻整理草稿）、`neural:consolidate-legacy`（老记忆一键体检去重与重索引）、`neural:node-fact-action`（卡片单条要点作废/恢复/删除）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use serde_json::{json, Value};

use super::distill::{
    flush_all_pending_drafts_async, flush_session_drafts_async, spawn_boot_draft_recovery_once,
};
use super::reconcile::consolidate_legacy_nodes_conn;
use super::redact::redact_raw_turn;
use super::schema::open_connection;
use super::store::{
    bootstrap_initial_data, count_pending_drafts_conn, default_agents, default_data_dir,
    delete_node_with_indexes_conn, delete_profile_with_indexes_conn, delete_task_with_indexes_conn,
    get_config_conn, get_explicit_session_agent_conn, import_legacy_dir_conn,
    invalidate_all_hot_caches, invalidate_hot_agents, invalidate_hot_content, load_agents_conn,
    load_links_conn, load_nodes_conn, load_profiles_conn, load_raw_turn_conn, load_tasks_conn,
    load_timeline_conn, notify_store_dir_switched, save_config_conn, save_data_dir_pointer,
    set_session_agent_conn, upsert_agent_conn, upsert_link_conn, upsert_node_conn, with_read,
    with_write,
};
use super::team_bridge::lazy_sync_team_deliveries_if_changed;
use super::types::{now_ts_minute, NeuralAgent, NeuralLink, NeuralNode, RawTurn};

pub fn handle_rpc(action: &str, payload: &Value) -> Result<Value, String> {
    match action {
        "neural:bootstrap" => {
            spawn_boot_draft_recovery_once();
            let _ = lazy_sync_team_deliveries_if_changed();
            with_read(|conn, data_dir| build_bootstrap_payload_conn(conn, data_dir))
        }

        "neural:set-session-agent" => {
            let sid = payload
                .get("sessionId")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let aid = payload
                .get("agentId")
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            let prev_agent = with_write(|store| {
                let prev = get_explicit_session_agent_conn(&store.conn, sid);
                set_session_agent_conn(&store.conn, sid, aid)?;
                Ok(prev)
            })?;
            // 切换智能体时，自动把上一任智能体在本会话攒的未整理草稿立刻刷盘抄正本
            if let Some(prev) = prev_agent {
                if !prev.is_empty() && prev != "none" && prev != aid.trim() {
                    flush_session_drafts_async(sid, Some(&prev));
                }
            }
            Ok(json!({ "success": true, "sessionId": sid, "agentId": aid }))
        }

        "neural:flush-drafts" => {
            let agent = payload
                .get("agent")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let _ = lazy_sync_team_deliveries_if_changed();
            let flush_res = if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.block_on(flush_all_pending_drafts_async(agent.as_deref()))?
            } else {
                json!({ "success": true, "flushedBatches": 0, "flushedTurns": 0 })
            };
            let mut boot =
                with_read(|conn, data_dir| build_bootstrap_payload_conn(conn, data_dir))?;
            boot["flushResult"] = flush_res;
            Ok(boot)
        }

        "neural:consolidate-legacy" => {
            let agent = payload
                .get("agent")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != "all");
            let (merged, remaining) =
                with_write(|store| consolidate_legacy_nodes_conn(&mut store.conn, agent))?;
            invalidate_hot_content(agent);
            let mut boot =
                with_read(|conn, data_dir| build_bootstrap_payload_conn(conn, data_dir))?;
            boot["consolidateStats"] = json!({
                "mergedNodes": merged,
                "remainingNodes": remaining,
            });
            Ok(boot)
        }

        "neural:node-fact-action" => {
            let node_id = payload.get("nodeId").and_then(|v| v.as_str()).unwrap_or("");
            let fact_id = payload.get("factId").and_then(|v| v.as_str()).unwrap_or("");
            let op = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let new_text = payload.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if node_id.trim().is_empty() || fact_id.trim().is_empty() {
                return Err("缺少 nodeId 或 factId".into());
            }
            with_write(|store| {
                let all = load_nodes_conn(&store.conn, None)?;
                let Some(mut node) = all.into_iter().find(|n| n.id == node_id) else {
                    return Err("未找到目标知识卡片".into());
                };
                let now = now_ts_minute();
                match op {
                    "delete" => node.facts.retain(|f| f.id != fact_id),
                    "supersede" => {
                        if let Some(f) = node.facts.iter_mut().find(|f| f.id == fact_id) {
                            f.status = "superseded".into();
                            f.invalid_at = now.clone();
                        }
                    }
                    "restore" => {
                        if let Some(f) = node.facts.iter_mut().find(|f| f.id == fact_id) {
                            f.status = "active".into();
                            f.invalid_at.clear();
                            f.superseded_by.clear();
                            f.updated_at = now.clone();
                        }
                    }
                    "edit" => {
                        if !new_text.trim().is_empty() {
                            if let Some(f) = node.facts.iter_mut().find(|f| f.id == fact_id) {
                                f.text = new_text.trim().to_string();
                                f.updated_at = now.clone();
                            }
                        }
                    }
                    _ => return Err(format!("未知 node-fact-action: {op}")),
                }
                let active_texts: Vec<String> = node
                    .facts
                    .iter()
                    .filter(|f| f.is_active())
                    .take(4)
                    .map(|f| f.text.clone())
                    .collect();
                if !active_texts.is_empty() {
                    node.summary = active_texts.join("；");
                }
                node.updated_at = now;
                upsert_node_conn(&store.conn, &node)?;
                build_bootstrap_payload_conn(&store.conn, &store.data_dir)
            })
        }

        "neural:update-config" => with_write(|store| {
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
            with_write(|store| {
                store
                    .conn
                    .execute("DELETE FROM agents", [])
                    .map_err(|e| e.to_string())?;
                for (idx, a) in agents.iter().enumerate() {
                    upsert_agent_conn(&store.conn, a, idx as i64)?;
                }
                invalidate_hot_agents();
                Ok(json!({ "success": true, "agents": load_agents_conn(&store.conn)? }))
            })
        }

        "neural:reset-agents" => with_write(|store| {
            store
                .conn
                .execute("DELETE FROM agents", [])
                .map_err(|e| e.to_string())?;
            for (idx, a) in default_agents().iter().enumerate() {
                upsert_agent_conn(&store.conn, a, idx as i64)?;
            }
            invalidate_hot_agents();
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
            with_write(|store| {
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
                        // 兼容旧版脑图保存时不传 v3 扩展字段，自动保留原节点的 aliases / search_tags / facts / hit_count
                        if n.aliases.is_empty() && !old.aliases.is_empty() {
                            n.aliases = old.aliases.clone();
                        }
                        if n.search_tags.is_empty() && !old.search_tags.is_empty() {
                            n.search_tags = old.search_tags.clone();
                        }
                        if n.facts.is_empty() && !old.facts.is_empty() {
                            n.facts = old.facts.clone();
                        }
                        if n.hit_count == 0 && old.hit_count > 0 {
                            n.hit_count = old.hit_count;
                        }
                        if n.last_used_at.is_empty() && !old.last_used_at.is_empty() {
                            n.last_used_at = old.last_used_at.clone();
                        }
                    }
                }
                let incoming_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
                let tx = store
                    .conn
                    .unchecked_transaction()
                    .map_err(|e| e.to_string())?;
                if let Some(ref base) = base_ids {
                    for old_id in base {
                        if !incoming_ids.contains(old_id) {
                            delete_node_with_indexes_conn(&tx, old_id)?;
                        }
                    }
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
                    tx.execute("DELETE FROM memory_anchors WHERE target_kind = 'node'", [])
                        .map_err(|e| e.to_string())?;
                    tx.execute("DELETE FROM links", [])
                        .map_err(|e| e.to_string())?;
                }
                for n in &nodes {
                    upsert_node_conn(&tx, n)?;
                }
                for l in &links {
                    upsert_link_conn(&tx, &l.source, &l.target, &l.relation);
                }
                tx.commit().map_err(|e| e.to_string())?;
                invalidate_hot_content(None);
                Ok(json!({ "success": true }))
            })
        }

        "neural:delete-profile" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let agent = payload.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            with_write(|store| {
                delete_profile_with_indexes_conn(&store.conn, id, agent)?;
                build_bootstrap_payload_conn(&store.conn, &store.data_dir)
            })
        }

        "neural:task-action" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let op = payload.get("action").and_then(|v| v.as_str()).unwrap_or("");
            with_write(|store| {
                if op == "delete" {
                    delete_task_with_indexes_conn(&store.conn, id)?;
                } else if op == "done" {
                    store
                        .conn
                        .execute(
                            "UPDATE tasks SET status = 'done', done_at = ?1, updated_at = ?1, remaining_note = '', next_step = '' WHERE id = ?2",
                            params![now_ts_minute(), id],
                        )
                        .map_err(|e| e.to_string())?;
                    invalidate_hot_content(None);
                }
                build_bootstrap_payload_conn(&store.conn, &store.data_dir)
            })
        }

        "neural:delete-timeline" => {
            let id = payload.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let ts = payload.get("ts").and_then(|v| v.as_str()).unwrap_or("");
            let agent = payload.get("agent").and_then(|v| v.as_str()).unwrap_or("");
            with_write(|store| {
                if !id.is_empty() {
                    if let Ok(rid) = store.conn.query_row(
                        "SELECT rowid FROM timeline WHERE id = ?1",
                        params![id],
                        |r| r.get::<_, i64>(0),
                    ) {
                        let _ = store
                            .conn
                            .execute("DELETE FROM timeline_fts WHERE rowid = ?1", params![rid]);
                    }
                    let _ = store.conn.execute(
                        "DELETE FROM memory_anchors WHERE target_kind = 'timeline' AND target_id = ?1",
                        params![id],
                    );
                    store
                        .conn
                        .execute("DELETE FROM timeline WHERE id = ?1", params![id])
                        .map_err(|e| e.to_string())?;
                } else if !ts.is_empty() {
                    let targets: Vec<(i64, String)> = store
                        .conn
                        .prepare("SELECT rowid, id FROM timeline WHERE ts = ?1 AND (?2 = '' OR agent_id = ?2)")
                        .ok()
                        .and_then(|mut s| {
                            s.query_map(params![ts, agent], |r| Ok((r.get(0)?, r.get(1)?)))
                                .ok()
                                .map(|rows| rows.flatten().collect())
                        })
                        .unwrap_or_default();
                    for (rid, tid) in targets {
                        let _ = store
                            .conn
                            .execute("DELETE FROM timeline_fts WHERE rowid = ?1", params![rid]);
                        let _ = store.conn.execute(
                            "DELETE FROM memory_anchors WHERE target_kind = 'timeline' AND target_id = ?1",
                            params![tid],
                        );
                        let _ = store
                            .conn
                            .execute("DELETE FROM timeline WHERE id = ?1", params![tid]);
                    }
                }
                invalidate_hot_content(None);
                build_bootstrap_payload_conn(&store.conn, &store.data_dir)
            })
        }

        "neural:clear-memory" => {
            let agent = payload
                .get("agent")
                .and_then(|v| v.as_str())
                .unwrap_or("all");
            with_write(|store| {
                if agent == "all" || agent.is_empty() {
                    store
                        .conn
                        .execute_batch(
                            "DELETE FROM profiles; DELETE FROM profiles_fts;
                             DELETE FROM tasks; DELETE FROM tasks_fts;
                             DELETE FROM nodes; DELETE FROM nodes_fts;
                             DELETE FROM links;
                             DELETE FROM timeline; DELETE FROM timeline_fts;
                             DELETE FROM raw_turns;
                             DELETE FROM memory_anchors;
                             DELETE FROM draft_turns;
                             DELETE FROM team_sync_log;",
                        )
                        .map_err(|e| e.to_string())?;
                    invalidate_all_hot_caches();
                } else {
                    store
                        .conn
                        .execute("DELETE FROM profiles_fts WHERE rowid IN (SELECT rowid FROM profiles WHERE agent_id = ?1)", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM profiles WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM tasks_fts WHERE rowid IN (SELECT rowid FROM tasks WHERE agent_id = ?1)", params![agent])
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
                        .execute("DELETE FROM nodes_fts WHERE rowid IN (SELECT rowid FROM nodes WHERE agent_id = ?1)", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM nodes WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM timeline_fts WHERE rowid IN (SELECT rowid FROM timeline WHERE agent_id = ?1)", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM timeline WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute("DELETE FROM raw_turns WHERE agent_id = ?1", params![agent])
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM memory_anchors WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM draft_turns WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                    store
                        .conn
                        .execute(
                            "DELETE FROM team_sync_log WHERE agent_id = ?1",
                            params![agent],
                        )
                        .ok();
                    invalidate_hot_content(Some(agent));
                }
                build_bootstrap_payload_conn(&store.conn, &store.data_dir)
            })
        }

        "neural:request-raw" => {
            let turn_id = payload.get("turnId").and_then(|v| v.as_str()).unwrap_or("");
            with_read(|conn, _| {
                let raw = load_raw_turn_conn(conn, turn_id)?;
                Ok(json!({ "success": true, "raw": raw }))
            })
        }

        "neural:import-legacy" => {
            let dir = payload.get("dir").and_then(|v| v.as_str()).unwrap_or("");
            if dir.trim().is_empty() {
                return Err("请提供有效的旧版数据目录路径".into());
            }
            with_write(|store| {
                let stats = import_legacy_dir_conn(&mut store.conn, Path::new(dir.trim()))?;
                invalidate_hot_content(None);
                let mut boot = build_bootstrap_payload_conn(&store.conn, &store.data_dir)?;
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
            let switched_dir = with_write(|store| {
                if target_dir == store.data_dir {
                    return Ok(None);
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
                store.data_dir = target_dir.clone();
                Ok(Some(target_dir))
            })?;
            if let Some(dir) = switched_dir {
                notify_store_dir_switched(&dir);
            }
            with_read(|conn, data_dir| build_bootstrap_payload_conn(conn, data_dir))
        }

        "neural:export-backup" => with_read(|conn, data_dir| {
            let mut boot = build_bootstrap_payload_conn(conn, data_dir)?;
            let raw_turns: Vec<RawTurn> = {
                let mut stmt = conn
                    .prepare_cached(
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

fn build_bootstrap_payload_conn(conn: &Connection, data_dir: &Path) -> Result<Value, String> {
    let cfg = get_config_conn(conn);
    let agents = load_agents_conn(conn)?;
    let profiles = load_profiles_conn(conn, None)?;
    let tasks = load_tasks_conn(conn, None)?;
    let nodes = load_nodes_conn(conn, None)?;
    let links = load_links_conn(conn)?;
    let timeline = load_timeline_conn(conn, None, 200)?;

    let mut session_map: HashMap<String, String> = HashMap::new();
    if let Ok(mut stmt) = conn.prepare_cached("SELECT session_id, agent_id FROM session_agents") {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        {
            for r in rows.flatten() {
                session_map.insert(r.0, r.1);
            }
        }
    }

    let raw_count: i64 = conn
        .prepare_cached("SELECT COUNT(*) FROM raw_turns")
        .ok()
        .and_then(|mut s| s.query_row([], |r| r.get(0)).ok())
        .unwrap_or(0);
    let pending_draft_count = count_pending_drafts_conn(conn, None);

    Ok(json!({
        "success": true,
        "dataDir": data_dir.to_string_lossy(),
        "dbPath": data_dir.join("neural_memory.db").to_string_lossy(),
        "config": cfg,
        "agents": agents,
        "sessionMap": session_map,
        "profiles": profiles,
        "tasks": tasks,
        "nodes": nodes,
        "links": links,
        "timeline": timeline,
        "rawTurnCount": raw_count,
        "pendingDraftCount": pending_draft_count,
    }))
}
