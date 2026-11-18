//! `zap-agent-teams` RPC 路由分发器（对接 `POST /api/plugins-rpc/zap-agent-teams`，动作前缀 `teams:*`）。

use std::path::{Path, PathBuf};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::AppState;

use super::planner::{convert_triage_to_pipeline, plan_team_run};
use super::runner::{
    cancel_team_run, is_run_executing, pause_team_run, retry_single_task, start_or_resume_run,
};
use super::store::{
    delete_run_conn, list_runs_conn, load_config_conn, load_role_presets_conn,
    load_run_detail_conn, save_config_conn, update_run_draft_cas_conn, upsert_role_preset_conn,
    with_team_store,
};
use super::types::{
    compute_and_assign_waves, new_team_id, normalize_domain, now_ts, RoleBindingPreset,
    TeamGlobalConfig, TeamTask,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveDraftPayload {
    run_id: String,
    #[serde(default)]
    expected_revision: Option<u64>,
    #[serde(default)]
    title: String,
    #[serde(default = "default_true")]
    auto_verify: bool,
    #[serde(default = "default_two")]
    max_repairs: u32,
    #[serde(default)]
    lead_model: String,
    #[serde(default)]
    tasks: Vec<TeamTask>,
}

fn default_true() -> bool {
    true
}

fn default_two() -> u32 {
    2
}

pub async fn handle_rpc(
    state: AppState,
    action: String,
    payload: Value,
) -> axum::response::Response {
    match dispatch_rpc(state, action.as_str(), payload).await {
        Ok(val) => Json(val).into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": err })),
        )
            .into_response(),
    }
}

async fn dispatch_rpc(state: AppState, action: &str, payload: Value) -> Result<Value, String> {
    let current_wd = { state.project.read().await.working_dir.clone() };

    match action {
        "teams:bootstrap" => {
            let wd_str = current_wd.to_string_lossy().replace('\\', "/");
            tokio::task::spawn_blocking(move || {
                let (cfg, presets, runs) = with_team_store(|s| {
                    Ok::<_, String>((
                        load_config_conn(&s.conn),
                        load_role_presets_conn(&s.conn)?,
                        list_runs_conn(&s.conn, 60)?,
                    ))
                })?;

                let agents = crate::neural_memory::list_neural_agents().unwrap_or_default();
                let mut projects = crate::list_projects().unwrap_or_default();
                if !wd_str.is_empty()
                    && !projects
                        .iter()
                        .any(|p| p.working_dir.to_string_lossy().replace('\\', "/") == wd_str)
                {
                    let name = current_wd
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "current".to_string());
                    projects.insert(
                        0,
                        crate::ProjectInfo {
                            hash: crate::hash_path(&current_wd),
                            name,
                            working_dir: current_wd.clone(),
                            description: None,
                            session_count: 0,
                            created_at: 0,
                            last_updated: u64::MAX,
                        },
                    );
                }

                let project_items: Vec<Value> = projects
                    .into_iter()
                    .map(|p| {
                        json!({
                            "hash": p.hash,
                            "name": p.name,
                            "workingDir": p.working_dir.to_string_lossy().replace('\\', "/"),
                            "sessionCount": p.session_count,
                        })
                    })
                    .collect();

                let (default_model, model_items) = match crate::api_config::load_config() {
                    Ok(c) => {
                        let resp = crate::api_config::config_response(&c);
                        let list: Vec<Value> = resp
                            .providers
                            .into_iter()
                            .map(|p| {
                                json!({
                                    "name": p.name,
                                    "model": p.model,
                                    "isDefault": p.is_default,
                                })
                            })
                            .collect();
                        (resp.default_provider, list)
                    }
                    Err(_) => (String::new(), Vec::new()),
                };

                let mut active_count = 0usize;
                let runs_json: Vec<Value> = runs
                    .into_iter()
                    .map(|r| {
                        let live = is_run_executing(&r.id);
                        if live || r.status == "running" {
                            active_count += 1;
                        }
                        let mut v = serde_json::to_value(&r).unwrap_or_else(|_| json!({}));
                        if let Some(obj) = v.as_object_mut() {
                            obj.insert("isExecuting".to_string(), json!(live));
                        }
                        v
                    })
                    .collect();

                Ok(json!({
                    "success": true,
                    "currentWorkingDir": wd_str,
                    "config": cfg,
                    "presets": presets,
                    "agents": agents,
                    "projects": project_items,
                    "defaultModel": default_model,
                    "models": model_items,
                    "runs": runs_json,
                    "activeRunCount": active_count,
                }))
            })
            .await
            .map_err(|e| e.to_string())?
        }

        "teams:list_dirs" => {
            let raw_path = payload
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let base = if raw_path.is_empty() {
                current_wd
            } else {
                PathBuf::from(&raw_path)
            };
            tokio::task::spawn_blocking(move || list_subdirectories(&base))
                .await
                .map_err(|e| e.to_string())?
        }

        "teams:plan" => {
            let goal = payload
                .get("goal")
                .or_else(|| payload.get("userGoal"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mode = payload
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("pipeline");
            let lead_model = payload
                .get("leadModel")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty());

            let detail = plan_team_run(goal, mode, lead_model, &current_wd, None, None).await?;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:triage_to_pipeline" => {
            let triage_id = payload
                .get("triageRunId")
                .or_else(|| payload.get("runId"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let detail = convert_triage_to_pipeline(triage_id, &current_wd).await?;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:get_run" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let after_seq = payload
                .get("afterSeq")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let live = is_run_executing(&run_id);
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| load_run_detail_conn(&s.conn, &run_id, after_seq))
            })
            .await
            .map_err(|e| e.to_string())??
            .ok_or_else(|| "未找到指定的团队派工单".to_string())?;

            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": live,
            }))
        }

        "teams:list_runs" => {
            let runs =
                tokio::task::spawn_blocking(|| with_team_store(|s| list_runs_conn(&s.conn, 80)))
                    .await
                    .map_err(|e| e.to_string())??;
            let mut active_count = 0usize;
            let list: Vec<Value> = runs
                .into_iter()
                .map(|r| {
                    let live = is_run_executing(&r.id);
                    if live || r.status == "running" {
                        active_count += 1;
                    }
                    let mut v = serde_json::to_value(&r).unwrap_or_else(|_| json!({}));
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("isExecuting".to_string(), json!(live));
                    }
                    v
                })
                .collect();
            Ok(json!({
                "success": true,
                "runs": list,
                "activeRunCount": active_count,
            }))
        }

        "teams:save_draft" => {
            let parsed: SaveDraftPayload =
                serde_json::from_value(payload).map_err(|e| format!("参数格式无效: {e}"))?;
            let detail = save_draft_internal(parsed).await?;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": is_run_executing(&detail.run.id),
            }))
        }

        "teams:start" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            if payload.get("tasks").is_some() {
                let parsed: SaveDraftPayload =
                    serde_json::from_value(payload).map_err(|e| format!("草稿参数无效: {e}"))?;
                let _ = save_draft_internal(parsed).await?;
            }
            start_or_resume_run(state, &run_id).await?;
            let rid_q = run_id.clone();
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| load_run_detail_conn(&s.conn, &rid_q, 0))
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": true,
            }))
        }

        "teams:pause" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            pause_team_run(&run_id).await?;
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| load_run_detail_conn(&s.conn, &run_id, 0))
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:cancel" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            cancel_team_run(&run_id).await?;
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| load_run_detail_conn(&s.conn, &run_id, 0))
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:retry_task" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let task_id = payload
                .get("taskId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            retry_single_task(state, &run_id, &task_id).await?;
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| load_run_detail_conn(&s.conn, &run_id, 0))
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": true,
            }))
        }

        "teams:delete_run" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if is_run_executing(&run_id) {
                let _ = cancel_team_run(&run_id).await;
            }
            tokio::task::spawn_blocking(move || {
                with_team_store(|s| delete_run_conn(&mut s.conn, &run_id))
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({ "success": true }))
        }

        "teams:save_config" => {
            let cfg: TeamGlobalConfig =
                serde_json::from_value(payload).map_err(|e| format!("配置格式错误: {e}"))?;
            let saved = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    save_config_conn(&s.conn, &cfg)?;
                    Ok(load_config_conn(&s.conn))
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({ "success": true, "config": saved }))
        }

        "teams:save_preset" => {
            let preset: RoleBindingPreset =
                serde_json::from_value(payload).map_err(|e| format!("预设格式错误: {e}"))?;
            let presets = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    upsert_role_preset_conn(&s.conn, &preset)?;
                    load_role_presets_conn(&s.conn)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({ "success": true, "presets": presets }))
        }

        other => Err(format!("未知的 zap-agent-teams RPC 动作: {other}")),
    }
}

async fn save_draft_internal(
    mut parsed: SaveDraftPayload,
) -> Result<super::types::TeamRunDetail, String> {
    let agents = tokio::task::spawn_blocking(|| {
        crate::neural_memory::list_neural_agents().unwrap_or_default()
    })
    .await
    .unwrap_or_default();

    let now = now_ts();
    for (idx, t) in parsed.tasks.iter_mut().enumerate() {
        if t.id.trim().is_empty() {
            t.id = new_team_id("tsk");
        }
        t.run_id = parsed.run_id.clone();
        t.domain = normalize_domain(&t.domain);
        t.sort_order = idx;
        t.working_dir = t.working_dir.trim().replace('\\', "/");
        t.project_hash = if t.working_dir.is_empty() {
            String::new()
        } else {
            crate::hash_path(Path::new(&t.working_dir))
        };
        t.agent_label = agents
            .iter()
            .find(|a| a.id == t.agent_id)
            .map(|a| a.label.clone())
            .unwrap_or_default();
        if t.updated_at.is_empty() {
            t.updated_at = now.clone();
        }
    }

    let _ = compute_and_assign_waves(&mut parsed.tasks)?;

    tokio::task::spawn_blocking(move || {
        with_team_store(|s| {
            let _ = update_run_draft_cas_conn(
                &mut s.conn,
                &parsed.run_id,
                parsed.expected_revision,
                &parsed.title,
                parsed.auto_verify,
                parsed.max_repairs,
                &parsed.lead_model,
                &parsed.tasks,
            )?;
            for t in &parsed.tasks {
                if !t.working_dir.is_empty() {
                    let _ = upsert_role_preset_conn(
                        &s.conn,
                        &RoleBindingPreset {
                            domain: t.domain.clone(),
                            working_dir: t.working_dir.clone(),
                            agent_id: t.agent_id.clone(),
                            model: t.model.clone(),
                            updated_at: String::new(),
                        },
                    );
                }
            }
            load_run_detail_conn(&s.conn, &parsed.run_id, 0)?
                .ok_or_else(|| "更新后读取派工单失败".to_string())
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

fn list_subdirectories(target: &Path) -> Result<Value, String> {
    let resolved = if target.exists() && target.is_dir() {
        target.to_path_buf()
    } else if let Some(parent) = target.parent().filter(|p| p.exists() && p.is_dir()) {
        parent.to_path_buf()
    } else {
        std::env::current_dir().map_err(|e| e.to_string())?
    };

    let mut dirs = Vec::new();
    let entries = std::fs::read_dir(&resolved)
        .map_err(|e| format!("无法读取目录 {}: {e}", resolved.display()))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with('.') {
                dirs.push(json!({
                    "name": name,
                    "path": p.to_string_lossy().replace('\\', "/"),
                }));
            }
        }
    }
    dirs.sort_by(|a, b| {
        let an = a.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let bn = b.get("name").and_then(|v| v.as_str()).unwrap_or("");
        an.to_ascii_lowercase().cmp(&bn.to_ascii_lowercase())
    });

    // Windows 盘符快捷切换列表
    let mut drives = Vec::new();
    #[cfg(windows)]
    {
        for letter in b'C'..=b'Z' {
            let d = format!("{}:/", letter as char);
            if Path::new(&d).is_dir() {
                drives.push(d);
            }
        }
    }

    let parent = resolved
        .parent()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();

    Ok(json!({
        "success": true,
        "current": resolved.to_string_lossy().replace('\\', "/"),
        "parent": parent,
        "drives": drives,
        "dirs": dirs,
    }))
}
