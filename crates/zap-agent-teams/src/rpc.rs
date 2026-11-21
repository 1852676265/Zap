//! `zap-agent-teams` RPC 路由分发器（对接 `POST /api/plugins-rpc/zap-agent-teams`，动作前缀 `teams:*`）。

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::prompts::{get_prompt_meta, list_prompt_metas, preview_role_prompt};
use super::store::RunDraftExtras;
use super::types::TeamRun;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::host::{HostProjectInfo, TeamHost};

use super::planner::{convert_triage_to_pipeline, create_pending_pipeline_run, plan_team_run};
use super::runner::{
    cancel_team_run, is_run_executing, pause_team_run, retry_single_task, revise_design_phase,
    send_member_note, start_design_phase, start_or_resume_run, start_sql_design_phase,
};
use super::store::{
    append_event_conn, confirm_sql_to_team_binding_conn, delete_custom_project_conn,
    delete_role_preset_conn, delete_run_conn, list_runs_conn, load_config_conn,
    load_custom_projects_conn, load_role_presets_conn, load_run_conn, load_run_detail_conn,
    load_task_statuses_by_run_conn, reset_run_to_team_binding_conn, save_config_conn,
    transition_run_stage_conn, update_run_design_texts_conn, update_run_draft_cas_conn,
    upsert_custom_project_conn, upsert_role_preset_conn, with_team_store,
};
use super::types::{
    clean_path_str, compute_and_assign_waves, new_team_id, normalize_domain, now_ts,
    path_to_clean_str, RoleBindingPreset, TeamGlobalConfig, TeamImage, TeamProjectPreset, TeamTask,
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
    requires_db: Option<bool>,
    #[serde(default)]
    execution_plan: Option<String>,
    #[serde(default)]
    sql_schema_draft: Option<String>,
    #[serde(default)]
    tasks: Vec<TeamTask>,
    #[serde(default)]
    sql_working_dir: Option<String>,
    #[serde(default)]
    sql_agent_id: Option<String>,
    #[serde(default)]
    sql_model: Option<String>,
    #[serde(default)]
    prompt_overrides: Option<BTreeMap<String, String>>,
}

fn default_true() -> bool {
    true
}

fn default_two() -> u32 {
    2
}

pub async fn handle_rpc(
    host: Arc<dyn TeamHost>,
    action: &str,
    payload: Value,
) -> Result<Value, String> {
    dispatch_rpc(host, action, payload).await
}

async fn dispatch_rpc(
    host: Arc<dyn TeamHost>,
    action: &str,
    payload: Value,
) -> Result<Value, String> {
    let current_wd = host.current_working_dir().await;

    match action {
        "teams:bootstrap" => {
            let wd_str = path_to_clean_str(&current_wd);
            let host_for_init = Arc::clone(&host);
            let wd_str_spawn = wd_str.clone();
            let current_wd_spawn = current_wd.clone();
            tokio::task::spawn_blocking(move || {
                let (cfg, presets, custom_projects, runs, status_map) = with_team_store(|s| {
                    let runs = list_runs_conn(&s.conn, 60)?;
                    let run_ids: Vec<String> = runs.iter().map(|r| r.id.clone()).collect();
                    let status_map = load_task_statuses_by_run_conn(&s.conn, &run_ids)?;
                    Ok::<_, String>((
                        load_config_conn(&s.conn),
                        load_role_presets_conn(&s.conn)?,
                        load_custom_projects_conn(&s.conn)?,
                        runs,
                        status_map,
                    ))
                })?;

                let agents = zap_neural_memory::list_neural_agents().unwrap_or_default();
                let mut projects = host_for_init.list_projects();
                if !wd_str_spawn.is_empty()
                    && !projects
                        .iter()
                        .any(|p| path_to_clean_str(&p.working_dir) == wd_str_spawn)
                {
                    let name = current_wd_spawn
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "current".to_string());
                    projects.insert(
                        0,
                        HostProjectInfo {
                            hash: host_for_init.hash_path(&current_wd_spawn),
                            name,
                            working_dir: current_wd_spawn.clone(),
                            session_count: 0,
                        },
                    );
                }

                let project_items: Vec<Value> = projects
                    .into_iter()
                    .map(|p| {
                        json!({
                            "hash": p.hash,
                            "name": p.name,
                            "workingDir": path_to_clean_str(&p.working_dir),
                            "sessionCount": p.session_count,
                        })
                    })
                    .collect();

                let (default_model, raw_models) = host_for_init.list_models();
                let model_items: Vec<Value> = raw_models
                    .into_iter()
                    .map(|p| {
                        json!({
                            "name": p.name,
                            "model": p.model,
                            "isDefault": p.is_default,
                        })
                    })
                    .collect();

                let (runs_json, active_count) = enrich_runs_json(runs, &status_map);

                Ok(json!({
                    "success": true,
                    "currentWorkingDir": wd_str,
                    "config": cfg,
                    "presets": presets,
                    "customProjects": custom_projects,
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
            let images: Vec<TeamImage> = payload
                .get("images")
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();

            let overrides = parse_prompt_map(&payload, "promptOverrides")?.unwrap_or_default();
            if mode.trim().eq_ignore_ascii_case("triage") {
                let detail = plan_team_run(
                    &host,
                    goal,
                    mode,
                    lead_model,
                    &current_wd,
                    None,
                    None,
                    images,
                    overrides,
                )
                .await?;
                Ok(json!({
                    "success": true,
                    "detail": detail,
                    "isExecuting": false,
                }))
            } else {
                let detail = create_pending_pipeline_run(
                    host.as_ref(),
                    goal,
                    lead_model,
                    None,
                    images,
                    overrides,
                )
                .await?;
                start_design_phase(host, detail.run.id.clone(), current_wd).await;
                Ok(json!({
                    "success": true,
                    "detail": detail,
                    "isExecuting": true,
                }))
            }
        }

        "teams:revise_design" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let feedback = payload
                .get("feedback")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            save_design_payload(host.as_ref(), &payload, &run_id).await?;
            revise_design_phase(host, run_id.clone(), feedback, current_wd).await?;
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

        "teams:triage_to_pipeline" => {
            let triage_id = payload
                .get("triageRunId")
                .or_else(|| payload.get("runId"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let detail = convert_triage_to_pipeline(host.as_ref(), triage_id).await?;
            start_design_phase(host, detail.run.id.clone(), current_wd).await;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": true,
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
            .ok_or_else(|| "未找到指定的团队任务".to_string())?;

            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": live,
            }))
        }

        "teams:list_runs" => {
            let (runs, status_map) = tokio::task::spawn_blocking(|| {
                with_team_store(|s| {
                    let runs = list_runs_conn(&s.conn, 80)?;
                    let run_ids: Vec<String> = runs.iter().map(|r| r.id.clone()).collect();
                    let status_map = load_task_statuses_by_run_conn(&s.conn, &run_ids)?;
                    Ok::<_, String>((runs, status_map))
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            let (list, active_count) = enrich_runs_json(runs, &status_map);
            Ok(json!({
                "success": true,
                "runs": list,
                "activeRunCount": active_count,
            }))
        }

        "teams:save_draft" => {
            let parsed: SaveDraftPayload =
                serde_json::from_value(payload).map_err(|e| format!("参数格式无效: {e}"))?;
            let detail = save_draft_internal(host.as_ref(), parsed).await?;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": is_run_executing(&detail.run.id),
            }))
        }

        "teams:start_sql_design" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            let sql_override = {
                let wd = optional_string(&payload, "sqlWorkingDir")?
                    .map(|v| clean_path_str(&v))
                    .unwrap_or_default();
                let aid = optional_string(&payload, "sqlAgentId")?.unwrap_or_default();
                let mdl = optional_string(&payload, "sqlModel")?.unwrap_or_default();
                if !wd.is_empty() || !aid.is_empty() || !mdl.is_empty() {
                    Some((wd, aid, mdl))
                } else {
                    None
                }
            };
            save_design_payload(host.as_ref(), &payload, &run_id).await?;
            start_sql_design_phase(host, &run_id, current_wd, sql_override).await?;
            let rid_q = run_id.clone();
            let (detail, presets) = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    Ok::<_, String>((
                        load_run_detail_conn(&s.conn, &rid_q, 0)?,
                        load_role_presets_conn(&s.conn)?,
                    ))
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "presets": presets,
                "isExecuting": true,
            }))
        }

        "teams:confirm_sql" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            if is_run_executing(&run_id) {
                return Err("当前任务正在生成或执行中，请稍候".to_string());
            }
            let ep = payload
                .get("executionPlan")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let sql_draft = payload
                .get("sqlSchemaDraft")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    confirm_sql_to_team_binding_conn(
                        &s.conn,
                        &run_id,
                        ep.as_deref(),
                        sql_draft.as_deref(),
                    )?;
                    let _ = append_event_conn(
                        &s.conn,
                        &run_id,
                        "sql_confirmed",
                        "",
                        "user",
                        "用户已确认数据库表结构定稿，进入开发智能体与项目目录配置阶段。",
                        "",
                    );
                    load_run_detail_conn(&s.conn, &run_id, 0)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:skip_sql" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            if is_run_executing(&run_id) {
                return Err("当前任务正在生成或执行中，请稍候".to_string());
            }
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    transition_run_stage_conn(
                        &s.conn,
                        &run_id,
                        "draft",
                        "team_binding",
                        Some(false),
                    )?;
                    let _ = append_event_conn(
                        &s.conn,
                        &run_id,
                        "sql_skipped",
                        "",
                        "user",
                        "用户选择本次不新建/修改表结构，直接进入研发团队配置阶段。",
                        "",
                    );
                    load_run_detail_conn(&s.conn, &run_id, 0)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:back_to_sql" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            if is_run_executing(&run_id) {
                return Err("任务正在执行中，请先暂停后再调整 SQL".to_string());
            }
            let target_req = payload
                .get("targetStage")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    let cur = load_run_conn(&s.conn, &run_id)?
                        .ok_or_else(|| format!("未找到任务: {run_id}"))?;
                    let target = match target_req.as_str() {
                        "sql_binding" => "sql_binding",
                        "awaiting_approval" => "awaiting_approval",
                        _ => {
                            if cur.sql_schema_draft.trim().is_empty() {
                                "sql_binding"
                            } else {
                                "awaiting_approval"
                            }
                        }
                    };
                    transition_run_stage_conn(&s.conn, &run_id, "draft", target, Some(true))?;
                    load_run_detail_conn(&s.conn, &run_id, 0)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:start" | "teams:approve_and_start" => {
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
                    serde_json::from_value(payload).map_err(|e| format!("任务参数无效: {e}"))?;
                let _ = save_draft_internal(host.as_ref(), parsed).await?;
            }
            start_or_resume_run(host, &run_id).await?;
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
            retry_single_task(host, &run_id, &task_id).await?;
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

        "teams:list_prompts" => {
            let run_id = optional_string(&payload, "runId")?;
            let result = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    let overrides = match run_id {
                        Some(id) => {
                            load_run_conn(&s.conn, &id)?
                                .ok_or_else(|| format!("未找到任务: {id}"))?
                                .prompt_overrides
                        }
                        None => BTreeMap::new(),
                    };
                    Ok(json!({"success": true, "templates": list_prompt_metas(),
                    "globalTemplates": load_config_conn(&s.conn).prompt_templates,
                    "runOverrides": overrides}))
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(result)
        }

        "teams:preview_prompt" => {
            let template_id = optional_string(&payload, "templateId")?
                .ok_or_else(|| "缺少 templateId".to_string())?;
            validate_template_id(&template_id)?;
            let run_id = optional_string(&payload, "runId")?;
            let task_id = optional_string(&payload, "taskId")?;
            let draft_template = optional_string(&payload, "draftTemplate")?;
            let working_dir = optional_string(&payload, "workingDir")?;
            let feedback = optional_string(&payload, "feedback")?;
            let run_draft: Option<TeamRun> = payload
                .get("runDraft")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| format!("任务草稿格式无效: {e}"))?;
            let tasks_draft: Option<Vec<TeamTask>> = payload
                .get("tasksDraft")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| format!("成员草稿格式无效: {e}"))?;
            let wd = path_to_clean_str(&current_wd);
            let preview = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    let effective_id = run_id
                        .as_deref()
                        .or_else(|| run_draft.as_ref().map(|r| r.id.as_str()))
                        .filter(|id| !id.trim().is_empty());
                    let stored = match effective_id {
                        Some(id) => Some(
                            load_run_detail_conn(&s.conn, id, 0)?
                                .ok_or_else(|| format!("未找到任务: {id}"))?,
                        ),
                        None => None,
                    };
                    if let (Some(id), Some(draft)) = (run_id.as_deref(), run_draft.as_ref()) {
                        if id != draft.id {
                            return Err("任务草稿与 runId 不匹配".to_string());
                        }
                    }
                    let run = run_draft
                        .as_ref()
                        .or_else(|| stored.as_ref().map(|d| &d.run));
                    if let Some(r) = run {
                        validate_prompt_map(&r.prompt_overrides)?;
                    }
                    let tasks = tasks_draft
                        .as_deref()
                        .or_else(|| stored.as_ref().map(|d| d.tasks.as_slice()))
                        .unwrap_or(&[]);
                    let mut ids = HashSet::new();
                    for task in tasks {
                        if run.map(|r| r.id.as_str()) != Some(task.run_id.as_str())
                            || !ids.insert(&task.id)
                        {
                            return Err("成员草稿与任务不匹配或成员 ID 重复".to_string());
                        }
                    }
                    let target = match task_id.as_deref() {
                        Some(id) => Some(
                            tasks
                                .iter()
                                .find(|t| t.id == id)
                                .ok_or_else(|| format!("未找到本任务成员: {id}"))?,
                        ),
                        None => None,
                    };
                    let (base_id, domain) = template_id
                        .split_once(':')
                        .map(|(a, b)| (a, Some(b)))
                        .unwrap_or((template_id.as_str(), None));
                    let globals = load_config_conn(&s.conn).prompt_templates;
                    preview_role_prompt(
                        base_id,
                        domain,
                        run,
                        tasks,
                        target,
                        stored
                            .as_ref()
                            .map(|d| d.contracts.as_slice())
                            .unwrap_or(&[]),
                        &load_custom_projects_conn(&s.conn)?,
                        &globals,
                        draft_template.as_deref(),
                        working_dir.as_deref(),
                        feedback.as_deref(),
                        &wd,
                    )
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({"success": true, "preview": preview}))
        }

        "teams:save_config" | "teams:save_prompt_templates" => {
            let templates = parse_prompt_map(&payload, "promptTemplates")?;
            if action == "teams:save_prompt_templates" && templates.is_none() {
                return Err("缺少 promptTemplates".to_string());
            }
            let mut cfg: TeamGlobalConfig =
                serde_json::from_value(payload).map_err(|e| format!("配置格式错误: {e}"))?;
            let templates_only = action == "teams:save_prompt_templates";
            let saved = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    let old = load_config_conn(&s.conn);
                    if templates_only {
                        cfg = old;
                    } else if templates.is_none() {
                        cfg.prompt_templates = old.prompt_templates;
                    }
                    if let Some(templates) = templates {
                        cfg.prompt_templates = templates;
                    }
                    save_config_conn(&s.conn, &cfg)?;
                    Ok(load_config_conn(&s.conn))
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({"success": true, "config": saved}))
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

        "teams:delete_preset" | "teams:delete_role_preset" => {
            let domain = payload
                .get("domain")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if domain.is_empty() {
                return Err("缺少预设角色 domain".to_string());
            }
            let presets = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    delete_role_preset_conn(&s.conn, &domain)?;
                    load_role_presets_conn(&s.conn)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({ "success": true, "presets": presets }))
        }

        "teams:back_to_binding" => {
            let run_id = payload
                .get("runId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if run_id.is_empty() {
                return Err("缺少 runId".to_string());
            }
            if is_run_executing(&run_id) {
                return Err("任务正在执行中，请先暂停后再退回调整".to_string());
            }
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    reset_run_to_team_binding_conn(&s.conn, &run_id)?;
                    load_run_detail_conn(&s.conn, &run_id, 0)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "detail": detail,
                "isExecuting": false,
            }))
        }

        "teams:save_project" => {
            let proj: TeamProjectPreset =
                serde_json::from_value(payload).map_err(|e| format!("项目参数错误: {e}"))?;
            let (saved, custom_projects) = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    let saved = upsert_custom_project_conn(&s.conn, &proj)?;
                    let all = load_custom_projects_conn(&s.conn)?;
                    Ok::<_, String>((saved, all))
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({ "success": true, "project": saved, "customProjects": custom_projects }))
        }

        "teams:delete_project" => {
            let id = payload
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if id.is_empty() {
                return Err("缺少项目 ID".to_string());
            }
            let custom_projects = tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    delete_custom_project_conn(&s.conn, &id)?;
                    load_custom_projects_conn(&s.conn)
                })
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({ "success": true, "customProjects": custom_projects }))
        }

        "teams:send_member_note" => {
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
            let note = payload
                .get("note")
                .or_else(|| payload.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let (mode, live) = send_member_note(host, &run_id, &task_id, &note).await?;
            let detail = tokio::task::spawn_blocking(move || {
                with_team_store(|s| load_run_detail_conn(&s.conn, &run_id, 0))
            })
            .await
            .map_err(|e| e.to_string())??;
            Ok(json!({
                "success": true,
                "mode": mode,
                "detail": detail,
                "isExecuting": live,
            }))
        }

        other => Err(format!("未知的 zap-agent-teams RPC 动作: {other}")),
    }
}

fn enrich_runs_json(
    runs: Vec<super::types::TeamRun>,
    status_map: &std::collections::HashMap<String, Vec<String>>,
) -> (Vec<Value>, usize) {
    let mut active_count = 0usize;
    let list = runs
        .into_iter()
        .map(|r| {
            let live = is_run_executing(&r.id);
            if live || r.status == "running" {
                active_count += 1;
            }
            let statuses = status_map.get(&r.id).cloned().unwrap_or_default();
            let total = statuses.len();
            let done = statuses
                .iter()
                .filter(|s| s.as_str() == "completed")
                .count();
            let mut v = serde_json::to_value(&r).unwrap_or_else(|_| json!({}));
            if let Some(obj) = v.as_object_mut() {
                obj.insert("isExecuting".to_string(), json!(live));
                obj.insert("taskStatuses".to_string(), json!(statuses));
                obj.insert("taskCount".to_string(), json!(total));
                obj.insert("completedTaskCount".to_string(), json!(done));
            }
            v
        })
        .collect();
    (list, active_count)
}

async fn save_draft_internal(
    host: &dyn TeamHost,
    mut parsed: SaveDraftPayload,
) -> Result<super::types::TeamRunDetail, String> {
    if let Some(ref overrides) = parsed.prompt_overrides {
        validate_prompt_map(overrides)?;
    }
    let agents =
        tokio::task::spawn_blocking(|| zap_neural_memory::list_neural_agents().unwrap_or_default())
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
        t.working_dir = clean_path_str(&t.working_dir);
        t.project_hash = if t.working_dir.is_empty() {
            String::new()
        } else {
            host.hash_path(Path::new(&t.working_dir))
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
                parsed.requires_db,
                &parsed.tasks,
                &RunDraftExtras {
                    sql_working_dir: parsed.sql_working_dir.as_deref(),
                    sql_agent_id: parsed.sql_agent_id.as_deref(),
                    sql_model: parsed.sql_model.as_deref(),
                    prompt_overrides: parsed.prompt_overrides.as_ref(),
                },
            )?;
            if parsed.execution_plan.is_some() || parsed.sql_schema_draft.is_some() {
                let _ = update_run_design_texts_conn(
                    &s.conn,
                    &parsed.run_id,
                    parsed.execution_plan.as_deref(),
                    parsed.sql_schema_draft.as_deref(),
                )?;
            }
            for t in &parsed.tasks {
                if !t.working_dir.is_empty() && Path::new(&t.working_dir).is_dir() {
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
                .ok_or_else(|| "更新后读取任务失败".to_string())
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
                    "path": path_to_clean_str(&p),
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

    let parent = resolved.parent().map(path_to_clean_str).unwrap_or_default();

    Ok(json!({
        "success": true,
        "current": path_to_clean_str(&resolved),
        "parent": parent,
        "drives": drives,
        "dirs": dirs,
    }))
}

fn optional_string(payload: &Value, key: &str) -> Result<Option<String>, String> {
    payload
        .get(key)
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("参数 {key} 必须为字符串"))
        })
        .transpose()
}

fn validate_template_id(id: &str) -> Result<(), String> {
    let (base, domain) = id
        .split_once(':')
        .map(|(a, b)| (a, Some(b)))
        .unwrap_or((id, None));
    if let (Some(meta), Some(domain)) = (get_prompt_meta(base), domain) {
        match meta.domain_scope {
            super::prompts::PromptDomainScope::None => {
                return Err(format!("该提示词不支持 domain 后缀: {id}"));
            }
            super::prompts::PromptDomainScope::Sql if domain != "sql" => {
                return Err(format!("SQL 阶段提示词仅支持 sql 领域: {id}"));
            }
            _ => {}
        }
    }
    if get_prompt_meta(base).is_none()
        || domain
            .map(|d| {
                d.is_empty()
                    || !d
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            })
            .unwrap_or(false)
    {
        return Err(format!("未知或无效的提示词模板 ID: {id}"));
    }
    Ok(())
}

fn validate_prompt_map(map: &BTreeMap<String, String>) -> Result<(), String> {
    for id in map.keys() {
        validate_template_id(id)?;
    }
    Ok(())
}

fn parse_prompt_map(
    payload: &Value,
    key: &str,
) -> Result<Option<BTreeMap<String, String>>, String> {
    let map: Option<BTreeMap<String, String>> = payload
        .get(key)
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| format!("参数 {key} 格式无效: {e}"))?;
    if let Some(ref map) = map {
        validate_prompt_map(map)?;
    }
    Ok(map)
}

#[cfg(test)]
mod prompt_validation_tests {
    use super::*;

    #[test]
    fn guard_suffix_is_rejected_in_ids_and_maps() {
        for id in [
            super::super::prompts::TPL_SYS_GUARD_READONLY,
            super::super::prompts::TPL_SYS_GUARD_COLLAB,
        ] {
            assert!(validate_template_id(id).is_ok());
            let suffixed = format!("{id}:backend");
            assert!(validate_template_id(&suffixed).is_err());
            assert!(
                validate_prompt_map(&BTreeMap::from([(suffixed.clone(), "draft".into())])).is_err()
            );
            let payload = json!({"promptTemplates": {(suffixed): "draft"}});
            assert!(parse_prompt_map(&payload, "promptTemplates").is_err());
        }
    }

    #[test]
    fn role_suffix_and_prompt_map_types_are_validated() {
        let id = format!("{}:frontend", super::super::prompts::TPL_WORKER_PIPELINE);
        assert!(validate_template_id(&id).is_ok());
        assert!(validate_template_id(&format!("{id}:extra")).is_err());
        assert!(parse_prompt_map(&json!({"promptTemplates": []}), "promptTemplates").is_err());
        assert!(
            parse_prompt_map(&json!({"promptTemplates": {(id): 42}}), "promptTemplates").is_err()
        );
        assert!(parse_prompt_map(&json!({}), "promptTemplates")
            .unwrap()
            .is_none());
    }
}

async fn save_design_payload(
    host: &dyn TeamHost,
    payload: &Value,
    run_id: &str,
) -> Result<(), String> {
    if payload.get("tasks").is_some() {
        let parsed =
            serde_json::from_value(payload.clone()).map_err(|e| format!("任务参数无效: {e}"))?;
        save_draft_internal(host, parsed).await?;
    } else {
        let ep = optional_string(payload, "executionPlan")?;
        let sql = optional_string(payload, "sqlSchemaDraft")?;
        if ep.is_some() || sql.is_some() {
            let id = run_id.to_string();
            tokio::task::spawn_blocking(move || {
                with_team_store(|s| {
                    load_run_conn(&s.conn, &id)?.ok_or_else(|| format!("未找到任务: {id}"))?;
                    update_run_design_texts_conn(&s.conn, &id, ep.as_deref(), sql.as_deref())
                })
            })
            .await
            .map_err(|e| e.to_string())??;
        }
    }
    Ok(())
}
