//! `zap-agent-teams` 常驻编排执行器：
//! - DAG 波次并发调度（信号量限流）+ 上游契约自动接力注入
//! - 跨项目独立会话绑定专属 `neural_memory` 智能体 + 无人值守 `ApprovalMode::Auto` 执行
//! - 权责分离独立验收闸门（Verifier）+ 最多 `max_repairs` 次（默认 2 次）自动打回原会话返工 + 超限熔断
//! - 三方跨库联合排障会诊（`triage`）并发勘察与根因汇总报告生成

use super::prompts::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use zap_kernel::message::Message;
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;

use crate::host::{HostChildTurnRequest, TeamHost};

use super::planner::{
    build_pipeline_plan_for_run, extract_json_object, generate_unified_interface_spec,
};
use super::store::{
    append_event_conn, append_task_note_conn, append_tasks_to_run_conn, claim_task_attempt_conn,
    finish_task_fenced_conn, load_config_conn, load_contracts_for_run_conn,
    load_custom_projects_conn, load_role_presets_conn, load_run_conn, load_run_detail_conn,
    load_task_conn, load_tasks_for_run_conn, schedule_revision_conn, set_run_status_conn,
    set_task_status_fenced_conn, update_run_delivery_conn, update_run_design_conn,
    update_run_interface_spec_conn, update_run_sql_binding_conn, update_run_stage_conn,
    update_task_followup_result_conn, upsert_contract_conn, upsert_role_preset_conn,
    with_team_store, RevisionTaskTarget,
};
use super::tools::{register_team_session_ctx, unregister_team_session_ctx, ActiveTeamSessionCtx};
use super::types::{
    clean_path_str, compute_and_assign_waves, domain_badge_cn, normalize_domain, path_to_clean_str,
    truncate_chars_safe, RoleBindingPreset, TeamContract, TeamImage, TeamRun, TeamTask,
    HARD_CAP_MAX_REPAIRS,
};

pub(crate) async fn global_prompt_templates() -> Result<BTreeMap<String, String>, String> {
    tokio::task::spawn_blocking(|| {
        with_team_store(|s| Ok(load_config_conn(&s.conn).prompt_templates))
    })
    .await
    .map_err(|e| e.to_string())?
}

pub(crate) async fn execution_prompt(
    run: &TeamRun,
    template_id: &str,
    domain: Option<&str>,
    vars: &[(&str, &str)],
) -> Result<String, String> {
    let globals = global_prompt_templates().await?;
    let (template, _) =
        resolve_effective_template(template_id, domain, Some(&run.prompt_overrides), &globals);
    Ok(render_template_single_pass(&template, vars))
}

struct ActiveRunHandle {
    cancel: CancellationToken,
    child_cancels: Mutex<Vec<CancellationToken>>,
    wake: Notify,
    append_epoch: AtomicU64,
}

impl ActiveRunHandle {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel: CancellationToken::new(),
            child_cancels: Mutex::new(Vec::new()),
            wake: Notify::new(),
            append_epoch: AtomicU64::new(0),
        })
    }

    fn register_child(&self, token: CancellationToken) {
        if self.cancel.is_cancelled() {
            token.cancel();
            return;
        }
        let mut guard = self.child_cancels.lock().unwrap_or_else(|e| e.into_inner());
        guard.push(token);
    }

    fn cancel_all(&self) {
        self.cancel.cancel();
        let guard = self.child_cancels.lock().unwrap_or_else(|e| e.into_inner());
        for c in guard.iter() {
            c.cancel();
        }
    }
}

fn active_runs() -> &'static Mutex<HashMap<String, Arc<ActiveRunHandle>>> {
    static RUNS: OnceLock<Mutex<HashMap<String, Arc<ActiveRunHandle>>>> = OnceLock::new();
    RUNS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn is_run_executing(run_id: &str) -> bool {
    let guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
    guard.contains_key(run_id.trim())
}

pub(crate) fn wake_active_run(run_id: &str) -> bool {
    let guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(handle) = guard.get(run_id.trim()) {
        handle.append_epoch.fetch_add(1, Ordering::SeqCst);
        handle.wake.notify_one();
        true
    } else {
        false
    }
}

/// 异步启动阶段 1（主控制定《执行计划》并自动拆解生成对应智能体），完成后停在 `team_binding` 等待用户增删智能体并绑定各自项目目录。
pub async fn start_design_phase(host: Arc<dyn TeamHost>, run_id: String, current_wd: PathBuf) {
    let rid = run_id.trim().to_string();
    let handle = ActiveRunHandle::new();
    {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains_key(&rid) {
            return;
        }
        guard.insert(rid.clone(), Arc::clone(&handle));
    }

    tokio::spawn(async move {
        run_design_pipeline(host, rid.clone(), handle, current_wd, None, false).await;
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid);
    });
}

/// 阶段 2：用户在 `sql_binding` 阶段配置好 SQL 参考目录 / 规范记忆 / 模型后，派发只读 SQL 架构师设计表结构 DDL。
pub async fn start_sql_design_phase(
    host: Arc<dyn TeamHost>,
    run_id: &str,
    current_wd: PathBuf,
    sql_override: Option<(String, String, String)>,
) -> Result<(), String> {
    let rid = run_id.trim().to_string();
    if rid.is_empty() {
        return Err("run_id 不能为空".to_string());
    }
    {
        let guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains_key(&rid) {
            return Err("当前任务正在生成或执行中，请稍候".to_string());
        }
    }

    let rid_check = rid.clone();
    let fallback_wd = path_to_clean_str(&current_wd);
    tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let run = load_run_conn(&store.conn, &rid_check)?
                .ok_or_else(|| format!("Run not found: {rid_check}"))?;
            let (raw_wd, raw_agent, raw_model) = if let Some((w, a, m)) = sql_override {
                (
                    clean_path_str(&w),
                    a.trim().to_string(),
                    m.trim().to_string(),
                )
            } else if !run.sql_working_dir.trim().is_empty() {
                (
                    run.sql_working_dir.clone(),
                    run.sql_agent_id.clone(),
                    run.sql_model.clone(),
                )
            } else {
                let presets = load_role_presets_conn(&store.conn)?;
                let projects = load_custom_projects_conn(&store.conn)?;
                if let Some(p) = presets.iter().find(|p| {
                    normalize_domain(&p.domain) == "sql"
                        && !clean_path_str(&p.working_dir).is_empty()
                }) {
                    (
                        clean_path_str(&p.working_dir),
                        p.agent_id.clone(),
                        p.model.clone(),
                    )
                } else if let Some(p) = projects.iter().find(|p| {
                    normalize_domain(&p.domain) == "sql"
                        && !clean_path_str(&p.working_dir).is_empty()
                }) {
                    (
                        clean_path_str(&p.working_dir),
                        p.agent_id.clone(),
                        p.model.clone(),
                    )
                } else {
                    (fallback_wd.clone(), "none".to_string(), String::new())
                }
            };

            let effective_wd = if raw_wd.is_empty() {
                fallback_wd.clone()
            } else {
                raw_wd
            };
            if effective_wd.is_empty() {
                return Err("请先选择 SQL 参考项目目录，再生成数据库表结构".to_string());
            }
            let p = Path::new(&effective_wd);
            if !p.exists() || !p.is_dir() {
                return Err(format!("SQL 参考项目目录不存在或不可访问：{effective_wd}"));
            }

            update_run_sql_binding_conn(
                &store.conn,
                &rid_check,
                Some(&effective_wd),
                Some(&raw_agent),
                Some(&raw_model),
                None,
            )?;

            let _ = store.conn.execute(
                "UPDATE team_runs SET requires_db = 1 WHERE id = ?1",
                rusqlite::params![rid_check],
            );
            update_run_stage_conn(&store.conn, &rid_check, Some("running"), "sql_designing")?;
            Ok(())
        })
    })
    .await
    .map_err(|e| e.to_string())??;

    let handle = ActiveRunHandle::new();
    {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(rid.clone(), Arc::clone(&handle));
    }

    tokio::spawn(async move {
        run_design_pipeline(host, rid.clone(), handle, current_wd, None, true).await;
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid);
    });

    Ok(())
}

/// 用户在确认台提出修改意见后，异步重新调整《执行计划》或《表结构 DDL》。
pub async fn revise_design_phase(
    host: Arc<dyn TeamHost>,
    run_id: String,
    feedback: String,
    current_wd: PathBuf,
) -> Result<(), String> {
    let rid = run_id.trim().to_string();
    let fb = feedback.trim().to_string();
    if fb.is_empty() {
        return Err("请输入需要调整的修改意见".to_string());
    }

    {
        let guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains_key(&rid) {
            return Err("当前任务正在生成或执行中，请稍候".to_string());
        }
    }

    let rid_stage = rid.clone();
    let run_sql_only = tokio::task::spawn_blocking(move || {
        with_team_store(|s| {
            let r = load_run_conn(&s.conn, &rid_stage)?;
            Ok::<_, String>(r.map(|x| x.stage == "awaiting_approval").unwrap_or(false))
        })
    })
    .await
    .map_err(|e| e.to_string())??;

    let handle = ActiveRunHandle::new();
    {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(rid.clone(), Arc::clone(&handle));
    }

    tokio::spawn(async move {
        run_design_pipeline(
            host,
            rid.clone(),
            handle,
            current_wd,
            Some(fb),
            run_sql_only,
        )
        .await;
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid);
    });

    Ok(())
}

async fn run_design_pipeline(
    host: Arc<dyn TeamHost>,
    run_id: String,
    handle: Arc<ActiveRunHandle>,
    current_wd: PathBuf,
    feedback: Option<String>,
    run_sql_stage: bool,
) {
    let loaded = tokio::task::spawn_blocking({
        let rid = run_id.clone();
        move || {
            with_team_store(|store| {
                Ok::<_, String>((
                    load_run_conn(&store.conn, &rid)?,
                    load_tasks_for_run_conn(&store.conn, &rid)?,
                ))
            })
        }
    })
    .await
    .ok()
    .and_then(Result::ok);

    let Some((Some(mut run), tasks)) = loaded else {
        return;
    };

    let fb_str = feedback.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let is_sql_only_revision = run_sql_stage
        && fb_str.is_some()
        && (run.requires_db || fb_str.map(feedback_mentions_db).unwrap_or(false))
        && !run.execution_plan.trim().is_empty();

    if !run_sql_stage || (fb_str.is_some() && !is_sql_only_revision) {
        let _ = tokio::task::spawn_blocking({
            let rid = run_id.clone();
            move || {
                with_team_store(|s| {
                    update_run_stage_conn(&s.conn, &rid, Some("running"), "planning")
                })
            }
        })
        .await;

        match build_pipeline_plan_for_run(&host, &run, &current_wd, fb_str).await {
            Ok((new_title, requires_db, execution_plan, new_tasks)) => {
                if handle.cancel.is_cancelled() {
                    return;
                }
                let keep_sql_draft = if !run_sql_stage && !run.sql_schema_draft.trim().is_empty() {
                    run.sql_schema_draft.clone()
                } else {
                    String::new()
                };
                let next_stage = if run_sql_stage && requires_db {
                    "sql_designing"
                } else if run_sql_stage {
                    "team_binding"
                } else if requires_db && keep_sql_draft.trim().is_empty() {
                    "sql_binding"
                } else {
                    "team_binding"
                };
                let next_status = if next_stage == "sql_designing" {
                    "running"
                } else {
                    "draft"
                };
                let _ = tokio::task::spawn_blocking({
                    let rid = run_id.clone();
                    let t = new_title.clone();
                    let ep = execution_plan.clone();
                    let nt = new_tasks.clone();
                    let ns = next_stage.to_string();
                    let st = next_status.to_string();
                    let ks = keep_sql_draft.clone();
                    move || {
                        with_team_store(|s| {
                            update_run_design_conn(
                                &mut s.conn,
                                &rid,
                                Some(&t),
                                &st,
                                &ns,
                                requires_db,
                                &ep,
                                &ks,
                                Some(&nt),
                            )?;
                            let _ = append_event_conn(
                                &s.conn,
                                &rid,
                                "design_plan_ready",
                                "",
                                "lead_planner",
                                &format!(
                                    "主控已完成需求分析与分工拆解（{} 个研发端 · {}）。",
                                    nt.len(),
                                    if ns == "sql_binding" {
                                        "需要新建/改表，请先配置 SQL 参考目录并生成表结构"
                                    } else if ns == "team_binding" {
                                        "无需改表或表结构已就绪，请配置研发智能体并绑定项目"
                                    } else {
                                        "正在由 SQL 架构师设计表结构"
                                    }
                                ),
                                &ep,
                            );
                            Ok(())
                        })
                    }
                })
                .await;

                run.title = new_title;
                run.requires_db = requires_db;
                run.execution_plan = execution_plan;
                if !run_sql_stage {
                    return;
                }
            }
            Err(e) => {
                mark_run_failed(&run_id, &format!("生成执行计划失败：{e}")).await;
                return;
            }
        }
    } else if run_sql_stage {
        run.requires_db = true;
    }

    if handle.cancel.is_cancelled() || !run_sql_stage {
        return;
    }

    if run.requires_db {
        let _ = tokio::task::spawn_blocking({
            let rid = run_id.clone();
            move || {
                with_team_store(|s| {
                    update_run_stage_conn(&s.conn, &rid, Some("running"), "sql_designing")?;
                    let _ = append_event_conn(
                        &s.conn,
                        &rid,
                        "design_sql_started",
                        "",
                        "sql_designer",
                        "SQL 智能体正在按你绑定的项目目录只读勘察表规范并设计表结构 DDL…",
                        "",
                    );
                    Ok(())
                })
            }
        })
        .await;

        let latest_tasks = tokio::task::spawn_blocking({
            let rid_t = run_id.clone();
            move || with_team_store(|s| load_tasks_for_run_conn(&s.conn, &rid_t))
        })
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or(tasks);

        match run_sql_designer_turn(&host, &handle, &run, &latest_tasks, &current_wd, fb_str).await
        {
            Ok(ddl_md) => {
                if handle.cancel.is_cancelled() {
                    return;
                }
                let _ = tokio::task::spawn_blocking({
                    let rid = run_id.clone();
                    let ep = run.execution_plan.clone();
                    let ddl = ddl_md.clone();
                    move || {
                        with_team_store(|s| {
                            update_run_design_conn(
                                &mut s.conn,
                                &rid,
                                None,
                                "draft",
                                "awaiting_approval",
                                true,
                                &ep,
                                &ddl,
                                None,
                            )?;
                            let _ = append_event_conn(
                                &s.conn,
                                &rid,
                                "design_awaiting_approval",
                                "",
                                "sql_designer",
                                "表结构方案已设计完成，等待你拍板确认（确认前不会修改任何代码）。",
                                &ddl,
                            );
                            Ok(())
                        })
                    }
                })
                .await;
            }
            Err(e) => {
                mark_run_failed(&run_id, &format!("SQL 表结构设计失败：{e}")).await;
            }
        }
    }
}

fn feedback_mentions_db(fb: &str) -> bool {
    let low = fb.to_ascii_lowercase();
    low.contains("表")
        || low.contains("字段")
        || low.contains("sql")
        || low.contains("ddl")
        || low.contains("索引")
        || low.contains("达梦")
        || low.contains("数据库")
}

async fn run_sql_designer_turn(
    host: &Arc<dyn TeamHost>,
    handle: &Arc<ActiveRunHandle>,
    run: &TeamRun,
    tasks: &[TeamTask],
    current_wd: &Path,
    feedback: Option<&str>,
) -> Result<String, String> {
    let rid = run.id.clone();
    let binding_run = tokio::task::spawn_blocking(move || {
        with_team_store(|s| {
            load_run_conn(&s.conn, &rid)?.ok_or_else(|| format!("Run not found: {rid}"))
        })
    })
    .await
    .map_err(|e| e.to_string())??;
    let run = &binding_run;
    let _ = (tasks, current_wd);
    if run.sql_working_dir.trim().is_empty() {
        return Err("SQL working directory is not bound to this run".to_string());
    }
    let effective_wd = PathBuf::from(run.sql_working_dir.trim());
    let effective_model = if run.sql_model.trim().is_empty() {
        run.lead_model.trim().to_string()
    } else {
        run.sql_model.trim().to_string()
    };
    let run_binding = run.clone();
    let wd_binding = effective_wd.clone();
    let session_id = tokio::task::spawn_blocking(move || {
        let sid = super::sessions::session_id_for_binding(
            &run_binding.sql_session_id,
            &format!("team_{}_sql_designer", run_binding.id),
            &wd_binding,
        )?;
        zap_neural_memory::bind_session_neural_agent(&sid, &run_binding.sql_agent_id)?;
        with_team_store(|s| {
            update_run_sql_binding_conn(&s.conn, &run_binding.id, None, None, None, Some(&sid))
        })?;
        Ok::<_, String>(sid)
    })
    .await
    .map_err(|e| e.to_string())??;
    let hint = inspect_dir_profile(&effective_wd).to_hint();
    let prompt = execution_prompt(
        run,
        if feedback.is_some() {
            TPL_SQL_DESIGN_REVISE
        } else {
            TPL_SQL_DESIGN_INITIAL
        },
        Some("sql"),
        &[
            ("working_dir", &run.sql_working_dir),
            ("dir_profile_hint", &hint),
            ("user_goal", &run.user_goal),
            ("execution_plan", &run.execution_plan),
            ("feedback", feedback.unwrap_or("")),
            ("sql_schema_draft", &run.sql_schema_draft),
        ],
    )
    .await?;
    register_team_session_ctx(
        &session_id,
        ActiveTeamSessionCtx {
            run_id: run.id.clone(),
            task_id: String::new(),
            domain: "sql".to_string(),
            actor_label: "SQL designer".to_string(),
            provider_name: effective_model.clone(),
            read_only_mode: true,
        },
    );
    let session_title = format!("🗄 [SQL表结构设计] {}", run.title);
    let result = run_native_session_turn(
        host,
        handle,
        run,
        &session_id,
        &session_title,
        &effective_wd,
        (!effective_model.is_empty()).then_some(effective_model.as_str()),
        &prompt,
        true,
        &run.goal_images,
    )
    .await;
    unregister_team_session_ctx(&session_id);
    let (output, _) = result?;

    if output.trim().is_empty() {
        return Err("SQL 智能体未返回有效的表结构方案".to_string());
    }
    Ok(output)
}

/// 启动或从断点恢复执行指定的团队任务（后台常驻运行，关掉/刷新网页不中断）。
pub async fn start_or_resume_run(host: Arc<dyn TeamHost>, run_id: &str) -> Result<(), String> {
    let rid = run_id.trim().to_string();
    if rid.is_empty() {
        return Err("run_id 不能为空".to_string());
    }

    {
        let guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains_key(&rid) {
            return Ok(());
        }
    }

    let current_wd = host.current_working_dir().await;
    let rid_prep = rid.clone();
    let (run_snap, max_parallel) = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let run = load_run_conn(&store.conn, &rid_prep)?
                .ok_or_else(|| format!("未找到任务: {rid_prep}"))?;
            // 若在设计阶段因宿主重启被暂停，恢复时继续对应设计阶段
            if run.mode == "pipeline"
                && ((run.stage == "planning" && run.execution_plan.trim().is_empty())
                    || (run.stage == "sql_designing" && run.sql_schema_draft.trim().is_empty()))
            {
                return Ok((run, 0usize));
            }
            let mut tasks = load_tasks_for_run_conn(&store.conn, &rid_prep)?;
            if tasks.is_empty() {
                return Err("当前任务中没有可执行的子任务".to_string());
            }
            for t in &tasks {
                if t.working_dir.trim().is_empty() {
                    return Err(format!("子任务「{}」尚未选择项目代码目录", t.title));
                }
                let p = Path::new(t.working_dir.trim());
                if !p.exists() || !p.is_dir() {
                    return Err(format!(
                        "子任务「{}」的项目目录不存在或不可访问：{}",
                        t.title, t.working_dir
                    ));
                }
            }
            // 校验 DAG 无环并保存用户的领域默认绑定习惯
            let _ = compute_and_assign_waves(&mut tasks)?;
            for t in &tasks {
                let _ = upsert_role_preset_conn(
                    &store.conn,
                    &RoleBindingPreset {
                        domain: normalize_domain(&t.domain),
                        working_dir: t.working_dir.clone(),
                        agent_id: t.agent_id.clone(),
                        model: t.model.clone(),
                        updated_at: String::new(),
                    },
                );
            }

            // 将非 completed 的任务收敛为待调度态
            store
                .conn
                .execute(
                    "UPDATE team_tasks
                     SET status = 'pending', attempt_id = ''
                     WHERE run_id = ?1 AND status != 'completed'",
                    rusqlite::params![rid_prep],
                )
                .map_err(|e| e.to_string())?;
            let next_stage = if run.mode == "triage" {
                "triage"
            } else if run.interface_spec.trim().is_empty() && tasks.iter().all(|t| t.attempt == 0) {
                "contracting"
            } else {
                "executing"
            };
            update_run_stage_conn(&store.conn, &rid_prep, Some("running"), next_stage)?;
            let _ = append_event_conn(
                &store.conn,
                &rid_prep,
                "run_started",
                "",
                "orchestrator",
                &format!("团队协作引擎已启动《{}》", run.title),
                "",
            );
            Ok((run, load_config_conn(&store.conn).max_parallel_per_wave))
        })
    })
    .await
    .map_err(|e| e.to_string())??;

    if max_parallel == 0 && run_snap.mode == "pipeline" {
        if run_snap.stage == "sql_designing" {
            start_sql_design_phase(host, &rid, current_wd, None).await?;
        } else {
            start_design_phase(host, rid, current_wd).await;
        }
        return Ok(());
    }

    let handle = ActiveRunHandle::new();
    {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains_key(&rid) {
            return Ok(());
        }
        guard.insert(rid.clone(), Arc::clone(&handle));
    }

    let rid_bg = rid.clone();
    tokio::spawn(async move {
        // 阶段 3：若为研发流水线且尚未生成统一接口方案，则先由主控基于定稿表结构锁定前后端接口、字典与权限方案
        if run_snap.mode == "pipeline" && run_snap.interface_spec.trim().is_empty() {
            let _ = tokio::task::spawn_blocking({
                let rid_ev = rid_bg.clone();
                move || {
                    with_team_store(|s| {
                        let _ = append_event_conn(
                            &s.conn,
                            &rid_ev,
                            "contract_generating",
                            "",
                            "lead_planner",
                            "用户已确认方案！主控正在基于表结构统一约定后端接口、字典码值与权限标识…",
                            "",
                        );
                        Ok(())
                    })
                }
            })
            .await;

            let tasks_for_spec = tokio::task::spawn_blocking({
                let rid_q = rid_bg.clone();
                move || with_team_store(|s| load_tasks_for_run_conn(&s.conn, &rid_q))
            })
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default();

            let spec =
                generate_unified_interface_spec(&run_snap, &tasks_for_spec, &current_wd).await;
            if handle.cancel.is_cancelled() {
                let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
                guard.remove(&rid_bg);
                return;
            }
            let _ = tokio::task::spawn_blocking({
                let rid_s = rid_bg.clone();
                let sp = spec.clone();
                let ddl = run_snap.sql_schema_draft.clone();
                move || {
                    with_team_store(|s| {
                        let extra_spec = load_run_conn(&s.conn, &rid_s)?
                            .map(|r| r.interface_spec)
                            .unwrap_or_default();
                        let merged_sp = if extra_spec.trim().is_empty() {
                            sp
                        } else {
                            format!("{}\n\n{}", sp.trim_end(), extra_spec.trim())
                        };
                        update_run_interface_spec_conn(&s.conn, &rid_s, "executing", &merged_sp)?;
                        if !ddl.trim().is_empty() {
                            let _ = upsert_contract_conn(
                                &s.conn,
                                &rid_s,
                                "sql_approved",
                                "sql",
                                "用户已拍板确认的数据库表结构 DDL",
                                &ddl,
                                "已在写代码前由用户确认定稿",
                            );
                        }
                        let _ = upsert_contract_conn(
                            &s.conn,
                            &rid_s,
                            "lead_contract",
                            "backend",
                            "主控统一约定的前后端接口、字典与权限方案",
                            &merged_sp,
                            "后端与各前端必须严格遵守同一份接口与字段契约并发开发",
                        );
                        let _ = append_event_conn(
                            &s.conn,
                            &rid_s,
                            "contract_locked",
                            "",
                            "lead_planner",
                            "前后端统一接口方案已锁定！后端与所有前端立即同时并发开工。",
                            &merged_sp,
                        );
                        Ok(())
                    })
                }
            })
            .await;
        }

        orchestrate_run_loop(
            host,
            rid_bg.clone(),
            Arc::clone(&handle),
            max_parallel.clamp(1, 6),
        )
        .await;
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard
            .get(&rid_bg)
            .is_some_and(|cur| Arc::ptr_eq(cur, &handle))
        {
            guard.remove(&rid_bg);
        }
    });

    Ok(())
}

/// 在全流程任意阶段（含运行中途不中断、暂停、已完成）安全追加新成员并按需调度执行。
pub(crate) async fn append_tasks_and_dispatch(
    host: Arc<dyn TeamHost>,
    run_id: &str,
    mut new_tasks: Vec<TeamTask>,
    interface_spec_addition: Option<String>,
    replace_interface_spec: Option<String>,
    auto_start: bool,
) -> Result<(super::types::TeamRunDetail, bool), String> {
    let rid = run_id.trim().to_string();
    if rid.is_empty() {
        return Err("缺少 runId".to_string());
    }

    let currently_executing = is_run_executing(&rid);
    let agents =
        tokio::task::spawn_blocking(|| zap_neural_memory::list_neural_agents().unwrap_or_default())
            .await
            .unwrap_or_default();

    for t in &mut new_tasks {
        t.run_id = rid.clone();
        t.domain = normalize_domain(&t.domain);
        t.working_dir = clean_path_str(&t.working_dir);
        if currently_executing || auto_start {
            if t.working_dir.is_empty() {
                return Err(format!("新增成员「{}」尚未选择项目代码目录", t.title));
            }
            let p = Path::new(&t.working_dir);
            if !p.exists() || !p.is_dir() {
                return Err(format!(
                    "新增成员「{}」的项目目录不存在或不可访问：{}",
                    t.title, t.working_dir
                ));
            }
        }
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
    }

    let rid_db = rid.clone();
    let tasks_for_db = new_tasks.clone();
    let spec_add = interface_spec_addition.clone();
    let spec_rep = replace_interface_spec.clone();
    let (was_running, has_new_tasks) = tokio::task::spawn_blocking(move || {
        with_team_store(|s| {
            let cur =
                load_run_conn(&s.conn, &rid_db)?.ok_or_else(|| format!("未找到任务: {rid_db}"))?;
            let was_running = cur.status == "running";
            let _ = append_tasks_to_run_conn(
                &mut s.conn,
                &rid_db,
                &tasks_for_db,
                spec_add.as_deref(),
                spec_rep.as_deref(),
            )?;
            for t in &tasks_for_db {
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
            Ok::<_, String>((was_running, !tasks_for_db.is_empty()))
        })
    })
    .await
    .map_err(|e| e.to_string())??;

    if has_new_tasks {
        let woken = wake_active_run(&rid);
        if !woken && (auto_start || was_running) {
            start_or_resume_run(host, &rid).await?;
        }
    }

    let rid_q = rid.clone();
    let detail = tokio::task::spawn_blocking(move || {
        with_team_store(|s| load_run_detail_conn(&s.conn, &rid_q, 0))
    })
    .await
    .map_err(|e| e.to_string())??
    .ok_or_else(|| format!("未找到任务: {rid}"))?;
    let live = is_run_executing(&rid);
    Ok((detail, live))
}

/// 发起单成员定向改造或跨成员联动变更（可选更新统一契约，自动按 DAG 波次先后唤醒既有会话执行）。
pub(crate) async fn dispatch_member_revision(
    host: Arc<dyn TeamHost>,
    run_id: &str,
    mut targets: Vec<RevisionTaskTarget>,
    interface_spec_addition: Option<String>,
    replace_interface_spec: Option<String>,
    auto_start: bool,
) -> Result<(super::types::TeamRunDetail, bool), String> {
    let rid = run_id.trim().to_string();
    if rid.is_empty() {
        return Err("缺少 runId".to_string());
    }
    if targets.is_empty() {
        return Err("请至少选择一位需要修改或联动的成员".to_string());
    }

    let rid_check = rid.clone();
    let target_ids: Vec<String> = targets.iter().map(|t| t.task_id.clone()).collect();
    let tasks_in_db = tokio::task::spawn_blocking(move || {
        with_team_store(|s| {
            let _ = load_run_conn(&s.conn, &rid_check)?
                .ok_or_else(|| format!("未找到任务: {rid_check}"))?;
            load_tasks_for_run_conn(&s.conn, &rid_check)
        })
    })
    .await
    .map_err(|e| e.to_string())??;

    for tid in &target_ids {
        let task = tasks_in_db
            .iter()
            .find(|t| t.id == *tid)
            .ok_or_else(|| format!("未找到子任务: {tid}"))?;
        if auto_start {
            let wd = task.working_dir.trim();
            if wd.is_empty() {
                return Err(format!("成员「{}」尚未绑定项目代码目录", task.title));
            }
            let p = Path::new(wd);
            if !p.exists() || !p.is_dir() {
                return Err(format!("成员「{}」的项目目录不可访问：{wd}", task.title));
            }
        }
    }

    for t in &mut targets {
        if let Some(note_clean) = Some(t.note.trim().to_string()).filter(|s| !s.is_empty()) {
            t.note = note_clean;
        }
        t.images.truncate(4);
    }

    let rid_db = rid.clone();
    let spec_add = interface_spec_addition.clone();
    let spec_rep = replace_interface_spec.clone();
    tokio::task::spawn_blocking(move || {
        with_team_store(|s| {
            schedule_revision_conn(
                &mut s.conn,
                &rid_db,
                &targets,
                spec_add.as_deref(),
                spec_rep.as_deref(),
                auto_start,
            )
        })
    })
    .await
    .map_err(|e| e.to_string())??;

    if auto_start {
        let woken = wake_active_run(&rid);
        if !woken {
            start_or_resume_run(host, &rid).await?;
        }
    }

    let rid_q = rid.clone();
    let detail = tokio::task::spawn_blocking(move || {
        with_team_store(|s| load_run_detail_conn(&s.conn, &rid_q, 0))
    })
    .await
    .map_err(|e| e.to_string())??
    .ok_or_else(|| format!("未找到任务: {rid}"))?;
    let live = is_run_executing(&rid);
    Ok((detail, live))
}

/// 暂停正在执行的派工单（立即取消在途子回合，保留已完成进度以便随时续跑）。
pub async fn pause_team_run(run_id: &str) -> Result<(), String> {
    let rid = run_id.trim().to_string();
    if let Some(handle) = {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid)
    } {
        handle.cancel_all();
    }

    tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            set_run_status_conn(&store.conn, &rid, "paused", None)?;
            let _ = store.conn.execute(
                "UPDATE team_tasks
                 SET status = 'ready', attempt_id = ''
                 WHERE run_id = ?1 AND status IN ('running', 'verifying', 'repairing')",
                rusqlite::params![rid],
            );
            let _ = append_event_conn(
                &store.conn,
                &rid,
                "run_paused",
                "",
                "user",
                "已暂停团队执行，已完成波次的进度与契约已安全保留。",
                "",
            );
            Ok(())
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// 终止正在执行的派工单。
pub async fn cancel_team_run(run_id: &str) -> Result<(), String> {
    let rid = run_id.trim().to_string();
    if let Some(handle) = {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid)
    } {
        handle.cancel_all();
    }

    tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            set_run_status_conn(&store.conn, &rid, "cancelled", None)?;
            let _ = store.conn.execute(
                "UPDATE team_tasks
                 SET status = 'cancelled', attempt_id = ''
                 WHERE run_id = ?1 AND status != 'completed'",
                rusqlite::params![rid],
            );
            let _ = append_event_conn(
                &store.conn,
                &rid,
                "run_cancelled",
                "",
                "user",
                "已手动终止该团队派工单。",
                "",
            );
            Ok(())
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

/// 重置单个失败/未达标子任务并继续运行整条流水线。
pub async fn retry_single_task(
    host: Arc<dyn TeamHost>,
    run_id: &str,
    task_id: &str,
) -> Result<(), String> {
    let rid = run_id.trim().to_string();
    let tid = task_id.trim().to_string();
    tokio::task::spawn_blocking({
        let rid = rid.clone();
        let tid = tid.clone();
        move || {
            with_team_store(|store| {
                store
                    .conn
                    .execute(
                        "UPDATE team_tasks
                         SET status = 'pending', attempt_id = '', repair_count = 0,
                             verify_verdict = '', verify_feedback = ''
                         WHERE id = ?1 AND run_id = ?2",
                        rusqlite::params![tid, rid],
                    )
                    .map_err(|e| e.to_string())?;
                let _ = append_event_conn(
                    &store.conn,
                    &rid,
                    "task_retry_queued",
                    &tid,
                    "user",
                    "已重置子任务状态并重新加入调度队列。",
                    "",
                );
                Ok(())
            })
        }
    })
    .await
    .map_err(|e| e.to_string())??;

    start_or_resume_run(host, &rid).await
}

/// 向指定成员发送补充指令：
/// - 若该成员尚未派生子会话或当前任务正在执行中：将指令追加到子任务描述并写入协作账本（`member_note`）；
/// - 若该成员已存在独立子会话且当前空闲：直接唤醒其原生子会话执行一轮定向补充回合，并将答复写入账本。
pub async fn send_member_note(
    host: Arc<dyn TeamHost>,
    run_id: &str,
    task_id: &str,
    note: &str,
    images: &[TeamImage],
) -> Result<(String, bool), String> {
    let rid = run_id.trim().to_string();
    let tid = task_id.trim().to_string();
    let clean_note = note.trim().to_string();
    if rid.is_empty() || tid.is_empty() {
        return Err("缺少 runId 或 taskId".to_string());
    }
    if clean_note.is_empty() {
        return Err("请输入补充指令内容".to_string());
    }

    let (run, task) = tokio::task::spawn_blocking({
        let rid = rid.clone();
        let tid = tid.clone();
        move || {
            with_team_store(|store| {
                let run = load_run_conn(&store.conn, &rid)?
                    .ok_or_else(|| format!("未找到任务单: {rid}"))?;
                let task = load_task_conn(&store.conn, &tid)?
                    .ok_or_else(|| format!("未找到子任务: {tid}"))?;
                if task.run_id != rid {
                    return Err("子任务不属于该任务单".to_string());
                }
                Ok::<_, String>((run, task))
            })
        }
    })
    .await
    .map_err(|e| e.to_string())??;

    let actor_label = if task.agent_label.trim().is_empty() {
        domain_badge_cn(&task.domain).to_string()
    } else {
        format!("{} ({})", domain_badge_cn(&task.domain), task.agent_label)
    };

    // 若成员尚未创建独立会话，或当前整单仍在执行中：追加指令到任务描述并写入事件账本，避免与在途回合冲突
    let can_dispatch_live = !task.session_id.trim().is_empty()
        && !task.working_dir.trim().is_empty()
        && Path::new(task.working_dir.trim()).is_dir()
        && !is_run_executing(&rid);

    if !can_dispatch_live {
        let mode = if is_run_executing(&rid) {
            "logged_while_running"
        } else {
            "appended_to_task"
        };
        let rid_sv = rid.clone();
        let tid_sv = tid.clone();
        let title_sv = task.title.clone();
        let note_sv = clean_note.clone();
        tokio::task::spawn_blocking(move || {
            with_team_store(|store| {
                append_task_note_conn(&store.conn, &rid_sv, &tid_sv, &note_sv)?;
                let _ = append_event_conn(
                    &store.conn,
                    &rid_sv,
                    "member_note",
                    &tid_sv,
                    "user",
                    &format!("向「{}」补充指令（已写入任务上下文）", title_sv),
                    &note_sv,
                );
                Ok::<_, String>(())
            })
        })
        .await
        .map_err(|e| e.to_string())??;
        return Ok((mode.to_string(), is_run_executing(&rid)));
    }

    let prompt = execution_prompt(
        &run,
        TPL_MEMBER_NOTE,
        Some(&task.domain),
        &[
            ("note", &clean_note),
            ("task_title", &task.title),
            ("domain_label", domain_badge_cn(&task.domain)),
            ("project_name", &task.project_name),
            ("working_dir", &task.working_dir),
        ],
    )
    .await?;
    // 认领活跃句柄，允许暂停/终止操作随时安全取消该补充回合
    let handle = ActiveRunHandle::new();
    {
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        if guard.contains_key(&rid) {
            return Err("当前任务正在执行中，请稍后再发送直达指令".to_string());
        }
        guard.insert(rid.clone(), Arc::clone(&handle));
    }

    let binding_task = task.clone();
    let host_for_binding = Arc::clone(&host);
    let binding_result = tokio::task::spawn_blocking(move || {
        let wd = Path::new(binding_task.working_dir.trim());
        let sid = super::sessions::session_id_for_binding(
            &binding_task.session_id,
            &format!("team_{}_{}", binding_task.run_id, binding_task.id),
            wd,
        )?;
        zap_neural_memory::bind_session_neural_agent(&sid, &binding_task.agent_id)?;
        with_team_store(|s| {
            let (_, attempt) = claim_task_attempt_conn(
                &s.conn,
                &binding_task.id,
                &sid,
                &host_for_binding.hash_path(wd),
                &binding_task.status,
            )?;
            Ok((sid, attempt))
        })
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    let (session_id, followup_attempt) = match binding_result {
        Ok(binding) => binding,
        Err(e) => {
            active_runs()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&rid);
            return Err(e);
        }
    };

    let rid_ev = rid.clone();
    let tid_ev = tid.clone();
    let title_ev = task.title.clone();
    let note_ev = clean_note.clone();
    let _ = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let _ = append_event_conn(
                &store.conn,
                &rid_ev,
                "member_note",
                &tid_ev,
                "user",
                &format!("向「{}」独立子会话发送直达补充指令", title_ev),
                &note_ev,
            );
            Ok::<_, String>(())
        })
    })
    .await;

    let effective_model = if !task.model.trim().is_empty() {
        task.model.trim().to_string()
    } else {
        run.lead_model.trim().to_string()
    };
    let session_title = format!("🤝 [{}] {}", domain_badge_cn(&task.domain), task.title);
    let working_dir = PathBuf::from(task.working_dir.trim());
    let read_only = run.mode == "triage";
    let mut existing_files = task.touched_files.clone();
    let rid_bg = rid.clone();
    let tid_bg = tid.clone();

    let images_owned: Vec<TeamImage> = images.iter().take(4).cloned().collect();
    tokio::spawn(async move {
        register_team_session_ctx(
            &session_id,
            ActiveTeamSessionCtx {
                run_id: rid_bg.clone(),
                task_id: tid_bg.clone(),
                domain: task.domain.clone(),
                actor_label: actor_label.clone(),
                provider_name: effective_model.clone(),
                read_only_mode: read_only,
            },
        );

        let turn_res = run_native_session_turn(
            &host,
            &handle,
            &run,
            &session_id,
            &session_title,
            &working_dir,
            (!effective_model.is_empty()).then_some(effective_model.as_str()),
            &prompt,
            read_only,
            &images_owned,
        )
        .await;

        unregister_team_session_ctx(&session_id);

        match turn_res {
            Ok((reply, files)) => {
                for f in files {
                    if !existing_files.contains(&f) {
                        existing_files.push(f);
                    }
                }
                let rid_ok = rid_bg.clone();
                let tid_ok = tid_bg.clone();
                let act_ok = actor_label.clone();
                let title_ok = task.title.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    with_team_store(|store| {
                        let current = load_task_conn(&store.conn, &tid_ok)?;
                        if !current.is_some_and(|t| t.attempt_id == followup_attempt) {
                            return Ok(());
                        }
                        update_task_followup_result_conn(
                            &store.conn,
                            &tid_ok,
                            &reply,
                            &existing_files,
                        )?;
                        let _ = append_event_conn(
                            &store.conn,
                            &rid_ok,
                            "member_reply",
                            &tid_ok,
                            &act_ok,
                            &format!("「{}」已完成补充指令响应", title_ok),
                            &reply,
                        );
                        Ok::<_, String>(())
                    })
                })
                .await;
            }
            Err(err) => {
                log_task_event(
                    &rid_bg,
                    &tid_bg,
                    &actor_label,
                    "member_reply",
                    &format!(
                        "「{}」补充指令执行中止：{}",
                        task.title,
                        truncate_chars_safe(&err, 120)
                    ),
                    &err,
                )
                .await;
            }
        }

        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid_bg);
    });

    Ok(("dispatched_to_session".to_string(), true))
}

async fn orchestrate_run_loop(
    host: Arc<dyn TeamHost>,
    run_id: String,
    handle: Arc<ActiveRunHandle>,
    max_parallel: usize,
) {
    let sem = Arc::new(Semaphore::new(max_parallel));
    let mut join_set: JoinSet<(String, usize, bool)> = JoinSet::new();
    let mut in_flight: HashSet<String> = HashSet::new();
    let mut logged_waves: HashSet<usize> = HashSet::new();
    let mut failed_wave: Option<usize> = None;

    'outer: loop {
        loop {
            if handle.cancel.is_cancelled() {
                return;
            }

            let snap = tokio::task::spawn_blocking({
                let rid = run_id.clone();
                move || {
                    with_team_store(|store| {
                        let run = load_run_conn(&store.conn, &rid)?;
                        let mut tasks = load_tasks_for_run_conn(&store.conn, &rid)?;
                        let _ = compute_and_assign_waves(&mut tasks)?;
                        Ok::<_, String>((run, tasks))
                    })
                }
            })
            .await;

            let (Some(run), tasks) = (match snap {
                Ok(Ok((r, t))) => (r, t),
                Ok(Err(e)) => {
                    mark_run_failed(&run_id, &format!("拓扑校验失败：{e}")).await;
                    return;
                }
                Err(e) => {
                    mark_run_failed(&run_id, &format!("任务读取异常：{e}")).await;
                    return;
                }
            }) else {
                return;
            };

            let status_by_id: HashMap<String, String> = tasks
                .iter()
                .map(|t| (t.id.clone(), t.status.clone()))
                .collect();

            if failed_wave.is_none() {
                let mut eligible: Vec<TeamTask> = tasks
                    .iter()
                    .filter(|t| {
                        t.status != "completed"
                            && !in_flight.contains(&t.id)
                            && t.depends_on.iter().all(|dep| {
                                status_by_id
                                    .get(dep)
                                    .map(|s| s == "completed")
                                    .unwrap_or(true)
                            })
                    })
                    .cloned()
                    .collect();

                // 按波次分批通知并派发（同波及可并行任务立即并发）
                if !eligible.is_empty() {
                    let min_wave = eligible.iter().map(|t| t.wave).min().unwrap_or(1);
                    eligible.retain(|t| t.wave == min_wave || in_flight.is_empty());

                    let waves_to_log: Vec<(usize, usize)> = {
                        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
                        for t in &eligible {
                            if !logged_waves.contains(&t.wave) {
                                *counts.entry(t.wave).or_default() += 1;
                            }
                        }
                        counts.into_iter().collect()
                    };
                    if !waves_to_log.is_empty() {
                        for (w, _) in &waves_to_log {
                            logged_waves.insert(*w);
                        }
                        let rid_ev = run_id.clone();
                        let _ = tokio::task::spawn_blocking(move || {
                            with_team_store(|store| {
                                for (w, cnt) in waves_to_log {
                                    let _ = append_event_conn(
                                        &store.conn,
                                        &rid_ev,
                                        "wave_started",
                                        "",
                                        "orchestrator",
                                        &format!(
                                            "进入第 {} 波次（并发执行 {} 个领域子任务）",
                                            w, cnt
                                        ),
                                        "",
                                    );
                                }
                                Ok(())
                            })
                        })
                        .await;
                    }

                    for task_item in eligible {
                        let tid = task_item.id.clone();
                        let twave = task_item.wave;
                        in_flight.insert(tid.clone());
                        let h = Arc::clone(&host);
                        let rid = run_id.clone();
                        let run_cfg = run.clone();
                        let hdl = Arc::clone(&handle);
                        let sem_clone = Arc::clone(&sem);
                        join_set.spawn(async move {
                            let Ok(_permit) = sem_clone.acquire_owned().await else {
                                return (tid, twave, false);
                            };
                            if hdl.cancel.is_cancelled() {
                                return (tid, twave, false);
                            }
                            let ok =
                                execute_single_task_lifecycle(h, rid, tid.clone(), run_cfg, hdl)
                                    .await;
                            (tid, twave, ok)
                        });
                    }
                }
            }

            if in_flight.is_empty() {
                if let Some(fw) = failed_wave {
                    mark_run_failed(
                        &run_id,
                        &format!("第 {fw} 波次存在未通过验收或执行失败的子任务，流水线已熔断保护现场，可查看原因后一键重试。"),
                    )
                    .await;
                    return;
                }
                let blocked_count = tasks.iter().filter(|t| t.status != "completed").count();
                if blocked_count > 0 {
                    mark_run_failed(
                        &run_id,
                        "存在依赖未就绪或状态异常的阻塞子任务，流水线已暂停保护现场。",
                    )
                    .await;
                    return;
                }
                let finish_epoch = handle.append_epoch.load(Ordering::SeqCst);
                let finalized = finalize_successful_run(&run_id, &run).await;
                let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
                if finalized && handle.append_epoch.load(Ordering::SeqCst) == finish_epoch {
                    if guard
                        .get(&run_id)
                        .is_some_and(|cur| Arc::ptr_eq(cur, &handle))
                    {
                        guard.remove(&run_id);
                    }
                    break 'outer;
                }
                drop(guard);
                let rid_resume = run_id.clone();
                let next_stage = if run.mode == "triage" {
                    "triage"
                } else {
                    "executing"
                };
                let _ = tokio::task::spawn_blocking(move || {
                    with_team_store(|s| {
                        update_run_stage_conn(&s.conn, &rid_resume, Some("running"), next_stage)
                    })
                })
                .await;
                continue 'outer;
            }

            tokio::select! {
                _ = handle.cancel.cancelled() => {
                    return;
                }
                Some(joined) = join_set.join_next() => {
                    match joined {
                        Ok((tid, twave, ok)) => {
                            in_flight.remove(&tid);
                            if !ok && failed_wave.is_none() {
                                failed_wave = Some(twave);
                            }
                        }
                        Err(_) => {
                            if failed_wave.is_none() {
                                failed_wave = Some(1);
                            }
                        }
                    }
                }
                _ = handle.wake.notified() => {
                    // 新插入成员唤醒调度器，立即检查并加入可执行波次
                }
            }
        }
    }
}

async fn execute_single_task_lifecycle(
    host: Arc<dyn TeamHost>,
    run_id: String,
    task_id: String,
    run: TeamRun,
    handle: Arc<ActiveRunHandle>,
) -> bool {
    let host_for_prep = Arc::clone(&host);
    let prep = tokio::task::spawn_blocking({
        let rid = run_id.clone();
        let tid = task_id.clone();
        move || {
            with_team_store(|store| {
                let task = load_task_conn(&store.conn, &tid)?
                    .ok_or_else(|| format!("未找到子任务 {tid}"))?;
                let all_tasks = load_tasks_for_run_conn(&store.conn, &rid)?;
                let all_contracts = load_contracts_for_run_conn(&store.conn, &rid)?;
                let cfg = load_config_conn(&store.conn);
                let globals = cfg.prompt_templates;
                let compact_prompts = cfg.compact_prompts;

                let session_id = super::sessions::session_id_for_binding(
                    &task.session_id,
                    &format!("team_{}_{}", rid, task.id),
                    Path::new(task.working_dir.trim()),
                )?;
                zap_neural_memory::bind_session_neural_agent(&session_id, &task.agent_id)?;
                let wd_clean = task.working_dir.trim().to_string();
                let proj_hash = host_for_prep.hash_path(Path::new(&wd_clean));
                let (_attempt_no, attempt_id) =
                    claim_task_attempt_conn(&store.conn, &tid, &session_id, &proj_hash, "running")?;

                let has_dependents = all_tasks
                    .iter()
                    .any(|other| other.depends_on.iter().any(|d| d == &tid));

                let dep_set: HashSet<String> = task.depends_on.iter().cloned().collect();
                let upstream_contracts: Vec<TeamContract> = all_contracts
                    .into_iter()
                    .filter(|c| dep_set.contains(&c.from_task_id))
                    .collect();

                let upstream_excerpts: Vec<(String, String)> = all_tasks
                    .into_iter()
                    .filter(|t| dep_set.contains(&t.id))
                    .map(|t| {
                        let mut summary = if !t.contract_summary.trim().is_empty() {
                            t.contract_summary
                        } else {
                            t.last_output_excerpt
                        };
                        if !t.touched_files.is_empty() {
                            let files = t.touched_files.join(", ");
                            summary = if summary.trim().is_empty() {
                                format!("Files: {files}")
                            } else {
                                format!("{summary}\nFiles: {files}")
                            };
                        }
                        (
                            format!("{} - {}", domain_badge_cn(&t.domain), t.title),
                            summary,
                        )
                    })
                    .collect();

                let actor = if task.agent_label.trim().is_empty() {
                    domain_badge_cn(&task.domain).to_string()
                } else {
                    format!("{} ({})", domain_badge_cn(&task.domain), task.agent_label)
                };

                let _ = append_event_conn(
                    &store.conn,
                    &rid,
                    "task_started",
                    &tid,
                    &actor,
                    &format!("开始执行「{}」→ 目录 `{}`", task.title, wd_clean),
                    "",
                );

                Ok::<_, String>((
                    task,
                    session_id,
                    attempt_id,
                    actor,
                    has_dependents,
                    upstream_contracts,
                    upstream_excerpts,
                    globals,
                    compact_prompts,
                ))
            })
        }
    })
    .await;

    let (
        task,
        session_id,
        attempt_id,
        actor,
        has_dependents,
        upstream_contracts,
        upstream_excerpts,
        globals,
        compact_prompts,
    ) = match prep {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            log_task_event(
                &run_id,
                &task_id,
                "orchestrator",
                "task_error",
                &format!("任务准备失败: {e}"),
                "",
            )
            .await;
            return false;
        }
        Err(e) => {
            log_task_event(
                &run_id,
                &task_id,
                "orchestrator",
                "task_error",
                &format!("任务调度异常: {e}"),
                "",
            )
            .await;
            return false;
        }
    };

    let effective_model = if !task.model.trim().is_empty() {
        task.model.trim().to_string()
    } else {
        run.lead_model.trim().to_string()
    };

    register_team_session_ctx(
        &session_id,
        ActiveTeamSessionCtx {
            run_id: run_id.clone(),
            task_id: task_id.clone(),
            domain: task.domain.clone(),
            actor_label: actor.clone(),
            provider_name: effective_model.clone(),
            read_only_mode: run.mode == "triage",
        },
    );

    let (initial_prompt, _, _) = render_worker_prompt_with_source(
        &run,
        &task,
        has_dependents,
        &upstream_contracts,
        &upstream_excerpts,
        &globals,
        None,
        compact_prompts,
    );
    let session_title = format!("🤝 [{}] {}", domain_badge_cn(&task.domain), task.title);
    let working_dir = PathBuf::from(task.working_dir.trim());

    let mut all_touched_files: Vec<String> = task.touched_files.clone();
    let mut current_repair_count = task.repair_count;
    let max_repairs = if run.mode == "pipeline" && run.auto_verify {
        run.max_repairs.min(HARD_CAP_MAX_REPAIRS)
    } else {
        0
    };

    let effective_images: Vec<TeamImage> = if !task.images.is_empty() {
        task.images.clone()
    } else {
        run.goal_images.clone()
    };
    let is_revision_round = !task.revision_note.trim().is_empty();
    let mut next_prompt = initial_prompt;

    let final_ok = loop {
        if handle.cancel.is_cancelled() {
            break false;
        }

        let pre_contract_sig: Option<(String, String, String)> = if is_revision_round {
            let rid_c = run_id.clone();
            let tid_c = task_id.clone();
            tokio::task::spawn_blocking(move || {
                with_team_store(|store| {
                    Ok(load_contracts_for_run_conn(&store.conn, &rid_c)?
                        .into_iter()
                        .find(|c| c.from_task_id == tid_c)
                        .map(|c| (c.title, c.schema_or_api, c.created_at)))
                })
            })
            .await
            .ok()
            .and_then(Result::ok)
            .flatten()
        } else {
            None
        };

        let turn_res = run_native_session_turn(
            &host,
            &handle,
            &run,
            &session_id,
            &session_title,
            &working_dir,
            (!effective_model.is_empty()).then_some(effective_model.as_str()),
            &next_prompt,
            run.mode == "triage",
            &effective_images,
        )
        .await;

        if handle.cancel.is_cancelled() {
            break false;
        }

        let (assistant_output, turn_files) = match turn_res {
            Ok(v) => v,
            Err(err_msg) => {
                let _ = commit_task_outcome_fenced(
                    &task_id,
                    &attempt_id,
                    "failed",
                    &err_msg,
                    None,
                    "rejected",
                    &err_msg,
                    current_repair_count,
                    &all_touched_files,
                )
                .await;
                log_task_event(
                    &run_id,
                    &task_id,
                    &actor,
                    "task_failed",
                    &format!(
                        "「{}」执行出错：{}",
                        task.title,
                        truncate_chars_safe(&err_msg, 120)
                    ),
                    &err_msg,
                )
                .await;
                break false;
            }
        };

        for f in turn_files {
            if !all_touched_files.contains(&f) {
                all_touched_files.push(f);
            }
        }

        // 若是流水线编码模式且存在下游依赖，但智能体未主动调用 team_publish_contract，自动兜底提炼契约摘要
        // 若为定向/联动改造回合且成员未调用 team_publish_contract 更新既有契约，则强制用本轮输出覆盖更新契约，防止下游读到 v1 旧路径
        if run.mode == "pipeline"
            && (has_dependents || matches!(task.domain.as_str(), "sql" | "backend"))
        {
            let force_update = if is_revision_round {
                let rid_c = run_id.clone();
                let tid_c = task_id.clone();
                let post_contract_sig: Option<(String, String, String)> =
                    tokio::task::spawn_blocking(move || {
                        with_team_store(|store| {
                            Ok(load_contracts_for_run_conn(&store.conn, &rid_c)?
                                .into_iter()
                                .find(|c| c.from_task_id == tid_c)
                                .map(|c| (c.title, c.schema_or_api, c.created_at)))
                        })
                    })
                    .await
                    .ok()
                    .and_then(Result::ok)
                    .flatten();
                post_contract_sig == pre_contract_sig
            } else {
                false
            };
            ensure_fallback_contract(&run_id, &task, &assistant_output, force_update).await;
        }

        // 无需独立验收（排障会诊模式或用户关闭了自动验收）→ 直接标记完成
        if run.mode == "triage" || !run.auto_verify {
            let committed = commit_task_outcome_fenced(
                &task_id,
                &attempt_id,
                "completed",
                &assistant_output,
                None,
                "skipped",
                "",
                current_repair_count,
                &all_touched_files,
            )
            .await;
            if committed {
                log_task_event(
                    &run_id,
                    &task_id,
                    &actor,
                    "task_completed",
                    &format!("已完成「{}」", task.title),
                    &assistant_output,
                )
                .await;
            }
            break committed;
        }

        // 进入权责分离独立验收闸门（Verifier）
        let _ = set_status_fenced(&task_id, &attempt_id, "verifying").await;
        log_task_event(
            &run_id,
            &task_id,
            "verifier",
            "task_verifying",
            &format!("独立验收官正在核查「{}」的交付产物与代码契约…", task.title),
            "",
        )
        .await;

        let verdict = verify_task_delivery(
            &effective_model,
            &working_dir,
            &run,
            &task,
            &upstream_contracts,
            &assistant_output,
            &all_touched_files,
        )
        .await;

        if handle.cancel.is_cancelled() {
            break false;
        }

        if verdict.passed {
            let committed = commit_task_outcome_fenced(
                &task_id,
                &attempt_id,
                "completed",
                &assistant_output,
                None,
                "passed",
                &verdict.summary,
                current_repair_count,
                &all_touched_files,
            )
            .await;
            if committed {
                log_task_event(
                    &run_id,
                    &task_id,
                    "verifier",
                    "task_verified_pass",
                    &format!("验收通过「{}」：{}", task.title, verdict.summary),
                    "",
                )
                .await;
            }
            break committed;
        }

        // 验收未通过：检查是否仍有自动打回返工额度
        if current_repair_count < max_repairs {
            current_repair_count += 1;
            let _ = commit_task_outcome_fenced(
                &task_id,
                &attempt_id,
                "repairing",
                &assistant_output,
                None,
                "rejected",
                &verdict.repair_instructions,
                current_repair_count,
                &all_touched_files,
            )
            .await;
            log_task_event(
                &run_id,
                &task_id,
                "verifier",
                "task_repair_triggered",
                &format!(
                    "验收打回返工（第 {}/{} 次）：{}",
                    current_repair_count,
                    max_repairs,
                    truncate_chars_safe(&verdict.summary, 100)
                ),
                &verdict.repair_instructions,
            )
            .await;

            let (template, _) = resolve_effective_template(
                TPL_WORKER_REPAIR,
                Some(&task.domain),
                Some(&run.prompt_overrides),
                &globals,
            );
            next_prompt = render_template_single_pass(
                &template,
                &[
                    ("repair_attempt", &current_repair_count.to_string()),
                    ("max_repairs", &max_repairs.to_string()),
                    ("verifier_summary", &verdict.summary),
                    ("repair_instructions", &verdict.repair_instructions),
                    ("acceptance_criteria", &task.acceptance_criteria),
                    ("working_dir", &task.working_dir),
                ],
            );
            continue;
        } else {
            let feedback = format!("{}\n{}", verdict.summary, verdict.repair_instructions);
            let _ = commit_task_outcome_fenced(
                &task_id,
                &attempt_id,
                "failed",
                &assistant_output,
                None,
                "rejected",
                &feedback,
                current_repair_count,
                &all_touched_files,
            )
            .await;
            log_task_event(
                &run_id,
                &task_id,
                "verifier",
                "task_failed_gate",
                &format!(
                    "「{}」连续 {} 次自动返工仍未通过验收，触发熔断暂停人工介入",
                    task.title, max_repairs
                ),
                &feedback,
            )
            .await;
            break false;
        }
    };

    unregister_team_session_ctx(&session_id);
    final_ok
}

#[allow(clippy::too_many_arguments)]
async fn run_native_session_turn(
    host: &Arc<dyn TeamHost>,
    handle: &Arc<ActiveRunHandle>,
    run: &TeamRun,
    session_id: &str,
    session_title: &str,
    working_dir: &Path,
    provider_override: Option<&str>,
    prompt: &str,
    read_only_mode: bool,
    images: &[TeamImage],
) -> Result<(String, Vec<String>), String> {
    let globals = global_prompt_templates().await?;
    let wd_text = path_to_clean_str(working_dir);
    let hint = inspect_dir_profile(working_dir).to_hint();
    let (guard, _) = render_sys_guard_with_source(
        read_only_mode,
        &wd_text,
        session_id,
        &hint,
        Some(&run.prompt_overrides),
        &globals,
    );
    let child_cancel = handle.cancel.child_token();
    handle.register_child(child_cancel.clone());
    let req = HostChildTurnRequest {
        message: prompt.to_string(),
        working_dir: working_dir.to_path_buf(),
        provider: provider_override.map(str::to_string),
        session_id: session_id.to_string(),
        images: images
            .iter()
            .map(|i| (i.media_type.clone(), i.data.clone()))
            .collect(),
        extra_system_append: Some(guard),
        session_title: Some(session_title.to_string()),
    };
    host.run_child_turn(req, child_cancel).await
}

#[derive(Debug, Deserialize)]
struct VerifierVerdict {
    #[serde(default = "default_pass")]
    passed: bool,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    repair_instructions: String,
}

fn default_pass() -> bool {
    true
}

async fn verify_task_delivery(
    provider_name: &str,
    working_dir: &Path,
    run: &TeamRun,
    task: &TeamTask,
    upstream_contracts: &[TeamContract],
    worker_output: &str,
    touched_files: &[String],
) -> VerifierVerdict {
    if worker_output.trim().is_empty() && touched_files.is_empty() {
        return VerifierVerdict {
            passed: false,
            summary: "智能体未产生任何有效输出或代码变更".to_string(),
            repair_instructions: "请实际检查目标代码库并完成要求的改造，输出改动说明与契约。"
                .to_string(),
        };
    }

    let Ok(config) = crate::host::load_config() else {
        return VerifierVerdict {
            passed: true,
            summary: "已完成执行（配置读取跳过独立复核）".to_string(),
            repair_instructions: String::new(),
        };
    };

    let wd_owned = working_dir.to_path_buf();
    let files_owned = touched_files.to_vec();
    let tid_owned = task.id.clone();
    let rid_owned = task.run_id.clone();
    let (disk_snippets, published_ctr, compact_prompts) = tokio::task::spawn_blocking(move || {
        let compact =
            with_team_store(|s| Ok(load_config_conn(&s.conn).compact_prompts)).unwrap_or(true);
        let max_files = if compact { 3 } else { 5 };
        let max_file_chars = if compact { 600 } else { 1200 };
        let mut snippets = Vec::new();
        for f in files_owned.iter().take(max_files) {
            let p = Path::new(f);
            let full = if p.is_absolute() {
                p.to_path_buf()
            } else {
                wd_owned.join(p)
            };
            if let Ok(content) = std::fs::read_to_string(&full) {
                snippets.push(format!(
                    "File `{f}` ({} bytes):\n{}",
                    content.len(),
                    truncate_chars_safe(&content, max_file_chars)
                ));
            }
        }
        let ctr = with_team_store(|s| {
            Ok(load_contracts_for_run_conn(&s.conn, &rid_owned)?
                .into_iter()
                .find(|c| c.from_task_id == tid_owned))
        })
        .ok()
        .flatten();
        (snippets.join("\n\n"), ctr, compact)
    })
    .await
    .unwrap_or_default();

    let prov_opt = (!provider_name.trim().is_empty()).then_some(provider_name.trim());
    let mut rt = zap_coding::CodingRuntimeConfig::from_config(
        &config,
        working_dir,
        prov_opt,
        None,
        false,
        false,
    );
    rt.mcp = false;
    rt.max_tokens = Some(4096);
    let coding_cfg = rt.agent_config();
    let options = ChatOptions {
        max_tokens: Some(4096),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory = crate::host::coding_provider_factory();
    let sid = format!("team-verifier-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid))).await
    else {
        return VerifierVerdict {
            passed: true,
            summary: "已完成交付".to_string(),
            repair_instructions: String::new(),
        };
    };

    let upstream_str = if upstream_contracts.is_empty() {
        "(none)".to_string()
    } else {
        let max_ctr = if compact_prompts { 500 } else { 800 };
        upstream_contracts
            .iter()
            .map(|c| {
                format!(
                    "[{}] {}: {}",
                    c.domain,
                    c.title,
                    truncate_chars_safe(&c.schema_or_api, max_ctr)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let ctr_str = published_ctr
        .map(|c| {
            let body = if compact_prompts {
                truncate_chars_safe(&c.schema_or_api, 700)
            } else {
                c.schema_or_api
            };
            format!("{}\n{}", c.title, body)
        })
        .unwrap_or_else(|| "(none published)".to_string());

    let approved_sql_str = if run.sql_schema_draft.trim().is_empty() {
        "(none)".to_string()
    } else if compact_prompts {
        truncate_chars_safe(&extract_compact_sql_ddl(&run.sql_schema_draft), 900)
    } else {
        truncate_chars_safe(&run.sql_schema_draft, 1400)
    };
    let locked_spec_str = if !task.revision_note.trim().is_empty() {
        format!(
            "[增量改造准绳，以本要求与最新上游契约为准]: {}\n\n[早期基准约定仅供参考]: {}",
            truncate_chars_safe(&task.revision_note, 700),
            if run.interface_spec.trim().is_empty() {
                "(none)".to_string()
            } else {
                truncate_chars_safe(&run.interface_spec, 800)
            }
        )
    } else if run.interface_spec.trim().is_empty() {
        "(none)".to_string()
    } else if compact_prompts {
        truncate_chars_safe(&run.interface_spec, 1000)
    } else {
        truncate_chars_safe(&run.interface_spec, 1600)
    };
    let verifier_desc_str = if !task.revision_note.trim().is_empty() {
        format!(
            "【本次增量改造要求】：{}\n【原任务背景】：{}",
            truncate_chars_safe(&task.revision_note, 600),
            truncate_chars_safe(&task.description, 400)
        )
    } else {
        task.description.clone()
    };
    let worker_out_str =
        truncate_chars_safe(worker_output, if compact_prompts { 1200 } else { 2200 });

    let prompt = match execution_prompt(
        run,
        TPL_VERIFIER_GATE,
        Some(&task.domain),
        &[
            ("task_title", &task.title),
            ("domain", &task.domain),
            ("task_description", &verifier_desc_str),
            ("acceptance_criteria", &task.acceptance_criteria),
            ("approved_sql", &approved_sql_str),
            ("locked_interface_spec", &locked_spec_str),
            ("upstream_contracts", &upstream_str),
            ("published_contract", &ctr_str),
            ("touched_files_snippets", &disk_snippets),
            ("worker_output", &worker_out_str),
        ],
    )
    .await
    {
        Ok(prompt) => prompt,
        Err(e) => {
            return VerifierVerdict {
                passed: false,
                summary: format!("Prompt configuration failed: {e}"),
                repair_instructions: "Restore template configuration access before retrying."
                    .to_string(),
            }
        }
    };

    let msgs = vec![Message::user(prompt)];
    let fut = async {
        let mut stream = match provider.chat_stream(&msgs, &[], &options).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("[zap-agent-teams] verify_task_delivery chat_stream failed: {e:?}");
                return None;
            }
        };
        let mut raw = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(d) => raw.push_str(&d),
                StreamEvent::Done { .. } => break,
                StreamEvent::Error(e) => {
                    tracing::warn!("[zap-agent-teams] verify_task_delivery stream error: {e:?}");
                    break;
                }
                _ => {}
            }
        }
        extract_json_object(&raw).and_then(|j| serde_json::from_str::<VerifierVerdict>(j).ok())
    };

    match tokio::time::timeout(Duration::from_secs(60), fut).await {
        Ok(Some(v)) => v,
        _ => VerifierVerdict {
            passed: true,
            summary: "已完成交付产物核查".to_string(),
            repair_instructions: String::new(),
        },
    }
}

async fn ensure_fallback_contract(
    run_id: &str,
    task: &TeamTask,
    assistant_output: &str,
    force_update: bool,
) {
    if assistant_output.trim().is_empty() {
        return;
    }
    let rid = run_id.to_string();
    let tid = task.id.clone();
    let dom = task.domain.clone();
    let title = if !task.revision_note.trim().is_empty() {
        format!("{} · 最新改造契约摘要", task.title)
    } else {
        format!("{} 交付摘要", task.title)
    };
    let excerpt = truncate_chars_safe(assistant_output, 2000);
    let _ = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let existing = load_contracts_for_run_conn(&store.conn, &rid)?;
            if force_update || !existing.iter().any(|c| c.from_task_id == tid) {
                let _ = upsert_contract_conn(
                    &store.conn,
                    &rid,
                    &tid,
                    &dom,
                    &title,
                    &excerpt,
                    if force_update {
                        "自动提取自本轮改造输出（覆盖早期契约）"
                    } else {
                        "自动提取自上游子任务最终回复"
                    },
                );
            }
            Ok(())
        })
    })
    .await;
}

async fn finalize_successful_run(run_id: &str, run: &TeamRun) -> bool {
    let rid = run_id.to_string();
    let data = tokio::task::spawn_blocking({
        let rid = rid.clone();
        move || {
            with_team_store(|store| {
                Ok::<_, String>((
                    load_tasks_for_run_conn(&store.conn, &rid)?,
                    load_contracts_for_run_conn(&store.conn, &rid)?,
                ))
            })
        }
    })
    .await
    .ok()
    .and_then(Result::ok);

    let (tasks, contracts) = data.unwrap_or_default();

    let latest_run = tokio::task::spawn_blocking({
        let rid_q = rid.clone();
        move || with_team_store(|s| load_run_conn(&s.conn, &rid_q))
    })
    .await
    .ok()
    .and_then(Result::ok)
    .flatten()
    .unwrap_or_else(|| run.clone());

    let delivery_sql = extract_delivery_sql(&latest_run, &contracts);

    let synthesis = if latest_run.mode == "triage" {
        synthesize_triage_report(&latest_run, &tasks).await
    } else {
        let mut lines = vec![format!(
            "### ✅ 跨项目协作交付完成：《{}》",
            latest_run.title
        )];
        for t in &tasks {
            let files_note = if t.touched_files.is_empty() {
                "无文件写入".to_string()
            } else {
                format!(
                    "改动 {} 个文件 ({})",
                    t.touched_files.len(),
                    t.touched_files.join(", ")
                )
            };
            lines.push(format!(
                "- **[{}] {}**：{}（返工 {} 次）",
                if t.project_name.is_empty() {
                    domain_badge_cn(&t.domain)
                } else {
                    &t.project_name
                },
                t.title,
                files_note,
                t.repair_count
            ));
        }
        if !contracts.is_empty() {
            lines.push(format!(
                "\n共沉淀 **{} 份** 跨项目表结构/接口契约，各专属智能体记忆已同步异步归档。",
                contracts.len()
            ));
        }
        lines.join("\n")
    };

    let rid_save = run_id.to_string();
    let syn_save = synthesis.clone();
    let sql_save = delivery_sql.clone();
    let is_triage = latest_run.mode == "triage";
    let final_stage = if is_triage { "triage" } else { "completed" };
    tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let current_tasks = load_tasks_for_run_conn(&store.conn, &rid_save)?;
            if current_tasks.iter().any(|t| t.status != "completed") {
                return Ok(false);
            }
            update_run_delivery_conn(
                &store.conn,
                &rid_save,
                "completed",
                final_stage,
                &sql_save,
                &syn_save,
            )?;
            let msg = if is_triage {
                "三方跨库联合排障会诊已完成！已生成跨层根因定位报告，可一键转化为修复任务。"
            } else {
                "全部项目并发开发与交叉验收顺利完成！已生成《一键交付卡》并沉淀至专属记忆。"
            };
            let _ = append_event_conn(
                &store.conn,
                &rid_save,
                "run_completed",
                "",
                "orchestrator",
                msg,
                &syn_save,
            );
            Ok(true)
        })
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or(false)
}

fn extract_delivery_sql(run: &TeamRun, contracts: &[TeamContract]) -> String {
    let mut blocks = Vec::new();
    let source = if !run.sql_schema_draft.trim().is_empty() {
        run.sql_schema_draft.as_str()
    } else {
        ""
    };
    if !source.is_empty() {
        let mut rest = source;
        while let Some(start) = rest.find("```sql") {
            let after = &rest[start + 6..];
            if let Some(end) = after.find("```") {
                let block = after[..end].trim();
                if !block.is_empty() {
                    blocks.push(block.to_string());
                }
                rest = &after[end + 3..];
            } else {
                break;
            }
        }
        if blocks.is_empty()
            && (source.contains("CREATE TABLE")
                || source.contains("ALTER TABLE")
                || source.contains("INSERT INTO"))
        {
            blocks.push(source.trim().to_string());
        }
    }
    if blocks.is_empty() {
        for c in contracts {
            if c.domain == "sql" && !c.schema_or_api.trim().is_empty() {
                blocks.push(c.schema_or_api.trim().to_string());
            }
        }
    }
    blocks.join("\n\n-- ----------------------------------------\n\n")
}

async fn synthesize_triage_report(run: &TeamRun, tasks: &[TeamTask]) -> String {
    let findings = tasks
        .iter()
        .map(|t| {
            format!(
                "## [{}] {} (仓库: `{}`)\n{}",
                domain_badge_cn(&t.domain),
                t.title,
                t.working_dir,
                truncate_chars_safe(&t.last_output_excerpt, 2000)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let fallback = format!("### 🩺 跨库联合会诊汇总报告\n\n{findings}");

    let Ok(config) = crate::host::load_config() else {
        return fallback;
    };
    let prov_opt = (!run.lead_model.trim().is_empty()).then_some(run.lead_model.trim());
    let wd = tasks
        .first()
        .map(|t| PathBuf::from(&t.working_dir))
        .unwrap_or_else(|| PathBuf::from("."));
    let mut rt =
        zap_coding::CodingRuntimeConfig::from_config(&config, &wd, prov_opt, None, false, false);
    rt.mcp = false;
    rt.max_tokens = Some(6144);
    let coding_cfg = rt.agent_config();
    let options = ChatOptions {
        max_tokens: Some(6144),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };
    let factory = crate::host::coding_provider_factory();
    let sid = format!("team-triage-syn-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid))).await
    else {
        return fallback;
    };

    let prompt = match execution_prompt(
        run,
        TPL_TRIAGE_SYNTHESIS,
        None,
        &[("user_goal", &run.user_goal), ("findings", &findings)],
    )
    .await
    {
        Ok(prompt) => prompt,
        Err(e) => return format!("Template configuration failed: {e}//n//n{fallback}"),
    };

    let msgs = vec![Message::user(prompt)];
    let fut = async {
        let mut stream = provider.chat_stream(&msgs, &[], &options).await.ok()?;
        let mut text = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Done { .. } | StreamEvent::Error(_) => break,
                _ => {}
            }
        }
        (!text.trim().is_empty()).then_some(text)
    };

    match tokio::time::timeout(Duration::from_secs(75), fut).await {
        Ok(Some(report)) => report,
        _ => fallback,
    }
}

async fn mark_run_failed(run_id: &str, reason: &str) {
    let rid = run_id.to_string();
    let msg = reason.to_string();
    let _ = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            set_run_status_conn(&store.conn, &rid, "failed", None)?;
            let _ = append_event_conn(
                &store.conn,
                &rid,
                "run_failed",
                "",
                "orchestrator",
                &msg,
                "",
            );
            Ok(())
        })
    })
    .await;
}

#[allow(clippy::too_many_arguments)]
async fn commit_task_outcome_fenced(
    task_id: &str,
    attempt_id: &str,
    status: &str,
    excerpt: &str,
    contract_summary: Option<&str>,
    verify_verdict: &str,
    verify_feedback: &str,
    repair_count: u32,
    touched_files: &[String],
) -> bool {
    let tid = task_id.to_string();
    let aid = attempt_id.to_string();
    let st = status.to_string();
    let ex = excerpt.to_string();
    let cs = contract_summary.map(str::to_string);
    let vv = verify_verdict.to_string();
    let vf = verify_feedback.to_string();
    let files = touched_files.to_vec();
    tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            finish_task_fenced_conn(
                &store.conn,
                &tid,
                &aid,
                &st,
                &ex,
                cs.as_deref(),
                &vv,
                &vf,
                repair_count,
                &files,
            )
        })
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or(false)
}

async fn set_status_fenced(task_id: &str, attempt_id: &str, status: &str) -> bool {
    let tid = task_id.to_string();
    let aid = attempt_id.to_string();
    let st = status.to_string();
    tokio::task::spawn_blocking(move || {
        with_team_store(|store| set_task_status_fenced_conn(&store.conn, &tid, &aid, &st))
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or(false)
}

async fn log_task_event(
    run_id: &str,
    task_id: &str,
    actor: &str,
    kind: &str,
    summary: &str,
    detail: &str,
) {
    let rid = run_id.to_string();
    let tid = task_id.to_string();
    let act = actor.to_string();
    let k = kind.to_string();
    let s = summary.to_string();
    let d = detail.to_string();
    let _ = tokio::task::spawn_blocking(move || {
        with_team_store(|store| append_event_conn(&store.conn, &rid, &k, &tid, &act, &s, &d))
    })
    .await;
}
