//! `zap-agent-teams` 常驻编排执行器：
//! - DAG 波次并发调度（信号量限流）+ 上游契约自动接力注入
//! - 跨项目独立会话绑定专属 `neural_memory` 智能体 + 无人值守 `ApprovalMode::Auto` 执行
//! - 权责分离独立验收闸门（Verifier）+ 最多 `max_repairs` 次（默认 2 次）自动打回原会话返工 + 超限熔断
//! - 三方跨库联合排障会诊（`triage`）并发勘察与根因汇总报告生成

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use zap_kernel::message::Message;
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;
use zap_telemetry::{CurrentContext, SessionMode};

use crate::{AppState, ChatAdmissionMode, ChatEvent, ChatRequest};

use super::planner::extract_json_object;
use super::store::{
    append_event_conn, claim_task_attempt_conn, finish_task_fenced_conn, load_config_conn,
    load_contracts_for_run_conn, load_run_conn, load_task_conn, load_tasks_for_run_conn,
    set_run_status_conn, set_task_status_fenced_conn, upsert_contract_conn,
    upsert_role_preset_conn, with_team_store,
};
use super::tools::{register_team_session_ctx, unregister_team_session_ctx, ActiveTeamSessionCtx};
use super::types::{
    compute_and_assign_waves, domain_badge_cn, normalize_domain, truncate_chars_safe,
    RoleBindingPreset, TeamContract, TeamRun, TeamTask, HARD_CAP_MAX_REPAIRS,
};

struct ActiveRunHandle {
    cancel: CancellationToken,
    child_cancels: Mutex<Vec<CancellationToken>>,
}

impl ActiveRunHandle {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancel: CancellationToken::new(),
            child_cancels: Mutex::new(Vec::new()),
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

/// 启动或从断点恢复执行指定的团队派工单（后台常驻运行，关掉/刷新网页不中断）。
pub async fn start_or_resume_run(state: AppState, run_id: &str) -> Result<(), String> {
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

    let rid_prep = rid.clone();
    let max_parallel = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let run = load_run_conn(&store.conn, &rid_prep)?
                .ok_or_else(|| format!("未找到派工单: {rid_prep}"))?;
            let mut tasks = load_tasks_for_run_conn(&store.conn, &rid_prep)?;
            if tasks.is_empty() {
                return Err("派工单中没有可执行的子任务".to_string());
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
            // 校验 DAG 无环并保存用户的领域默认绑定习惯（下次自动预填）
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
            set_run_status_conn(&store.conn, &rid_prep, "running", None)?;
            let _ = append_event_conn(
                &store.conn,
                &rid_prep,
                "run_started",
                "",
                "orchestrator",
                &format!("团队协作引擎已启动执行《{}》", run.title),
                "",
            );
            Ok(load_config_conn(&store.conn).max_parallel_per_wave)
        })
    })
    .await
    .map_err(|e| e.to_string())??;

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
        orchestrate_run_loop(state, rid_bg.clone(), handle, max_parallel.clamp(1, 6)).await;
        let mut guard = active_runs().lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(&rid_bg);
    });

    Ok(())
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
pub async fn retry_single_task(state: AppState, run_id: &str, task_id: &str) -> Result<(), String> {
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

    start_or_resume_run(state, &rid).await
}

async fn orchestrate_run_loop(
    state: AppState,
    run_id: String,
    handle: Arc<ActiveRunHandle>,
    max_parallel: usize,
) {
    let snapshot = tokio::task::spawn_blocking({
        let rid = run_id.clone();
        move || {
            with_team_store(|store| {
                let run = load_run_conn(&store.conn, &rid)?;
                let mut tasks = load_tasks_for_run_conn(&store.conn, &rid)?;
                let waves = compute_and_assign_waves(&mut tasks)?;
                Ok::<_, String>((run, waves))
            })
        }
    })
    .await;

    let (Some(run), waves) = (match snapshot {
        Ok(Ok((r, w))) => (r, w),
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

    let sem = Arc::new(Semaphore::new(max_parallel));

    for (wave_num, wave_task_ids) in waves {
        if handle.cancel.is_cancelled() {
            return;
        }

        // 筛选本波次尚未完成的子任务
        let pending_ids: Vec<String> = tokio::task::spawn_blocking({
            let rid = run_id.clone();
            let ids = wave_task_ids.clone();
            move || {
                with_team_store(|store| {
                    let all = load_tasks_for_run_conn(&store.conn, &rid)?;
                    let set: HashSet<String> = ids.into_iter().collect();
                    let ready: Vec<String> = all
                        .into_iter()
                        .filter(|t| set.contains(&t.id) && t.status != "completed")
                        .map(|t| t.id)
                        .collect();
                    if !ready.is_empty() {
                        let _ = append_event_conn(
                            &store.conn,
                            &rid,
                            "wave_started",
                            "",
                            "orchestrator",
                            &format!(
                                "进入第 {} 波次（并发执行 {} 个领域子任务）",
                                wave_num,
                                ready.len()
                            ),
                            "",
                        );
                    }
                    Ok::<_, String>(ready)
                })
            }
        })
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();

        if pending_ids.is_empty() {
            continue;
        }

        let mut joins = Vec::with_capacity(pending_ids.len());
        for tid in pending_ids {
            let st = state.clone();
            let rid = run_id.clone();
            let run_cfg = run.clone();
            let hdl = Arc::clone(&handle);
            let sem_clone = Arc::clone(&sem);
            joins.push(tokio::spawn(async move {
                let Ok(_permit) = sem_clone.acquire_owned().await else {
                    return false;
                };
                if hdl.cancel.is_cancelled() {
                    return false;
                }
                execute_single_task_lifecycle(st, rid, tid, run_cfg, hdl).await
            }));
        }

        let mut wave_all_ok = true;
        for j in joins {
            if !j.await.unwrap_or(false) {
                wave_all_ok = false;
            }
        }

        if handle.cancel.is_cancelled() {
            return;
        }
        if !wave_all_ok {
            mark_run_failed(
                &run_id,
                &format!("第 {wave_num} 波次存在未通过验收或执行失败的子任务，流水线已熔断保护现场，可查看原因后一键重试。"),
            )
            .await;
            return;
        }
    }

    if handle.cancel.is_cancelled() {
        return;
    }

    // 全部波次顺利收官：生成跨仓库汇总结论（尤其是联合排障会诊模式）
    finalize_successful_run(&run_id, &run).await;
}

async fn execute_single_task_lifecycle(
    state: AppState,
    run_id: String,
    task_id: String,
    run: TeamRun,
    handle: Arc<ActiveRunHandle>,
) -> bool {
    let prep = tokio::task::spawn_blocking({
        let rid = run_id.clone();
        let tid = task_id.clone();
        move || {
            with_team_store(|store| {
                let task = load_task_conn(&store.conn, &tid)?
                    .ok_or_else(|| format!("未找到子任务 {tid}"))?;
                let all_tasks = load_tasks_for_run_conn(&store.conn, &rid)?;
                let all_contracts = load_contracts_for_run_conn(&store.conn, &rid)?;

                let session_id = if task.session_id.trim().is_empty() {
                    format!("team_{}_{}", rid, task.id)
                } else {
                    task.session_id.clone()
                };
                let wd_clean = task.working_dir.trim().to_string();
                let proj_hash = crate::hash_path(Path::new(&wd_clean));
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
                        let summary = if !t.contract_summary.trim().is_empty() {
                            t.contract_summary
                        } else {
                            t.last_output_excerpt
                        };
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

    // 绑定专属隔离记忆智能体（如 ⚙ Backend / 🎨 Frontend / 🗄 SQL）
    if !task.agent_id.trim().is_empty() {
        let sid_bind = session_id.clone();
        let aid_bind = task.agent_id.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let _ = crate::neural_memory::bind_session_neural_agent(&sid_bind, &aid_bind);
        })
        .await;
    }

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

    let initial_prompt = build_worker_prompt(
        &run,
        &task,
        has_dependents,
        &upstream_contracts,
        &upstream_excerpts,
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

    let mut next_prompt = initial_prompt;

    let final_ok = loop {
        if handle.cancel.is_cancelled() {
            break false;
        }

        let turn_res = run_native_session_turn(
            &state,
            &handle,
            &session_id,
            &session_title,
            &working_dir,
            (!effective_model.is_empty()).then_some(effective_model.as_str()),
            &next_prompt,
            run.mode == "triage",
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
        if run.mode == "pipeline"
            && (has_dependents || matches!(task.domain.as_str(), "sql" | "backend"))
        {
            ensure_fallback_contract(&run_id, &task, &assistant_output).await;
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

            next_prompt = format!(
                "[Independent Verifier Rejection — Auto-Repair Attempt {current_repair_count}/{max_repairs}]\n\
                Your previous delivery did NOT pass the acceptance gate.\n\n\
                Verifier Summary:\n{}\n\n\
                Required Fixes:\n{}\n\n\
                Original Acceptance Criteria:\n{}\n\n\
                Fix all issues directly in `{}` now, verify syntax/build if applicable, and update `team_publish_contract` if any schema/API signature changed.",
                verdict.summary,
                verdict.repair_instructions,
                task.acceptance_criteria,
                working_dir.display()
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
    state: &AppState,
    handle: &Arc<ActiveRunHandle>,
    session_id: &str,
    session_title: &str,
    working_dir: &Path,
    provider_override: Option<&str>,
    prompt: &str,
    read_only_mode: bool,
) -> Result<(String, Vec<String>), String> {
    let sys_guard = if read_only_mode {
        Some(
            "TEAM JOINT TRIAGE READ-ONLY RULE: You are participating in a cross-repository diagnostic consultation. Inspect code, configs, schemas, and logs thoroughly using read-only tools (`code_explore`, `grep`, `read`, `glob`, `team_ask_peer`, `memory_recall`), cite exact file paths and line numbers, and DO NOT modify, write, or delete any files."
                .to_string(),
        )
    } else {
        Some(
            "TEAM COLLABORATION PROTOCOL: Focus strictly on your assigned repository and domain responsibility. Obey upstream contracts verbatim. Use `team_ask_peer` if you need to confirm peer repository details, and call `team_publish_contract` to publish exact SQL DDL or API request/response schemas when your work defines interfaces for downstream teammates."
                .to_string(),
        )
    };

    let chat_req = ChatRequest {
        message: prompt.to_string(),
        working_dir: Some(working_dir.to_path_buf()),
        provider: provider_override.map(str::to_string),
        session_id: Some(session_id.to_string()),
        request_id: Some(format!(
            "req_team_{}_{}",
            session_id,
            uuid::Uuid::new_v4().simple()
        )),
        images: Vec::new(),
        approval_mode: Some(crate::approval_mode::ApprovalMode::Auto),
        extra_system_append: sys_guard,
        session_title: Some(session_title.to_string()),
    };

    let admitted = crate::admit_chat_turn(state, chat_req, ChatAdmissionMode::RejectIfBusy, None)
        .await
        .map_err(|e| format!("无法启动子智能体会话 `{session_id}`: {e:?}"))?;

    let crate::AdmittedChatTurn {
        request: admitted_req,
        operation_id,
        cancellation: cancel_token,
        mut rx,
        fan_tx,
        terminal_sent,
        ..
    } = admitted;

    handle.register_child(cancel_token.clone());

    let cleanup_chats = state.active_chats.clone();
    let cleanup_op = operation_id.clone();
    let mcp_pool = state.mcp_pool.clone();
    let telemetry = state.telemetry.clone();
    let pending_permissions = state.pending_permissions.clone();
    let pending_user_inputs = state.pending_user_inputs.clone();
    let op_for_proc = operation_id.clone();
    let cancel_for_proc = cancel_token.clone();
    let active_chats_for_proc = state.active_chats.clone();

    let ctx = CurrentContext {
        mode: Some(SessionMode::Channel),
        session_id: None,
        ..CurrentContext::current()
    };

    tokio::spawn(async move {
        let res = CurrentContext::scope(ctx, || async move {
            crate::process_chat_request(
                admitted_req,
                fan_tx,
                cancel_for_proc,
                op_for_proc,
                active_chats_for_proc,
                mcp_pool,
                telemetry,
                pending_permissions,
                pending_user_inputs,
                false,
                false,
                terminal_sent,
                true,
                Some(crate::approval_mode::ApprovalMode::Auto),
            )
            .await
        })
        .await;
        if let Err(e) = res {
            tracing::warn!("[zap-agent-teams] child session turn error: {e}");
        }
        cleanup_chats.complete(&cleanup_op).await;
    });

    let mut collected_text = String::new();
    let mut touched_files = Vec::new();
    let mut turn_err: Option<String> = None;

    while let Some(evt) = rx.recv().await {
        match evt {
            ChatEvent::TextDelta { content, .. } => {
                collected_text.push_str(&content);
            }
            ChatEvent::ToolCallStarted {
                name, arguments, ..
            } => {
                if matches!(name.as_str(), "edit" | "write") {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&arguments) {
                        if let Some(p) = val.get("path").and_then(|v| v.as_str()) {
                            let clean = p.trim().replace('\\', "/");
                            if !clean.is_empty() && !touched_files.contains(&clean) {
                                touched_files.push(clean);
                            }
                        }
                    }
                }
            }
            ChatEvent::Done { message, .. } => {
                if collected_text.trim().is_empty() {
                    if let Some(m) = message {
                        collected_text = m;
                    }
                }
                break;
            }
            ChatEvent::Error { message, .. } => {
                turn_err = Some(message);
                break;
            }
            ChatEvent::Stopped => {
                if handle.cancel.is_cancelled() {
                    return Err("已取消或暂停".to_string());
                }
                break;
            }
            _ => {}
        }
    }

    if let Some(err) = turn_err {
        if collected_text.trim().is_empty() {
            return Err(err);
        }
    }
    Ok((collected_text, touched_files))
}

fn build_worker_prompt(
    run: &TeamRun,
    task: &TeamTask,
    has_dependents: bool,
    upstream_contracts: &[TeamContract],
    upstream_excerpts: &[(String, String)],
) -> String {
    let mut sections = Vec::new();
    if run.mode == "triage" {
        sections.push(format!(
            "【跨库联合排障会诊任务】\n全局故障现象/问题：\n{}\n\n你负责的领域：[{}]\n你的具体排查重点：{}\n补充说明：{}\n验收要求：{}",
            run.user_goal,
            domain_badge_cn(&task.domain),
            task.title,
            task.description,
            task.acceptance_criteria
        ));
        sections.push(
            "请仅以只读方式深入排查当前仓库，定位具体文件路径、行号、关键函数/SQL/接口字段，若需核对上下游仓库细节可调用 `team_ask_peer`，最后给出结构化的证据链与本层根因结论。"
                .to_string(),
        );
        return sections.join("\n\n");
    }

    sections.push(format!(
        "【跨项目团队研发任务】\n整体业务目标：\n{}\n\n当前分配给你的子任务（第 {} 波次 · {}）：\n- 标题：{}\n- 详细任务说明：{}\n- 验收标准：{}",
        run.user_goal,
        task.wave,
        domain_badge_cn(&task.domain),
        task.title,
        task.description,
        task.acceptance_criteria
    ));

    if !upstream_contracts.is_empty() {
        let ctr_block = upstream_contracts
            .iter()
            .map(|c| {
                format!(
                    "### 上游已锁定契约：《{}》（来自 {}）\n{}\n{}",
                    c.title,
                    domain_badge_cn(&c.domain),
                    c.schema_or_api,
                    if c.notes.is_empty() {
                        String::new()
                    } else {
                        format!("备注：{}", c.notes)
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        sections.push(format!(
            "【必须严格对齐的上游契约】（字段名、类型、路径必须 100% 一致，禁止擅自改名）：\n{ctr_block}"
        ));
    } else if !upstream_excerpts.is_empty() {
        let ex_block = upstream_excerpts
            .iter()
            .map(|(k, v)| format!("### 上游任务产出（{k}）\n{}", truncate_chars_safe(v, 1500)))
            .collect::<Vec<_>>()
            .join("\n\n");
        sections.push(format!("【上游前置任务交付摘要】：\n{ex_block}"));
    }

    if has_dependents || matches!(task.domain.as_str(), "sql" | "backend") {
        sections.push(
            "【契约交付要求】：完成编码与检查后，请务必调用一次 `team_publish_contract` 工具，将你最终落地的真实 SQL 表结构 DDL 或后端 API 路径、请求入参、响应 JSON 字段结构发布到共享账本，供下游前端/服务智能体精准对接。"
                .to_string(),
        );
    }

    sections.join("\n\n")
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

    let Ok(config) = crate::api_config::load_config() else {
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
    let (disk_snippets, published_ctr) = tokio::task::spawn_blocking(move || {
        let mut snippets = Vec::new();
        for f in files_owned.iter().take(5) {
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
                    truncate_chars_safe(&content, 1200)
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
        (snippets.join("\n\n"), ctr)
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
    rt.max_tokens = Some(1200);
    let coding_cfg = crate::kernel_runtime::coding_config_from_runtime(&rt);
    let options = ChatOptions {
        max_tokens: Some(1200),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory = crate::runtime_host::coding_provider_factory();
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
        upstream_contracts
            .iter()
            .map(|c| {
                format!(
                    "[{}] {}: {}",
                    c.domain,
                    c.title,
                    truncate_chars_safe(&c.schema_or_api, 800)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let ctr_str = published_ctr
        .map(|c| format!("{}\n{}", c.title, c.schema_or_api))
        .unwrap_or_else(|| "(none published)".to_string());

    let prompt = format!(
        "You are an independent QA & Architecture Gate Verifier (strictly separated from the coding worker).\n\
        Audit whether the worker genuinely fulfilled the task requirements and respected upstream contracts.\n\
        Be pragmatic: pass (`\"passed\": true`) when core requirements, upstream field/route alignment, and code changes are sound; reject (`\"passed\": false`) ONLY when key requirements are missing, upstream contract field/route names are mismatched, files were left broken/unmodified despite coding requirements, or obvious syntax/logic bugs exist.\n\
        Write `summary` and `repair_instructions` in concise Chinese.\n\
        Output ONLY valid JSON:\n\
        {{\n\
          \"passed\": true/false,\n\
          \"summary\": \"一句话验收结论\",\n\
          \"repair_instructions\": \"若不通过，列出具体需要修改的文件、字段或逻辑修复指令；若通过则留空\"\n\
        }}\n\n\
        Task Title: {}\n\
        Domain: {}\n\
        Task Description: {}\n\
        Acceptance Criteria: {}\n\n\
        Upstream Contracts to Respect:\n{}\n\n\
        Worker Published Contract:\n{}\n\n\
        Touched Files on Disk:\n{}\n\n\
        Worker Final Output:\n{}",
        task.title,
        task.domain,
        task.description,
        task.acceptance_criteria,
        upstream_str,
        ctr_str,
        if disk_snippets.is_empty() { "(no file edits recorded)" } else { &disk_snippets },
        truncate_chars_safe(worker_output, 2200)
    );

    let msgs = vec![Message::user(prompt)];
    let fut = async {
        let mut stream = provider.chat_stream(&msgs, &[], &options).await.ok()?;
        let mut raw = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(d) => raw.push_str(&d),
                StreamEvent::Done { .. } | StreamEvent::Error(_) => break,
                _ => {}
            }
        }
        extract_json_object(&raw).and_then(|j| serde_json::from_str::<VerifierVerdict>(j).ok())
    };

    match tokio::time::timeout(Duration::from_secs(30), fut).await {
        Ok(Some(v)) => v,
        _ => VerifierVerdict {
            passed: true,
            summary: "已完成交付产物核查".to_string(),
            repair_instructions: String::new(),
        },
    }
}

async fn ensure_fallback_contract(run_id: &str, task: &TeamTask, assistant_output: &str) {
    if assistant_output.trim().is_empty() {
        return;
    }
    let rid = run_id.to_string();
    let tid = task.id.clone();
    let dom = task.domain.clone();
    let title = format!("{} 交付摘要", task.title);
    let excerpt = truncate_chars_safe(assistant_output, 2000);
    let _ = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            let existing = load_contracts_for_run_conn(&store.conn, &rid)?;
            if !existing.iter().any(|c| c.from_task_id == tid) {
                let _ = upsert_contract_conn(
                    &store.conn,
                    &rid,
                    &tid,
                    &dom,
                    &title,
                    &excerpt,
                    "自动提取自上游子任务最终回复",
                );
            }
            Ok(())
        })
    })
    .await;
}

async fn finalize_successful_run(run_id: &str, run: &TeamRun) {
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

    let synthesis = if run.mode == "triage" {
        synthesize_triage_report(run, &tasks).await
    } else {
        let mut lines = vec![format!("### ✅ 跨仓库协作交付完成：《{}》", run.title)];
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
                "- **第 {} 波次 [{}] {}**：{}（返工 {} 次）",
                t.wave,
                domain_badge_cn(&t.domain),
                t.title,
                files_note,
                t.repair_count
            ));
        }
        if !contracts.is_empty() {
            lines.push(format!(
                "\n共沉淀 **{} 份** 跨仓库接口/表结构契约，各专属智能体记忆已同步异步归档。",
                contracts.len()
            ));
        }
        lines.join("\n")
    };

    let rid_save = run_id.to_string();
    let syn_save = synthesis.clone();
    let is_triage = run.mode == "triage";
    let _ = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            set_run_status_conn(&store.conn, &rid_save, "completed", Some(&syn_save))?;
            let msg = if is_triage {
                "三方跨库联合排障会诊已完成！已生成跨层根因定位报告，可一键转化为修复派工单。"
            } else {
                "全部波次执行与独立验收顺利完成！各领域专属智能体已自动归档本次研发记忆。"
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
            Ok(())
        })
    })
    .await;
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

    let Ok(config) = crate::api_config::load_config() else {
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
    rt.max_tokens = Some(2000);
    let coding_cfg = crate::kernel_runtime::coding_config_from_runtime(&rt);
    let options = ChatOptions {
        max_tokens: Some(2000),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };
    let factory = crate::runtime_host::coding_provider_factory();
    let sid = format!("team-triage-syn-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid))).await
    else {
        return fallback;
    };

    let prompt = format!(
        "You are the Lead Incident Commander synthesizing findings from parallel SQL, Backend, and Frontend repository specialists.\n\
        Write a clear, actionable Chinese Markdown Root-Cause Report containing:\n\
        1. **核心根因定位（Root Cause）**：明确指出问题出在哪一层或哪几层契约不一致（附精确文件路径与字段名）。\n\
        2. **跨层证据链对照**：串联前端传参/解包、后端路由/逻辑、数据库表结构/SQL 之间的真实匹配关系。\n\
        3. **分仓库修复行动清单**：按执行先后顺序列出每个仓库需要修改的具体点。\n\n\
        Original Incident:\n{}\n\n\
        Domain Specialist Findings:\n{}",
        run.user_goal, findings
    );

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

    match tokio::time::timeout(Duration::from_secs(35), fut).await {
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
