//! `zap-agent-teams` 智能拆单规划器（AI Lead Planner + 领域智能体与项目目录自动预填 + 排障转修复流水线）。

use std::collections::{BTreeMap, HashMap};

use super::prompts::{
    extract_compact_sql_ddl, render_template_single_pass, resolve_effective_template,
    TPL_LEAD_INTERFACE_SPEC, TPL_LEAD_PLANNER_PIPELINE, TPL_LEAD_PLANNER_TRIAGE,
};
use std::path::Path;
use std::time::Duration;

use futures::StreamExt;
use serde::Deserialize;
use zap_kernel::message::{ImageContent, Message};
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;

use crate::host::{HostProjectInfo, TeamHost};
use zap_neural_memory::NeuralAgent;

use super::store::{
    append_event_conn, create_run_with_tasks_conn, load_config_conn, load_custom_projects_conn,
    load_role_presets_conn, load_run_detail_conn, with_team_store,
};
use super::types::{
    compute_and_assign_waves, domain_badge_cn, new_team_id, normalize_domain, now_ts,
    path_to_clean_str, truncate_chars_safe, RoleBindingPreset, TeamContract, TeamImage,
    TeamProjectPreset, TeamRun, TeamRunDetail, TeamTask,
};

#[derive(Debug, Deserialize)]
pub(crate) struct LlmPlanOutput {
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) requires_db: bool,
    #[serde(default)]
    pub(crate) execution_plan: String,
    #[serde(default)]
    pub(crate) tasks: Vec<LlmPlannedTask>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LlmPlannedTask {
    #[serde(default)]
    pub(crate) key: String,
    #[serde(default)]
    pub(crate) project_id: String,
    #[serde(default)]
    pub(crate) domain: String,
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) description: String,
    #[serde(default)]
    pub(crate) acceptance_criteria: String,
    #[serde(default)]
    pub(crate) depends_on: Vec<String>,
}

pub(crate) fn build_user_message(text: &str, images: &[TeamImage]) -> Message {
    if images.is_empty() {
        Message::user(text)
    } else {
        let imgs = images
            .iter()
            .map(|i| ImageContent {
                media_type: i.media_type.clone(),
                data: i.data.clone(),
            })
            .collect();
        Message::user_with_images(text, imgs)
    }
}

fn resolve_lead_agent_id(override_id: Option<&str>, default_id: &str) -> String {
    if let Some(ov) = override_id.map(str::trim).filter(|s| !s.is_empty()) {
        return ov.to_string();
    }
    let def = default_id.trim();
    if def.is_empty() {
        "none".to_string()
    } else {
        def.to_string()
    }
}

fn strip_recall_tool_footer(text: &str) -> String {
    if let Some(idx) = text.find("[Memory Recall Tool]") {
        let prefix = text[..idx].trim_end();
        if text.trim_end().ends_with("</system-reminder>")
            && !prefix.ends_with("</system-reminder>")
        {
            format!("{prefix}\n</system-reminder>")
        } else {
            prefix.to_string()
        }
    } else {
        text.to_string()
    }
}

/// 当主控经理绑定了专属记忆智能体（非空且不为 `"none"`）时，读取其静态规则档案与按本次查询召回的相关神经记忆知识。
pub(crate) async fn build_lead_agent_memory_context(
    lead_agent_id: &str,
    query_text: &str,
) -> Option<String> {
    let aid = lead_agent_id.trim();
    if aid.is_empty() || aid == "none" {
        return None;
    }
    let aid_owned = aid.to_string();
    let query_owned = truncate_chars_safe(query_text, 1000);
    tokio::task::spawn_blocking(move || {
        let virtual_sid = format!("team_lead_memory_{aid_owned}");
        if zap_neural_memory::bind_session_neural_agent(&virtual_sid, &aid_owned).is_err() {
            return None;
        }
        let mut parts: Vec<String> = Vec::new();
        if let Some(static_blk) = zap_neural_memory::build_static_system_block(&virtual_sid) {
            let trimmed = static_blk.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
        if !query_owned.trim().is_empty() {
            if let Some(dynamic_blk) = zap_neural_memory::build_dynamic_turn_reminder(
                &virtual_sid,
                &query_owned,
                Path::new(""),
            ) {
                let cleaned = strip_recall_tool_footer(dynamic_blk.trim());
                if !cleaned.trim().is_empty() {
                    parts.push(cleaned);
                }
            }
        }
        if parts.is_empty() {
            return None;
        }
        let body = truncate_chars_safe(&parts.join("\n\n"), 6000);
        Some(format!(
            "[Lead Manager Neural Memory Context]\n\
Apply the bound lead manager agent's persona, architectural rules, preferences, and recalled knowledge below when planning, designing unified contracts, or assessing revision impact.\n\
Note: tool calls are disabled in this planning request; rely directly on the injected context below.\n\n\
{body}"
        ))
    })
    .await
    .ok()
    .flatten()
}

fn build_lead_messages(
    lead_memory: Option<String>,
    prompt: &str,
    images: &[TeamImage],
) -> Vec<Message> {
    let user_msg = build_user_message(prompt, images);
    match lead_memory.filter(|s| !s.trim().is_empty()) {
        Some(sys) => vec![Message::system(sys), user_msg],
        None => vec![user_msg],
    }
}

/// 立即创建 `pipeline` 模式的初始任务单（`stage = "planning"`, `status = "running"`），随后由后台异步跑阶段 1+2。
#[allow(clippy::too_many_arguments)]
pub async fn create_pending_pipeline_run(
    host: &dyn TeamHost,
    user_goal: &str,
    lead_model_override: Option<&str>,
    lead_agent_id_override: Option<&str>,
    source_triage_id: Option<&str>,
    images: Vec<TeamImage>,
    prompt_overrides: BTreeMap<String, String>,
) -> Result<TeamRunDetail, String> {
    let goal = user_goal.trim();
    if goal.is_empty() && images.is_empty() {
        return Err("请输入需要拆解的任务需求或粘贴参考图".to_string());
    }
    let goal_text = if goal.is_empty() {
        "请根据附带的效果图/原型截图完成跨项目功能设计与开发。".to_string()
    } else {
        goal.to_string()
    };

    let (global_cfg, presets, custom_projects, agents) = tokio::task::spawn_blocking(|| {
        let (cfg, presets, custom_projects) = with_team_store(|s| {
            Ok((
                load_config_conn(&s.conn),
                load_role_presets_conn(&s.conn)?,
                load_custom_projects_conn(&s.conn)?,
            ))
        })?;
        let agents = zap_neural_memory::list_neural_agents().unwrap_or_default();
        Ok::<_, String>((cfg, presets, custom_projects, agents))
    })
    .await
    .map_err(|e| format!("加载任务配置失败: {e}"))??;

    let preset_map: HashMap<String, RoleBindingPreset> = presets
        .into_iter()
        .map(|p| (normalize_domain(&p.domain), p))
        .collect();

    let config = crate::host::load_config()?;
    let default_model = config.effective_model_selection().unwrap_or_default();
    let lead_model = lead_model_override
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            (!global_cfg.default_model.trim().is_empty())
                .then(|| global_cfg.default_model.trim().to_string())
        })
        .unwrap_or(default_model);

    let run_id = new_team_id("run");
    let now = now_ts();
    let run_title = default_run_title(&goal_text, "pipeline");

    // 如果用户已有项目库，先生成占位任务列表以便侧栏/舞台立即可见（阶段 1 完成后会用 AI 拆单结果精准替换）
    let mut initial_tasks = Vec::new();
    for (idx, p) in custom_projects.iter().enumerate() {
        let dom = normalize_domain(&p.domain);
        let av = if p.avatar_id.trim().is_empty() {
            default_avatar_for_domain(&dom).to_string()
        } else {
            p.avatar_id.clone()
        };
        let aid = resolve_agent_id_for_domain(&p.agent_id, &dom, &preset_map, &agents);
        let agent_label = agents
            .iter()
            .find(|a| a.id == aid)
            .map(|a| a.label.clone())
            .unwrap_or_default();
        initial_tasks.push(TeamTask {
            id: new_team_id("tsk"),
            run_id: run_id.clone(),
            domain: dom,
            title: format!("{} · 待生成分工", p.name),
            description: String::new(),
            acceptance_criteria: String::new(),
            depends_on: Vec::new(),
            wave: 1,
            sort_order: idx,
            project_slot_id: p.id.clone(),
            project_name: p.name.clone(),
            avatar_id: av,
            working_dir: p.working_dir.clone(),
            project_hash: if p.working_dir.trim().is_empty() {
                String::new()
            } else {
                host.hash_path(Path::new(p.working_dir.trim()))
            },
            agent_id: aid,
            agent_label,
            model: p.model.clone(),
            session_id: String::new(),
            custom_prompt: String::new(),
            images: Vec::new(),
            revision_note: String::new(),
            status: "pending".to_string(),
            attempt: 0,
            attempt_id: String::new(),
            repair_count: 0,
            contract_summary: String::new(),
            last_output_excerpt: String::new(),
            verify_verdict: String::new(),
            verify_feedback: String::new(),
            touched_files: Vec::new(),
            round_touched_files: Vec::new(),
            started_at: String::new(),
            updated_at: now.clone(),
            finished_at: String::new(),
        });
    }

    let run = TeamRun {
        id: run_id.clone(),
        title: run_title,
        user_goal: goal_text,
        mode: "pipeline".to_string(),
        status: "running".to_string(),
        stage: "planning".to_string(),
        requires_db: false,
        execution_plan: String::new(),
        sql_schema_draft: String::new(),
        interface_spec: String::new(),
        delivery_sql: String::new(),
        sql_working_dir: String::new(),
        sql_agent_id: "none".to_string(),
        sql_model: String::new(),
        sql_session_id: String::new(),
        prompt_overrides,
        goal_images: images,
        revision: 1,
        auto_verify: global_cfg.auto_verify,
        max_repairs: global_cfg.max_repairs,
        lead_model,
        lead_agent_id: resolve_lead_agent_id(
            lead_agent_id_override,
            &global_cfg.default_lead_agent_id,
        ),
        current_round_no: 1,
        source_triage_id: source_triage_id.unwrap_or("").to_string(),
        synthesis_report: String::new(),
        created_at: now.clone(),
        updated_at: now,
        finished_at: String::new(),
    };

    tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            create_run_with_tasks_conn(&mut store.conn, &run, &initial_tasks)?;
            let _ = append_event_conn(
                &store.conn,
                &run.id,
                "design_planning_started",
                "",
                "lead_planner",
                "已创建新任务，主控智能体正在制定整体《执行计划》…",
                "",
            );
            load_run_detail_conn(&store.conn, &run.id, 0)?
                .ok_or_else(|| "创建任务后读取失败".to_string())
        })
    })
    .await
    .map_err(|e| format!("保存新任务失败: {e}"))?
}

/// 为指定 `pipeline` 任务生成《执行计划》、判断是否需要数据库设计并拆解出各项目子任务。
pub async fn build_pipeline_plan_for_run(
    host: &std::sync::Arc<dyn TeamHost>,
    run: &TeamRun,
    current_working_dir: &Path,
    feedback: Option<&str>,
) -> Result<(String, bool, String, Vec<TeamTask>), String> {
    let host_clone = std::sync::Arc::clone(host);
    let (presets, custom_projects, agents, projects, global_cfg) =
        tokio::task::spawn_blocking(move || {
            let (presets, custom_projects, global_cfg) = with_team_store(|s| {
                Ok((
                    load_role_presets_conn(&s.conn)?,
                    load_custom_projects_conn(&s.conn)?,
                    load_config_conn(&s.conn),
                ))
            })?;
            let agents = zap_neural_memory::list_neural_agents().unwrap_or_default();
            let projects = host_clone.list_projects();
            Ok::<_, String>((presets, custom_projects, agents, projects, global_cfg))
        })
        .await
        .map_err(|e| format!("加载项目上下文失败: {e}"))??;

    let config = crate::host::load_config()?;
    let goal_for_llm = match feedback.map(str::trim).filter(|s| !s.is_empty()) {
        Some(fb) => format!(
            "{}\n\n【用户补充修改意见】：\n{}\n\n【上一版执行计划】：\n{}",
            run.user_goal,
            fb,
            truncate_chars_safe(&run.execution_plan, 1500)
        ),
        None => run.user_goal.clone(),
    };

    let llm_plan = call_planner_llm(
        &config,
        &run.lead_model,
        &run.lead_agent_id,
        current_working_dir,
        &goal_for_llm,
        "pipeline",
        &custom_projects,
        &projects,
        &run.goal_images,
        &run.prompt_overrides,
        &global_cfg.prompt_templates,
    )
    .await
    .unwrap_or_else(|| build_heuristic_plan(&run.user_goal, "pipeline", &custom_projects));

    let title = if llm_plan.title.trim().is_empty() {
        run.title.clone()
    } else {
        truncate_chars_safe(llm_plan.title.trim(), 72)
    };
    let requires_db = llm_plan.requires_db
        || goal_implies_db(&run.user_goal)
        || custom_projects
            .iter()
            .any(|p| normalize_domain(&p.domain) == "sql");
    let execution_plan = if llm_plan.execution_plan.trim().is_empty() {
        format!(
            "### 执行计划\n- 需求目标：{}\n- 涉及项目数：{}\n- 是否涉及数据库表结构设计：{}",
            run.user_goal,
            llm_plan.tasks.len(),
            if requires_db { "是" } else { "否" }
        )
    } else {
        llm_plan.execution_plan.trim().to_string()
    };

    let mut raw_tasks = if llm_plan.tasks.is_empty() {
        build_heuristic_plan(&run.user_goal, "pipeline", &custom_projects).tasks
    } else {
        llm_plan.tasks
    };

    // 数据库表结构由独立前置阶段（阶段 2）设计并在交付台一键复制，不在研发团队列表里额外占一张 SQL 编码卡片（除非整单仅含 SQL 任务）
    let has_non_sql = raw_tasks
        .iter()
        .any(|t| normalize_domain(&t.domain) != "sql");
    if has_non_sql {
        raw_tasks.retain(|t| normalize_domain(&t.domain) != "sql");
    }

    let tasks = materialize_planned_tasks(
        host.as_ref(),
        &run.id,
        &run.user_goal,
        "pipeline",
        &run.lead_model,
        current_working_dir,
        raw_tasks,
        &custom_projects,
        &presets,
        &agents,
        &projects,
        None,
    );

    Ok((title, requires_db, execution_plan, tasks))
}

/// 阶段 3：主控根据用户已拍板确认的《表结构方案》和《执行计划》，统一约定《前后端接口、字典与权限方案》。
pub async fn generate_unified_interface_spec(
    run: &TeamRun,
    tasks: &[TeamTask],
    current_working_dir: &Path,
) -> String {
    let task_summary = tasks
        .iter()
        .map(|t| {
            format!(
                "- [{}] {} (`{}`): {}",
                domain_badge_cn(&t.domain),
                if t.project_name.is_empty() {
                    &t.title
                } else {
                    &t.project_name
                },
                t.working_dir,
                truncate_chars_safe(&t.description, 220)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let fallback_plan = if run.execution_plan.trim().is_empty() {
        "见任务描述".to_string()
    } else {
        truncate_chars_safe(&run.execution_plan, 600)
    };
    let fallback_sql = if run.sql_schema_draft.trim().is_empty() {
        "（无新增/改表）".to_string()
    } else {
        truncate_chars_safe(&extract_compact_sql_ddl(&run.sql_schema_draft), 1000)
    };
    let fallback = format!(
        "### 前后端统一接口与联调约定\n\n#### 1. 接口约定摘要\n{}\n\n#### 2. 表结构要点\n{}\n\n#### 3. 协作分工\n{}\n\n请后端与各端严格对齐 REST 路径、JSON 驼峰字段、字典码值与权限标识。",
        fallback_plan,
        fallback_sql,
        truncate_chars_safe(&task_summary, 900)
    );

    let Ok(config) = crate::host::load_config() else {
        return fallback;
    };
    let prov_opt = (!run.lead_model.trim().is_empty()).then_some(run.lead_model.trim());
    let mut rt = zap_coding::CodingRuntimeConfig::from_config(
        &config,
        current_working_dir,
        prov_opt,
        None,
        false,
        false,
    );
    rt.mcp = false;
    rt.max_tokens = Some(8192);
    let coding_cfg = rt.agent_config();
    let options = ChatOptions {
        max_tokens: Some(8192),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Medium),
        ..coding_cfg.chat_options.clone()
    };
    let factory = crate::host::coding_provider_factory();
    let sid = format!("team-contract-spec-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid))).await
    else {
        return fallback;
    };

    let sql_part = if run.sql_schema_draft.trim().is_empty() {
        "（本次任务不涉及数据库建表/改表）".to_string()
    } else {
        run.sql_schema_draft.clone()
    };

    let (global_templates, compact_prompts) = tokio::task::spawn_blocking(|| {
        with_team_store(|s| {
            let cfg = load_config_conn(&s.conn);
            Ok((cfg.prompt_templates, cfg.compact_prompts))
        })
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or_else(|| (BTreeMap::new(), true));
    let (template, _) = resolve_effective_template(
        TPL_LEAD_INTERFACE_SPEC,
        None,
        Some(&run.prompt_overrides),
        &global_templates,
    );
    let (spec_goal, spec_plan, spec_sql) = if compact_prompts {
        (
            truncate_chars_safe(&run.user_goal, 600),
            truncate_chars_safe(&run.execution_plan, 1800),
            if run.sql_schema_draft.trim().is_empty() {
                sql_part.clone()
            } else {
                truncate_chars_safe(&extract_compact_sql_ddl(&sql_part), 2200)
            },
        )
    } else {
        (run.user_goal.clone(), run.execution_plan.clone(), sql_part)
    };
    let prompt = render_template_single_pass(
        &template,
        &[
            ("user_goal", &spec_goal),
            ("execution_plan", &spec_plan),
            ("sql_schema_draft", &spec_sql),
            ("task_summary", &task_summary),
        ],
    );

    let lead_memory = build_lead_agent_memory_context(
        &run.lead_agent_id,
        &format!("{} {}", run.title, truncate_chars_safe(&run.user_goal, 500)),
    )
    .await;
    let msgs = build_lead_messages(lead_memory, &prompt, &run.goal_images);
    let fut = async {
        let mut stream = match provider.chat_stream(&msgs, &[], &options).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "[zap-agent-teams] generate_unified_interface_spec chat_stream failed: {e:?}"
                );
                return None;
            }
        };
        let mut text = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Done { .. } => break,
                StreamEvent::Error(e) => {
                    tracing::warn!(
                        "[zap-agent-teams] generate_unified_interface_spec stream error: {e:?}"
                    );
                    break;
                }
                _ => {}
            }
        }
        (!text.trim().is_empty()).then_some(text)
    };

    match tokio::time::timeout(Duration::from_secs(120), fut).await {
        Ok(Some(spec)) => spec,
        _ => fallback,
    }
}

fn goal_implies_db(goal: &str) -> bool {
    let low = goal.to_ascii_lowercase();
    low.contains("表")
        || low.contains("字段")
        || low.contains("数据库")
        || low.contains("sql")
        || low.contains("ddl")
        || low.contains("达梦")
        || low.contains("dm8")
        || low.contains("新增功能")
        || low.contains("管理功能")
        || low.contains("后台配置")
        || low.contains("模块")
        || low.contains("公告")
        || low.contains("审批")
        || low.contains("记录")
}

/// 根据用户输入的需求（或线上报错），用于 `triage` 模式同步拆解任务草稿并预填项目目录与记忆智能体。
#[allow(clippy::too_many_arguments)]
pub async fn plan_team_run(
    host: &std::sync::Arc<dyn TeamHost>,
    user_goal: &str,
    mode: &str,
    lead_model_override: Option<&str>,
    lead_agent_id_override: Option<&str>,
    current_working_dir: &Path,
    source_triage_id: Option<&str>,
    inherited_bindings: Option<&HashMap<String, (String, String, String)>>,
    images: Vec<TeamImage>,
    prompt_overrides: BTreeMap<String, String>,
) -> Result<TeamRunDetail, String> {
    let goal = user_goal.trim();
    if goal.is_empty() && images.is_empty() {
        return Err("请输入需要拆解的任务目标或故障现象".to_string());
    }
    let goal_text = if goal.is_empty() {
        "请结合附带的截图进行跨库分析与定位。".to_string()
    } else {
        goal.to_string()
    };

    let norm_mode = if mode.trim().eq_ignore_ascii_case("triage") {
        "triage"
    } else {
        "pipeline"
    };

    let host_clone = std::sync::Arc::clone(host);
    let (global_cfg, presets, custom_projects, agents, projects) =
        tokio::task::spawn_blocking(move || {
            let (cfg, presets, custom_projects) = with_team_store(|s| {
                Ok((
                    load_config_conn(&s.conn),
                    load_role_presets_conn(&s.conn)?,
                    load_custom_projects_conn(&s.conn)?,
                ))
            })?;
            let agents = zap_neural_memory::list_neural_agents().unwrap_or_default();
            let projects = host_clone.list_projects();
            Ok::<_, String>((cfg, presets, custom_projects, agents, projects))
        })
        .await
        .map_err(|e| format!("加载规划上下文失败: {e}"))??;

    let config = crate::host::load_config()?;
    let default_model = config.effective_model_selection().unwrap_or_default();
    let lead_model = lead_model_override
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            (!global_cfg.default_model.trim().is_empty())
                .then(|| global_cfg.default_model.trim().to_string())
        })
        .unwrap_or(default_model);
    let lead_agent_id =
        resolve_lead_agent_id(lead_agent_id_override, &global_cfg.default_lead_agent_id);

    let llm_plan = call_planner_llm(
        &config,
        &lead_model,
        &lead_agent_id,
        current_working_dir,
        &goal_text,
        norm_mode,
        &custom_projects,
        &projects,
        &images,
        &prompt_overrides,
        &global_cfg.prompt_templates,
    )
    .await
    .unwrap_or_else(|| build_heuristic_plan(&goal_text, norm_mode, &custom_projects));

    let run_id = new_team_id("run");
    let now = now_ts();
    let run_title = if llm_plan.title.trim().is_empty() {
        default_run_title(&goal_text, norm_mode)
    } else {
        truncate_chars_safe(llm_plan.title.trim(), 72)
    };

    let raw_tasks = if llm_plan.tasks.is_empty() {
        build_heuristic_plan(&goal_text, norm_mode, &custom_projects).tasks
    } else {
        llm_plan.tasks
    };

    let tasks = materialize_planned_tasks(
        host.as_ref(),
        &run_id,
        &goal_text,
        norm_mode,
        &lead_model,
        current_working_dir,
        raw_tasks,
        &custom_projects,
        &presets,
        &agents,
        &projects,
        inherited_bindings,
    );

    let run = TeamRun {
        id: run_id.clone(),
        title: run_title,
        user_goal: goal_text,
        mode: norm_mode.to_string(),
        status: "draft".to_string(),
        stage: if norm_mode == "triage" {
            "triage".to_string()
        } else {
            "awaiting_approval".to_string()
        },
        requires_db: llm_plan.requires_db,
        execution_plan: llm_plan.execution_plan,
        sql_schema_draft: String::new(),
        interface_spec: String::new(),
        delivery_sql: String::new(),
        sql_working_dir: String::new(),
        sql_agent_id: "none".to_string(),
        sql_model: String::new(),
        sql_session_id: String::new(),
        prompt_overrides,
        goal_images: images,
        revision: 1,
        auto_verify: global_cfg.auto_verify && norm_mode == "pipeline",
        max_repairs: if norm_mode == "pipeline" {
            global_cfg.max_repairs
        } else {
            0
        },
        lead_model,
        lead_agent_id,
        current_round_no: 1,
        source_triage_id: source_triage_id.unwrap_or("").to_string(),
        synthesis_report: String::new(),
        created_at: now.clone(),
        updated_at: now,
        finished_at: String::new(),
    };

    let detail = tokio::task::spawn_blocking(move || {
        with_team_store(|store| {
            create_run_with_tasks_conn(&mut store.conn, &run, &tasks)?;
            let mode_text = if run.mode == "triage" {
                "跨库联合排障会诊任务"
            } else {
                "跨项目研发任务"
            };
            let _ = append_event_conn(
                &store.conn,
                &run.id,
                "run_planned",
                "",
                "lead_planner",
                &format!(
                    "已生成{}（共 {} 个项目分工），请核对项目目录与专属智能体后启动执行。",
                    mode_text,
                    tasks.len()
                ),
                "",
            );
            load_run_detail_conn(&store.conn, &run.id, 0)?
                .ok_or_else(|| "创建任务后读取失败".to_string())
        })
    })
    .await
    .map_err(|e| format!("保存任务草稿失败: {e}"))??;

    Ok(detail)
}

#[allow(clippy::too_many_arguments)]
fn materialize_planned_tasks(
    host: &dyn TeamHost,
    run_id: &str,
    goal: &str,
    norm_mode: &str,
    lead_model: &str,
    current_working_dir: &Path,
    raw_tasks: Vec<LlmPlannedTask>,
    custom_projects: &[TeamProjectPreset],
    presets: &[RoleBindingPreset],
    agents: &[NeuralAgent],
    projects: &[HostProjectInfo],
    inherited_bindings: Option<&HashMap<String, (String, String, String)>>,
) -> Vec<TeamTask> {
    let now = now_ts();
    let preset_map: HashMap<String, RoleBindingPreset> = presets
        .iter()
        .cloned()
        .map(|p| (normalize_domain(&p.domain), p))
        .collect();

    // 第一遍：分配真实任务 ID 并建立 key/domain -> task_id 映射
    let mut key_to_id: HashMap<String, String> = HashMap::new();
    let mut domain_first_id: HashMap<String, String> = HashMap::new();
    let mut allocated: Vec<(String, LlmPlannedTask)> = Vec::with_capacity(raw_tasks.len());

    for (idx, item) in raw_tasks.into_iter().take(8).enumerate() {
        let tid = new_team_id("tsk");
        let dom = normalize_domain(&item.domain);
        let key = if item.key.trim().is_empty() {
            format!("step_{}", idx + 1)
        } else {
            item.key.trim().to_string()
        };
        key_to_id.insert(key.to_ascii_lowercase(), tid.clone());
        domain_first_id.entry(dom).or_insert_with(|| tid.clone());
        allocated.push((tid, item));
    }

    let mut tasks: Vec<TeamTask> = Vec::with_capacity(allocated.len());
    let fallback_wd = path_to_clean_str(current_working_dir);

    for (idx, (tid, item)) in allocated.into_iter().enumerate() {
        let matched_custom = match_custom_project(&item.project_id, &item.domain, custom_projects);
        let dom = matched_custom
            .map(|p| normalize_domain(&p.domain))
            .unwrap_or_else(|| normalize_domain(&item.domain));
        let deps: Vec<String> = if norm_mode == "triage" || dom != "qa" {
            // 新四阶段流中：阶段 3 已先锁定表结构与前后端统一接口方案，因此后端与所有前端默认同时并发开干；仅 QA 集成测试依赖研发节点
            if norm_mode == "pipeline" && dom != "qa" {
                Vec::new()
            } else {
                Vec::new()
            }
        } else {
            item.depends_on
                .iter()
                .filter_map(|raw_dep| {
                    let k = raw_dep.trim().to_ascii_lowercase();
                    key_to_id
                        .get(&k)
                        .or_else(|| domain_first_id.get(&normalize_domain(&k)))
                        .filter(|&mapped| mapped != &tid)
                        .cloned()
                })
                .collect()
        };

        let (wd, aid, task_model, slot_id, proj_name, avatar_id) = if let Some(cp) = matched_custom
        {
            let chosen_wd = if cp.working_dir.trim().is_empty() {
                fallback_wd.clone()
            } else {
                cp.working_dir.trim().to_string()
            };
            let chosen_aid = resolve_agent_id_for_domain(&cp.agent_id, &dom, &preset_map, agents);
            let chosen_mdl = if cp.model.trim().is_empty() {
                lead_model.to_string()
            } else {
                cp.model.trim().to_string()
            };
            let chosen_avatar = if cp.avatar_id.trim().is_empty() {
                default_avatar_for_domain(&dom).to_string()
            } else {
                cp.avatar_id.clone()
            };
            (
                chosen_wd,
                chosen_aid,
                chosen_mdl,
                cp.id.clone(),
                cp.name.clone(),
                chosen_avatar,
            )
        } else {
            let (wd, aid, task_model) = resolve_role_binding(
                &dom,
                goal,
                &fallback_wd,
                lead_model,
                &preset_map,
                inherited_bindings,
                agents,
                projects,
            );
            (
                wd,
                aid,
                task_model,
                String::new(),
                String::new(),
                default_avatar_for_domain(&dom).to_string(),
            )
        };

        let agent_label = agents
            .iter()
            .find(|a| a.id == aid)
            .map(|a| a.label.clone())
            .unwrap_or_default();
        let proj_hash = if wd.trim().is_empty() {
            String::new()
        } else {
            host.hash_path(Path::new(&wd))
        };

        let title = if item.title.trim().is_empty() {
            if !proj_name.is_empty() {
                format!("{} 智能体", proj_name)
            } else {
                format!("{} 智能体", domain_badge_cn(&dom))
            }
        } else {
            item.title.trim().to_string()
        };

        let acceptance = if item.acceptance_criteria.trim().is_empty() {
            default_acceptance_for_domain(&dom, norm_mode)
        } else {
            item.acceptance_criteria.trim().to_string()
        };

        tasks.push(TeamTask {
            id: tid,
            run_id: run_id.to_string(),
            domain: dom,
            title,
            description: item.description.trim().to_string(),
            acceptance_criteria: acceptance,
            depends_on: deps,
            wave: 1,
            sort_order: idx,
            project_slot_id: slot_id,
            project_name: proj_name,
            avatar_id,
            working_dir: wd,
            project_hash: proj_hash,
            agent_id: aid,
            agent_label,
            model: task_model,
            session_id: String::new(),
            custom_prompt: String::new(),
            images: Vec::new(),
            revision_note: String::new(),
            status: "pending".to_string(),
            attempt: 0,
            attempt_id: String::new(),
            repair_count: 0,
            contract_summary: String::new(),
            last_output_excerpt: String::new(),
            verify_verdict: String::new(),
            verify_feedback: String::new(),
            touched_files: Vec::new(),
            round_touched_files: Vec::new(),
            started_at: String::new(),
            updated_at: now.clone(),
            finished_at: String::new(),
        });
    }

    if compute_and_assign_waves(&mut tasks).is_err() {
        for t in &mut tasks {
            t.depends_on.clear();
        }
        let _ = compute_and_assign_waves(&mut tasks);
    }
    tasks
}

/// 将已完成的「三方跨库联合排障会诊任务」一键转化为「跨项目修复任务」，保留原参考图并进入四阶段设计。
pub async fn convert_triage_to_pipeline(
    host: &dyn TeamHost,
    triage_run_id: &str,
) -> Result<TeamRunDetail, String> {
    let tid = triage_run_id.trim().to_string();
    let triage_detail = tokio::task::spawn_blocking(move || {
        with_team_store(|store| load_run_detail_conn(&store.conn, &tid, 0))
    })
    .await
    .map_err(|e| e.to_string())??
    .ok_or_else(|| "未找到对应的排障会诊任务".to_string())?;

    let combined_goal = if triage_detail.run.synthesis_report.trim().is_empty() {
        format!("根据排障需求修复问题：\n{}", triage_detail.run.user_goal)
    } else {
        format!(
            "请基于以下跨库联合会诊根因结论落地修复方案：\n\n【原始问题】：\n{}\n\n【联合会诊结论与修复建议】：\n{}",
            triage_detail.run.user_goal,
            truncate_chars_safe(&triage_detail.run.synthesis_report, 3000)
        )
    };

    create_pending_pipeline_run(
        host,
        &combined_goal,
        Some(&triage_detail.run.lead_model),
        Some(&triage_detail.run.lead_agent_id),
        Some(&triage_detail.run.id),
        triage_detail.run.goal_images,
        triage_detail.run.prompt_overrides,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
fn resolve_role_binding(
    domain: &str,
    goal: &str,
    fallback_wd: &str,
    lead_model: &str,
    presets: &HashMap<String, RoleBindingPreset>,
    inherited: Option<&HashMap<String, (String, String, String)>>,
    agents: &[NeuralAgent],
    projects: &[HostProjectInfo],
) -> (String, String, String) {
    if let Some(map) = inherited {
        if let Some((wd, aid, mdl)) = map.get(domain) {
            if !wd.is_empty() || aid != "none" {
                return (
                    if wd.is_empty() {
                        fallback_wd.to_string()
                    } else {
                        wd.clone()
                    },
                    aid.clone(),
                    if mdl.is_empty() {
                        lead_model.to_string()
                    } else {
                        mdl.clone()
                    },
                );
            }
        }
    }

    let preset = presets.get(domain);
    let wd = preset
        .map(|p| p.working_dir.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| guess_project_for_domain(domain, goal, projects))
        .unwrap_or_else(|| fallback_wd.to_string());

    let aid = resolve_agent_id_for_domain(
        preset.map(|p| p.agent_id.as_str()).unwrap_or(""),
        domain,
        presets,
        agents,
    );

    let mdl = preset
        .map(|p| p.model.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| lead_model.to_string());

    (wd, aid, mdl)
}

fn resolve_agent_id_for_domain(
    explicit_agent_id: &str,
    domain: &str,
    presets: &HashMap<String, RoleBindingPreset>,
    agents: &[NeuralAgent],
) -> String {
    let explicit = explicit_agent_id.trim();
    if !explicit.is_empty() && explicit != "none" && agents.iter().any(|a| a.id == explicit) {
        return explicit.to_string();
    }
    presets
        .get(domain)
        .map(|p| p.agent_id.trim().to_string())
        .filter(|s| !s.is_empty() && s != "none" && agents.iter().any(|a| a.id == *s))
        .or_else(|| guess_agent_for_domain(domain, agents))
        .unwrap_or_else(|| "none".to_string())
}

fn guess_agent_for_domain(domain: &str, agents: &[NeuralAgent]) -> Option<String> {
    let patterns: &[&str] = match domain {
        "sql" => &["sql", "db", "data", "mysql", "postgres", "数据库", "数仓"],
        "backend" => &[
            "backend", "server", "api", "rust", "java", "go", "后端", "服务",
        ],
        "frontend" => &["frontend", "web", "ui", "vue", "react", "前端", "界面"],
        "mobile" => &[
            "mobile",
            "app",
            "android",
            "ios",
            "flutter",
            "uniapp",
            "react-native",
            "移动",
            "小程序",
            "frontend",
            "前端",
        ],
        "qa" => &["qa", "test", "verify", "review", "测试", "验收"],
        _ => &["general", "default", "arch", "lead", "通用", "架构"],
    };
    for pat in patterns {
        if let Some(found) = agents.iter().find(|a| {
            a.id.to_ascii_lowercase().contains(pat)
                || a.name.to_ascii_lowercase().contains(pat)
                || a.label.to_ascii_lowercase().contains(pat)
        }) {
            return Some(found.id.clone());
        }
    }
    None
}

fn guess_project_for_domain(
    domain: &str,
    goal: &str,
    projects: &[HostProjectInfo],
) -> Option<String> {
    let goal_low = goal.to_ascii_lowercase();
    // 若用户原话点名了某个已知仓库名且与领域沾边，优先命中
    let patterns: &[&str] = match domain {
        "sql" => &[
            "sql",
            "db",
            "schema",
            "migration",
            "data",
            "server",
            "backend",
        ],
        "backend" => &["server", "backend", "api", "service", "core", "daemon"],
        "frontend" => &["web", "front", "ui", "admin", "client", "app", "portal"],
        _ => &[],
    };
    for p in projects {
        let name_low = p.name.to_ascii_lowercase();
        if !name_low.is_empty()
            && goal_low.contains(&name_low)
            && patterns.iter().any(|k| name_low.contains(k))
        {
            return Some(p.working_dir.to_string_lossy().replace('\\', "/"));
        }
    }
    for pat in patterns {
        if let Some(p) = projects
            .iter()
            .find(|p| p.name.to_ascii_lowercase().contains(pat))
        {
            return Some(p.working_dir.to_string_lossy().replace('\\', "/"));
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
async fn call_planner_llm(
    config: &zap_config::Config,
    lead_model: &str,
    lead_agent_id: &str,
    working_dir: &Path,
    goal: &str,
    mode: &str,
    custom_projects: &[TeamProjectPreset],
    projects: &[HostProjectInfo],
    images: &[TeamImage],
    run_overrides: &BTreeMap<String, String>,
    global_templates: &BTreeMap<String, String>,
) -> Option<LlmPlanOutput> {
    let prov_opt = (!lead_model.trim().is_empty()).then_some(lead_model.trim());
    let mut rt = zap_coding::CodingRuntimeConfig::from_config(
        config,
        working_dir,
        prov_opt,
        None,
        false,
        false,
    );
    rt.mcp = false;
    rt.max_tokens = Some(8192);
    let coding_cfg = rt.agent_config();
    let options = ChatOptions {
        max_tokens: Some(8192),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory = crate::host::coding_provider_factory();
    let sid = format!("team-planner-{}", uuid::Uuid::new_v4());
    let provider = tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid)))
        .await
        .ok()?
        .ok()?;

    let user_projects_desc = if custom_projects.is_empty() {
        "(none configured yet)".to_string()
    } else {
        custom_projects
            .iter()
            .map(|p| {
                format!(
                    "- project_id: `{}` | name: `{}` | role_tag: `{}` | dir: `{}` | notes: {}",
                    p.id,
                    p.name,
                    p.domain,
                    p.working_dir,
                    if p.notes.trim().is_empty() {
                        "-"
                    } else {
                        p.notes.trim()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let known_repos = if projects.is_empty() {
        "(none)".to_string()
    } else {
        let configured_dirs: std::collections::HashSet<String> = custom_projects
            .iter()
            .map(|p| p.working_dir.trim().to_ascii_lowercase())
            .collect();
        let limit = if custom_projects.is_empty() { 10 } else { 5 };
        let items: Vec<String> = projects
            .iter()
            .filter(|p| {
                !configured_dirs.contains(&path_to_clean_str(&p.working_dir).to_ascii_lowercase())
            })
            .take(limit)
            .map(|p| format!("- {} ({})", p.name, path_to_clean_str(&p.working_dir)))
            .collect();
        if items.is_empty() {
            "(covered by Projects)".to_string()
        } else {
            items.join("\n")
        }
    };

    let template_id = if mode == "triage" {
        TPL_LEAD_PLANNER_TRIAGE
    } else {
        TPL_LEAD_PLANNER_PIPELINE
    };
    let (template, _) =
        resolve_effective_template(template_id, None, Some(run_overrides), global_templates);
    let wd = path_to_clean_str(working_dir);
    let prompt = render_template_single_pass(
        &template,
        &[
            ("user_goal", goal),
            ("user_projects", &user_projects_desc),
            ("known_repos", &known_repos),
            ("current_working_dir", &wd),
        ],
    );

    let lead_memory = build_lead_agent_memory_context(lead_agent_id, goal).await;
    let msgs = build_lead_messages(lead_memory, &prompt, images);
    let fut = async {
        let mut stream = match provider.chat_stream(&msgs, &[], &options).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("[zap-agent-teams] call_planner_llm chat_stream failed: {e:?}");
                return None;
            }
        };
        let mut raw = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(d) => raw.push_str(&d),
                StreamEvent::Done { .. } => break,
                StreamEvent::Error(e) => {
                    tracing::warn!("[zap-agent-teams] call_planner_llm stream error: {e:?}");
                    break;
                }
                _ => {}
            }
        }
        let parsed =
            extract_json_object(&raw).and_then(|j| serde_json::from_str::<LlmPlanOutput>(j).ok());
        if parsed.is_none() {
            tracing::warn!(
                "[zap-agent-teams] call_planner_llm failed to parse JSON (len={}): {}",
                raw.len(),
                truncate_chars_safe(&raw, 300)
            );
        }
        parsed
    };

    tokio::time::timeout(Duration::from_secs(75), fut)
        .await
        .ok()
        .flatten()
}

pub(crate) fn extract_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    (end >= start).then_some(&raw[start..=end])
}

fn default_run_title(goal: &str, mode: &str) -> String {
    let first_line = goal
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or(goal)
        .trim();
    let short = truncate_chars_safe(first_line, 32);
    if mode == "triage" {
        format!("联合会诊：{short}")
    } else {
        short
    }
}

fn default_acceptance_for_domain(domain: &str, mode: &str) -> String {
    if mode == "triage" {
        return "只读定位真实文件路径、行号与根因证据，不修改业务代码。".to_string();
    }
    match domain {
        "sql" => "表结构与约束完整，发布准确 DDL 契约。".to_string(),
        "backend" => "对齐表结构与锁定接口契约，编译/语法校验通过。".to_string(),
        "frontend" | "mobile" => "对齐接口契约完成页面交互，构建/类型校验无报错。".to_string(),
        _ => "按分工完成改动并通过基础语法/构建检查。".to_string(),
    }
}

fn match_custom_project<'a>(
    project_id: &str,
    domain: &str,
    custom_projects: &'a [TeamProjectPreset],
) -> Option<&'a TeamProjectPreset> {
    let pid = project_id.trim();
    if !pid.is_empty() {
        if let Some(p) = custom_projects
            .iter()
            .find(|p| p.id == pid || p.name.eq_ignore_ascii_case(pid))
        {
            return Some(p);
        }
    }
    let dom = normalize_domain(domain);
    let matches: Vec<&TeamProjectPreset> = custom_projects
        .iter()
        .filter(|p| normalize_domain(&p.domain) == dom)
        .collect();
    if matches.len() == 1 {
        return Some(matches[0]);
    }
    None
}

fn default_avatar_for_domain(domain: &str) -> &'static str {
    match domain {
        "sql" => "sql_alchemist",
        "backend" => "backend_geek",
        "frontend" => "frontend_artist",
        "mobile" => "mobile_ranger",
        "qa" => "qa_detective",
        _ => "whale_maid",
    }
}

fn build_heuristic_plan(
    goal: &str,
    mode: &str,
    custom_projects: &[TeamProjectPreset],
) -> LlmPlanOutput {
    if !custom_projects.is_empty() {
        let mut tasks = Vec::new();
        if mode == "triage" {
            for (idx, p) in custom_projects.iter().enumerate() {
                tasks.push(LlmPlannedTask {
                    key: format!("t{}", idx + 1),
                    project_id: p.id.clone(),
                    domain: normalize_domain(&p.domain),
                    title: format!("排查「{}」代码与数据线索", p.name),
                    description: format!(
                        "针对问题「{goal}」，只读勘察项目「{}」相关代码并输出关键证据与根因判断。",
                        p.name
                    ),
                    acceptance_criteria: default_acceptance_for_domain(&p.domain, "triage"),
                    depends_on: Vec::new(),
                });
            }
            return LlmPlanOutput {
                title: default_run_title(goal, mode),
                requires_db: goal_implies_db(goal),
                execution_plan: String::new(),
                tasks,
            };
        }

        let rank = |d: &str| match d {
            "sql" => 0usize,
            "backend" => 1,
            "frontend" => 2,
            "mobile" => 2,
            "general" => 3,
            "qa" => 4,
            _ => 3,
        };
        let mut ordered: Vec<&TeamProjectPreset> = custom_projects.iter().collect();
        ordered.sort_by_key(|p| (rank(&normalize_domain(&p.domain)), p.sort_order));

        for (idx, p) in ordered.into_iter().enumerate() {
            let dom = normalize_domain(&p.domain);
            let key = format!("t{}", idx + 1);
            tasks.push(LlmPlannedTask {
                key: key.clone(),
                project_id: p.id.clone(),
                domain: dom.clone(),
                title: format!("{} · 落地改造", p.name),
                description: format!("围绕需求「{goal}」完成项目「{}」的开发修改，并在需要时通过 team_publish_contract 发布契约。", p.name),
                acceptance_criteria: default_acceptance_for_domain(&dom, "pipeline"),
                depends_on: Vec::new(),
            });
        }
        return LlmPlanOutput {
            title: default_run_title(goal, mode),
            requires_db: goal_implies_db(goal),
            execution_plan: format!(
                "### 执行计划\n1. 先完成执行计划确认与表结构设计（如涉及数据库）。\n2. 主控基于定稿表结构统一约定接口、字典码值与权限标识。\n3. 由各绑定项目智能体并发推进代码实现与交叉验收。"
            ),
            tasks,
        };
    }

    let low = goal.to_ascii_lowercase();
    let only_backend =
        (low.contains("只改后端") || low.contains("仅后端") || low.contains("纯后端"))
            && !low.contains("前端");
    let only_frontend =
        (low.contains("只改前端") || low.contains("仅前端") || low.contains("纯前端"))
            && !low.contains("后端");
    let has_mobile = low.contains("移动")
        || low.contains("小程序")
        || low.contains("app")
        || low.contains("mobile")
        || low.contains("uniapp")
        || low.contains("h5");
    let has_sql = goal_implies_db(goal);

    let mut tasks = Vec::new();
    if mode == "triage" {
        for (idx, (dom, label)) in [
            ("backend", "后端智能体 · 排查接口路由、业务逻辑与 SQL 线索"),
            ("frontend", "前端智能体 · 排查请求传参、响应解包与页面渲染"),
        ]
        .iter()
        .enumerate()
        {
            tasks.push(LlmPlannedTask {
                key: format!("t{}", idx + 1),
                project_id: String::new(),
                domain: (*dom).to_string(),
                title: (*label).to_string(),
                description: format!(
                    "针对问题「{goal}」，只读勘察对应项目代码并输出关键代码证据与根因判断。"
                ),
                acceptance_criteria: default_acceptance_for_domain(dom, "triage"),
                depends_on: Vec::new(),
            });
        }
        if has_mobile {
            tasks.push(LlmPlannedTask {
                key: "t3".to_string(),
                project_id: String::new(),
                domain: "mobile".to_string(),
                title: "移动端智能体 · 排查移动端请求与页面兼容".to_string(),
                description: format!("针对问题「{goal}」，只读勘察移动端代码并输出排查证据。"),
                acceptance_criteria: default_acceptance_for_domain("mobile", "triage"),
                depends_on: Vec::new(),
            });
        }
        return LlmPlanOutput {
            title: default_run_title(goal, mode),
            requires_db: false,
            execution_plan: String::new(),
            tasks,
        };
    }

    if !only_frontend {
        tasks.push(LlmPlannedTask {
            key: "t_backend".to_string(),
            project_id: String::new(),
            domain: "backend".to_string(),
            title: "后端智能体 · 服务端接口与数据层改造".to_string(),
            description: format!(
                "围绕需求「{goal}」实现后端接口、校验逻辑与实体/Mapper 改造，严格对齐锁定契约。"
            ),
            acceptance_criteria: default_acceptance_for_domain("backend", "pipeline"),
            depends_on: Vec::new(),
        });
    }
    if !only_backend {
        tasks.push(LlmPlannedTask {
            key: "t_frontend".to_string(),
            project_id: String::new(),
            domain: "frontend".to_string(),
            title: "前端智能体 · 管理/Web 页面与接口联调".to_string(),
            description: format!(
                "围绕需求「{goal}」严格按照统一接口契约完成前端页面、组件交互与状态流转。"
            ),
            acceptance_criteria: default_acceptance_for_domain("frontend", "pipeline"),
            depends_on: Vec::new(),
        });
    }
    if has_mobile {
        tasks.push(LlmPlannedTask {
            key: "t_mobile".to_string(),
            project_id: String::new(),
            domain: "mobile".to_string(),
            title: "移动端智能体 · 移动端/小程序页面对接".to_string(),
            description: format!(
                "围绕需求「{goal}」严格按照统一接口契约完成移动端页面适配与交互实现。"
            ),
            acceptance_criteria: default_acceptance_for_domain("mobile", "pipeline"),
            depends_on: Vec::new(),
        });
    }

    LlmPlanOutput {
        title: default_run_title(goal, mode),
        requires_db: has_sql,
        execution_plan: format!(
            "### 执行计划\n- 任务目标：{goal}\n- 自动生成智能体：{}\n- 是否涉及数据库表结构设计：{}\n- 请先确认或增删智能体并各自绑定代码项目，确认后{}",
            tasks.iter().map(|t| t.title.as_str()).collect::<Vec<_>>().join("、"),
            if has_sql { "是" } else { "否" },
            if has_sql {
                "由 SQL 智能体依据绑定的后端项目出表结构供你拍板。"
            } else {
                "由主控锁定前后端接口并并发开工。"
            }
        ),
        tasks,
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct IncrementalCandidate {
    #[serde(default)]
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) domain: String,
    #[serde(default)]
    pub(crate) working_dir: String,
    #[serde(default)]
    pub(crate) notes: String,
    #[serde(default)]
    pub(crate) ref_task_ids: Vec<String>,
    #[serde(default)]
    pub(crate) images: Vec<TeamImage>,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct IncrementalTaskDraft {
    #[serde(default)]
    pub(crate) candidate_index: usize,
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) description: String,
    #[serde(default)]
    pub(crate) custom_prompt: String,
    #[serde(default)]
    pub(crate) acceptance_criteria: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct IncrementalPlanResult {
    pub(crate) tasks: Vec<IncrementalTaskDraft>,
    pub(crate) interface_spec_addition: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmIncrementalOutput {
    #[serde(default)]
    tasks: Vec<IncrementalTaskDraft>,
    #[serde(default, alias = "interface_spec_addition")]
    interface_spec_addition: String,
}

fn build_referenced_member_context(
    ref_ids: &[String],
    existing_tasks: &[TeamTask],
    contracts: &[TeamContract],
) -> String {
    let selected: Vec<&TeamTask> = if ref_ids.is_empty() {
        Vec::new()
    } else {
        existing_tasks
            .iter()
            .filter(|t| ref_ids.iter().any(|r| r == &t.id))
            .collect()
    };
    if selected.is_empty() {
        return String::new();
    }
    selected
        .iter()
        .map(|t| {
            let prompt_body = if !t.custom_prompt.trim().is_empty() {
                truncate_chars_safe(&t.custom_prompt, 900)
            } else if !t.description.trim().is_empty() {
                truncate_chars_safe(&t.description, 700)
            } else {
                "(无指令正文)".to_string()
            };
            let ctr = contracts
                .iter()
                .find(|c| c.from_task_id == t.id)
                .map(|c| {
                    format!(
                        "\n  - 已发布契约《{}》: {}",
                        c.title,
                        truncate_chars_safe(&c.schema_or_api, 700)
                    )
                })
                .unwrap_or_default();
            let excerpt = if !t.last_output_excerpt.trim().is_empty() {
                format!(
                    "\n  - 交付摘录: {}",
                    truncate_chars_safe(&t.last_output_excerpt, 500)
                )
            } else {
                String::new()
            };
            let files = if !t.touched_files.is_empty() {
                format!("\n  - 改动文件: {}", t.touched_files.join(", "))
            } else {
                String::new()
            };
            format!(
                "#### 引用成员 [{}] {} (`{}`)\n  - 原提示词/任务指令:\n{}\n{}{}{}",
                domain_badge_cn(&t.domain),
                if t.project_name.is_empty() {
                    &t.title
                } else {
                    &t.project_name
                },
                t.working_dir,
                prompt_body,
                ctr,
                excerpt,
                files
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// 针对途中/完成后新发现的涉及项目，结合既有计划、接口契约、引用成员提示词与上传图片，由经理生成专属分工与提示词（不改原有成员）。
pub(crate) async fn plan_incremental_tasks(
    run: &TeamRun,
    existing_tasks: &[TeamTask],
    contracts: &[TeamContract],
    candidates: &[IncrementalCandidate],
    user_note: &str,
    current_working_dir: &Path,
) -> IncrementalPlanResult {
    let fallback_tasks: Vec<IncrementalTaskDraft> = candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let dom = normalize_domain(&c.domain);
            let label = if c.name.trim().is_empty() {
                domain_badge_cn(&dom).to_string()
            } else {
                c.name.trim().to_string()
            };
            let note_part = user_note.trim();
            let desc = if !note_part.is_empty() {
                if c.notes.trim().is_empty() {
                    format!("结合项目「{label}」落地补充要求：{note_part}")
                } else {
                    format!("结合项目「{label}」（{}）落地补充要求：{note_part}", c.notes.trim())
                }
            } else if !c.notes.trim().is_empty() {
                format!("围绕当前需求完成「{label}」改造（{}）。", c.notes.trim())
            } else {
                format!(
                    "围绕目标「{}」完成「{label}」的对应改造与对齐。",
                    truncate_chars_safe(&run.user_goal, 120)
                )
            };
            let ref_ctx = build_referenced_member_context(&c.ref_task_ids, existing_tasks, contracts);
            let mut prompt_lines = vec![
                format!("你是负责「{label}」({}) 的跨仓协同工程师，代码目录：`{}`。", domain_badge_cn(&dom), c.working_dir),
                format!("【整体需求目标】：{}", truncate_chars_safe(&run.user_goal, 400)),
                format!("【本成员核心任务】：{desc}"),
            ];
            if !c.images.is_empty() {
                prompt_lines.push(format!("【专属视觉参考图】：本成员附带了 {} 张样式/原型参考图，请严格以本成员附图样式完成页面布局、色彩与交互。", c.images.len()));
            }
            if !ref_ctx.is_empty() {
                prompt_lines.push(format!("【引用既有成员的提示词与产出参考】：\n{ref_ctx}"));
            }
            prompt_lines.push("{{interface_spec_section}}\n\n{{upstream_section}}\n\n{{contract_publish_section}}".to_string());

            IncrementalTaskDraft {
                candidate_index: idx,
                title: format!("{label} · 协同改造"),
                description: desc,
                custom_prompt: prompt_lines.join("\n\n"),
                acceptance_criteria: default_acceptance_for_domain(&dom, &run.mode),
            }
        })
        .collect();

    let fallback = IncrementalPlanResult {
        tasks: fallback_tasks.clone(),
        interface_spec_addition: String::new(),
    };

    if candidates.is_empty() {
        return fallback;
    }
    let Ok(config) = crate::host::load_config() else {
        return fallback;
    };

    let prov_opt = (!run.lead_model.trim().is_empty()).then_some(run.lead_model.trim());
    let mut rt = zap_coding::CodingRuntimeConfig::from_config(
        &config,
        current_working_dir,
        prov_opt,
        None,
        false,
        false,
    );
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
    let sid = format!("team-inc-plan-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid))).await
    else {
        return fallback;
    };

    let existing_desc = if existing_tasks.is_empty() {
        "(none)".to_string()
    } else {
        existing_tasks
            .iter()
            .map(|t| {
                let desc_snip = if !t.custom_prompt.trim().is_empty() {
                    truncate_chars_safe(&t.custom_prompt, 240)
                } else {
                    truncate_chars_safe(&t.description, 200)
                };
                format!(
                    "- id=`{}` | [{} | {}] {} ({})\n  指令摘要: {}",
                    t.id, t.domain, t.status, t.title, t.working_dir, desc_snip
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut all_images: Vec<TeamImage> = Vec::new();
    let cand_desc =
        candidates
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                let ref_ctx =
                    build_referenced_member_context(&c.ref_task_ids, existing_tasks, contracts);
                let mut img_tag = String::from("0 张");
                if !c.images.is_empty() {
                    let start = all_images.len() + 1;
                    for im in &c.images {
                        if all_images.len() >= 8 {
                            break;
                        }
                        all_images.push(im.clone());
                    }
                    let end = all_images.len();
                    if end >= start {
                        img_tag = format!(
                            "{} 张（对应消息附图第 {}~{} 张）",
                            end - start + 1,
                            start,
                            end
                        );
                    }
                }
                format!(
                "- index={idx} | name=`{}` | domain=`{}` | dir=`{}` | 附图={} | 补充要求={}\n{}",
                c.name,
                normalize_domain(&c.domain),
                c.working_dir,
                img_tag,
                if c.notes.trim().is_empty() { "-" } else { c.notes.trim() },
                if ref_ctx.is_empty() {
                    "  (未显式指定引用成员)".to_string()
                } else {
                    format!("  重点引用的既有成员详情:\n{ref_ctx}")
                }
            )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

    let contracts_summary = if contracts.is_empty() {
        "(none)".to_string()
    } else {
        contracts
            .iter()
            .take(6)
            .map(|c| {
                format!(
                    "- [{}] {}: {}",
                    c.domain,
                    c.title,
                    truncate_chars_safe(&c.schema_or_api, 400)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let prompt = format!(
        "You are the Engineering Manager (Lead Planner) in an existing {} run.\n\
Generate tailored task scopes AND complete ready-to-dispatch member prompts (`customPrompt`) for the candidate project(s).\n\
If a candidate references existing members, synthesize the referenced members' prompt conventions, API endpoints, files, and contracts into the new member's `customPrompt`.\n\
If a candidate has attached reference screenshot(s), analyze the visual layout/theme/components in those specific image(s) and spell them out concretely in `customPrompt` (do NOT confuse styles across different candidates).\n\
Keep existing members untouched. Append `{{{{interface_spec_section}}}}\\n\\n{{{{upstream_section}}}}` near the end of `customPrompt` so runtime contracts inject cleanly.\n\
Write text in the user's language. Output ONLY raw JSON (no markdown fences):\n\
{{\"tasks\":[{{\"candidateIndex\":0,\"title\":\"<=28 chars\",\"description\":\"concise summary\",\"customPrompt\":\"complete detailed prompt ready for user editing\",\"acceptanceCriteria\":\"concise check\"}}],\"interfaceSpecAddition\":\"optional concise markdown addendum if new endpoints/fields/integrations are needed, else empty\"}}\n\n\
Goal: {}\n\
Plan: {}\n\
Current Interface Spec: {}\n\
Published Contracts:\n{}\n\
Existing Members:\n{}\n\
Candidate Project(s):\n{}\n\
User Note: {}",
        run.mode,
        truncate_chars_safe(&run.user_goal, 500),
        truncate_chars_safe(&run.execution_plan, 900),
        if run.interface_spec.trim().is_empty() {
            "(none yet)".to_string()
        } else {
            truncate_chars_safe(&run.interface_spec, 1400)
        },
        contracts_summary,
        existing_desc,
        cand_desc,
        if user_note.trim().is_empty() { "(none)" } else { user_note.trim() }
    );

    let lead_memory = build_lead_agent_memory_context(
        &run.lead_agent_id,
        &format!("{} {}", truncate_chars_safe(&run.user_goal, 400), user_note),
    )
    .await;
    let msgs = build_lead_messages(lead_memory, &prompt, &all_images);
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
        extract_json_object(&raw).and_then(|j| serde_json::from_str::<LlmIncrementalOutput>(j).ok())
    };

    match tokio::time::timeout(Duration::from_secs(55), fut).await {
        Ok(Some(out)) if !out.tasks.is_empty() => {
            let mut merged = Vec::with_capacity(candidates.len());
            for (idx, fb) in fallback_tasks.into_iter().enumerate() {
                if let Some(hit) = out
                    .tasks
                    .iter()
                    .find(|t| t.candidate_index == idx)
                    .or_else(|| out.tasks.get(idx))
                {
                    merged.push(IncrementalTaskDraft {
                        candidate_index: idx,
                        title: if hit.title.trim().is_empty() {
                            fb.title
                        } else {
                            hit.title.trim().to_string()
                        },
                        description: if hit.description.trim().is_empty() {
                            fb.description
                        } else {
                            hit.description.trim().to_string()
                        },
                        custom_prompt: if hit.custom_prompt.trim().is_empty() {
                            fb.custom_prompt
                        } else {
                            hit.custom_prompt.trim().to_string()
                        },
                        acceptance_criteria: if hit.acceptance_criteria.trim().is_empty() {
                            fb.acceptance_criteria
                        } else {
                            hit.acceptance_criteria.trim().to_string()
                        },
                    });
                } else {
                    merged.push(fb);
                }
            }
            IncrementalPlanResult {
                tasks: merged,
                interface_spec_addition: out.interface_spec_addition.trim().to_string(),
            }
        }
        _ => fallback,
    }
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RevisionPlanMember {
    pub(crate) task_id: String,
    #[serde(default)]
    pub(crate) selected: bool,
    #[serde(default)]
    pub(crate) role_in_change: String,
    #[serde(default)]
    pub(crate) wave_order: usize,
    #[serde(default)]
    pub(crate) instruction: String,
    #[serde(default)]
    pub(crate) reason: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RevisionPlanResult {
    pub(crate) cascading: bool,
    pub(crate) summary: String,
    #[serde(default)]
    pub(crate) interface_spec_addition: String,
    #[serde(default)]
    pub(crate) requires_sql: bool,
    #[serde(default)]
    pub(crate) members: Vec<RevisionPlanMember>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmRevisionPlanOutput {
    #[serde(default)]
    cascading: bool,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    interface_spec_addition: String,
    #[serde(default)]
    requires_sql: Option<bool>,
    #[serde(default)]
    members: Vec<RevisionPlanMember>,
}

pub(crate) fn revision_implies_sql(instruction: &str) -> bool {
    let low = instruction.to_ascii_lowercase();
    low.contains("建表")
        || low.contains("改表")
        || low.contains("新表")
        || low.contains("加表")
        || low.contains("表结构")
        || low.contains("加字段")
        || low.contains("新增字段")
        || low.contains("增加字段")
        || low.contains("删字段")
        || low.contains("改字段")
        || low.contains("数据库")
        || low.contains("索引")
        || low.contains("外键")
        || low.contains("迁移")
        || low.contains("达梦")
        || low.contains("dm8")
        || low.contains("ddl")
        || low.contains("sql")
        || low.contains("schema")
        || low.contains("alter table")
        || low.contains("create table")
        || low.contains("add column")
        || low.contains("字典表")
        || low.contains("数据表")
}

fn heuristic_revision_plan(
    existing_tasks: &[TeamTask],
    primary_task_id: Option<&str>,
    instruction: &str,
) -> RevisionPlanResult {
    let low = instruction.to_ascii_lowercase();
    let mentions_contract_change = low.contains("接口")
        || low.contains("路径")
        || low.contains("路由")
        || low.contains("url")
        || low.contains("api")
        || low.contains("route")
        || low.contains("path")
        || low.contains("字段")
        || low.contains("参数")
        || low.contains("返回值")
        || low.contains("dto")
        || low.contains("表")
        || low.contains("联动")
        || low.contains("同步");

    let primary_opt = primary_task_id
        .and_then(|pid| existing_tasks.iter().find(|t| t.id == pid))
        .or_else(|| {
            if low.contains("后端") || low.contains("backend") {
                existing_tasks.iter().find(|t| t.domain == "backend")
            } else if low.contains("前端") || low.contains("frontend") {
                existing_tasks.iter().find(|t| t.domain == "frontend")
            } else if low.contains("移动") || low.contains("mobile") {
                existing_tasks.iter().find(|t| t.domain == "mobile")
            } else {
                existing_tasks.first()
            }
        });

    let primary_id = primary_opt.map(|t| t.id.as_str()).unwrap_or("");
    let primary_is_upstream =
        primary_opt.is_some_and(|t| matches!(t.domain.as_str(), "backend" | "sql"));
    let should_cascade =
        mentions_contract_change && (primary_is_upstream || primary_task_id.is_none());

    let mut members = Vec::new();
    let mut cascade_count = 0usize;
    for t in existing_tasks {
        let is_primary = t.id == primary_id;
        let is_downstream_candidate = !is_primary
            && should_cascade
            && (t.depends_on.iter().any(|d| d == primary_id)
                || (primary_is_upstream
                    && matches!(t.domain.as_str(), "frontend" | "mobile" | "qa"))
                || (primary_task_id.is_none()
                    && ((low.contains("前端") && t.domain == "frontend")
                        || (low.contains("移动") && t.domain == "mobile")
                        || (low.contains("后端") && t.domain == "backend"))));
        if is_downstream_candidate {
            cascade_count += 1;
        }
        let selected = is_primary || is_downstream_candidate;
        let role_in_change = if is_primary {
            "primary"
        } else if is_downstream_candidate {
            "cascade"
        } else {
            "unaffected"
        };
        let wave_order = if is_primary {
            1
        } else if is_downstream_candidate {
            2
        } else {
            0
        };
        let inst = if is_primary {
            instruction.trim().to_string()
        } else if is_downstream_candidate {
            format!(
                "联动同步上游变更（原要求：{}）。请对照上游最新发布的接口契约与路径字段，同步修改本仓库中的请求调用、类型定义并校验构建。",
                truncate_chars_safe(instruction, 240)
            )
        } else {
            String::new()
        };
        let reason = if is_primary {
            "主改成员（优先执行）".to_string()
        } else if is_downstream_candidate {
            "涉及接口/契约联动，需等上游改完后同步适配".to_string()
        } else {
            "当前改动暂不波及该成员".to_string()
        };
        members.push(RevisionPlanMember {
            task_id: t.id.clone(),
            selected,
            role_in_change: role_in_change.to_string(),
            wave_order,
            instruction: inst,
            reason,
        });
    }

    RevisionPlanResult {
        cascading: cascade_count > 0,
        summary: if cascade_count > 0 {
            format!(
                "检测到变更可能影响接口/契约，建议先改「{}」，改完后自动联动 {} 位下游成员同步适配。",
                primary_opt
                    .map(|t| if t.project_name.is_empty() { t.title.as_str() } else { t.project_name.as_str() })
                    .unwrap_or("主源成员"),
                cascade_count
            )
        } else {
            "评估为局部独立调整，仅执行所选单个成员改造，不影响其他成员。".to_string()
        },
        interface_spec_addition: String::new(),
        requires_sql: revision_implies_sql(instruction),
        members,
    }
}

/// 由经理（Lead）评估用户提出的修改要求：判断是否会波及其他智能体、编排先后波次，并生成每个受影响成员的增量改造指令。
pub(crate) async fn plan_revision_impact(
    run: &TeamRun,
    existing_tasks: &[TeamTask],
    contracts: &[TeamContract],
    primary_task_id: Option<&str>,
    instruction: &str,
    images: &[TeamImage],
    current_working_dir: &Path,
) -> RevisionPlanResult {
    let clean_inst = instruction.trim();
    let fallback = heuristic_revision_plan(existing_tasks, primary_task_id, clean_inst);
    if existing_tasks.is_empty() || clean_inst.is_empty() {
        return fallback;
    }
    let Ok(config) = crate::host::load_config() else {
        return fallback;
    };

    let prov_opt = (!run.lead_model.trim().is_empty()).then_some(run.lead_model.trim());
    let mut rt = zap_coding::CodingRuntimeConfig::from_config(
        &config,
        current_working_dir,
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
    let sid = format!("team-rev-plan-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid))).await
    else {
        return fallback;
    };

    let members_desc = existing_tasks
        .iter()
        .map(|t| {
            let files = if t.touched_files.is_empty() {
                "-".to_string()
            } else {
                t.touched_files.iter().take(6).cloned().collect::<Vec<_>>().join(", ")
            };
            format!(
                "- taskId=`{}` | domain=`{}` | wave={} | status=`{}` | name=`{}` | title=`{}` | dir=`{}` | touchedFiles=[{}] | excerpt={}",
                t.id,
                t.domain,
                t.wave,
                t.status,
                t.project_name,
                t.title,
                t.working_dir,
                files,
                truncate_chars_safe(&t.last_output_excerpt, 240)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let contracts_desc = if contracts.is_empty() {
        "(none)".to_string()
    } else {
        contracts
            .iter()
            .take(5)
            .map(|c| {
                format!(
                    "- [{}] {}: {}",
                    c.domain,
                    c.title,
                    truncate_chars_safe(&c.schema_or_api, 350)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let anchor_hint = primary_task_id
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("(unspecified - determine from request)");

    let prompt = format!(
        "You are the Engineering Manager evaluating a revision/change request on an existing multi-agent team run.\n\
Determine WHICH team members need to act, whether the change cascades across repositories, and whether it requires database schema / SQL changes.\n\
Rules:\n\
1. If the change is purely internal to one repo (e.g., frontend styling, internal refactor without API/schema/field changes), set `cascading: false` and select ONLY that single member.\n\
2. If the change alters API paths, parameters, response fields, auth permissions, or DB schema, set `cascading: true`: assign the source member `roleInChange: \"primary\"`, `waveOrder: 1` (instructing it to call `team_publish_contract` with the updated contract), and assign affected consumer members (e.g. frontend/mobile) `roleInChange: \"cascade\"`, `waveOrder: 2`.\n\
3. Set `requiresSql: true` ONLY if this revision requires creating/altering database tables, columns, indexes, constraints, or data migration SQL; otherwise set `requiresSql: false`.\n\
4. Include ALL existing taskId entries in `members` (with `selected: false` and `roleInChange: \"unaffected\"` for untouched ones).\n\
Write text in Chinese. Output ONLY raw JSON (no markdown fences):\n\
{{\"cascading\":true/false,\"requiresSql\":true/false,\"summary\":\"1-2 sentence explanation\",\"interfaceSpecAddition\":\"optional addendum if shared spec changed, else empty\",\"members\":[{{\"taskId\":\"...\",\"selected\":true/false,\"roleInChange\":\"primary|cascade|unaffected\",\"waveOrder\":1,\"instruction\":\"specific revision instruction for this member\",\"reason\":\"why affected or unaffected\"}}]}}\n\n\
Goal: {}\n\
Current Interface Spec: {}\n\
Published Contracts:\n{}\n\
Team Members:\n{}\n\
Primary Anchor TaskId: {}\n\
User Change Request: {}",
        truncate_chars_safe(&run.user_goal, 400),
        truncate_chars_safe(&run.interface_spec, 1200),
        contracts_desc,
        members_desc,
        anchor_hint,
        clean_inst
    );

    let lead_memory = build_lead_agent_memory_context(
        &run.lead_agent_id,
        &format!(
            "{} {}",
            truncate_chars_safe(&run.user_goal, 400),
            clean_inst
        ),
    )
    .await;
    let msgs = build_lead_messages(lead_memory, &prompt, images);
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
        extract_json_object(&raw)
            .and_then(|j| serde_json::from_str::<LlmRevisionPlanOutput>(j).ok())
    };

    match tokio::time::timeout(Duration::from_secs(45), fut).await {
        Ok(Some(out)) if !out.members.is_empty() => {
            let mut merged_members = Vec::with_capacity(existing_tasks.len());
            for t in existing_tasks {
                if let Some(hit) = out.members.iter().find(|m| m.task_id == t.id) {
                    merged_members.push(RevisionPlanMember {
                        task_id: t.id.clone(),
                        selected: hit.selected,
                        role_in_change: if hit.role_in_change.trim().is_empty() {
                            if hit.selected {
                                "primary".to_string()
                            } else {
                                "unaffected".to_string()
                            }
                        } else {
                            hit.role_in_change.clone()
                        },
                        wave_order: hit.wave_order,
                        instruction: if hit.selected && hit.instruction.trim().is_empty() {
                            clean_inst.to_string()
                        } else {
                            hit.instruction.clone()
                        },
                        reason: hit.reason.clone(),
                    });
                } else if let Some(fb) = fallback.members.iter().find(|m| m.task_id == t.id) {
                    merged_members.push(fb.clone());
                }
            }
            if !merged_members.iter().any(|m| m.selected) {
                return fallback;
            }
            RevisionPlanResult {
                cascading: out.cascading
                    || merged_members.iter().filter(|m| m.selected).count() > 1,
                summary: if out.summary.trim().is_empty() {
                    fallback.summary
                } else {
                    out.summary
                },
                interface_spec_addition: out.interface_spec_addition.trim().to_string(),
                requires_sql: out.requires_sql.unwrap_or(fallback.requires_sql),
                members: merged_members,
            }
        }
        _ => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{HostModelInfo, TeamHost};
    use std::path::PathBuf;
    use tokio_util::sync::CancellationToken;

    struct DummyHost;

    #[async_trait::async_trait]
    impl TeamHost for DummyHost {
        async fn current_working_dir(&self) -> PathBuf {
            PathBuf::from("/workspace")
        }

        fn list_projects(&self) -> Vec<HostProjectInfo> {
            Vec::new()
        }

        fn hash_path(&self, path: &Path) -> String {
            format!("hash_{}", path.to_string_lossy().len())
        }

        fn list_models(&self) -> (String, Vec<HostModelInfo>) {
            ("gpt-5".to_string(), Vec::new())
        }

        async fn run_child_turn(
            &self,
            _req: crate::host::HostChildTurnRequest,
            _cancel: CancellationToken,
        ) -> Result<(String, Vec<String>), String> {
            Ok((String::new(), Vec::new()))
        }
    }

    fn test_agents() -> Vec<NeuralAgent> {
        vec![
            NeuralAgent {
                id: "xiaoli".into(),
                icon: "🎓".into(),
                name: "学渣".into(),
                label: "🎓 学渣".into(),
                color: "#14b8a6".into(),
                hint: String::new(),
                system_prompt: String::new(),
            },
            NeuralAgent {
                id: "backend".into(),
                icon: "⚙".into(),
                name: "后端大白".into(),
                label: "⚙ 后端大白".into(),
                color: "#eab308".into(),
                hint: String::new(),
                system_prompt: String::new(),
            },
            NeuralAgent {
                id: "frontend".into(),
                icon: "🎨".into(),
                name: "前端小白".into(),
                label: "🎨 前端小白".into(),
                color: "#ec4899".into(),
                hint: String::new(),
                system_prompt: String::new(),
            },
        ]
    }

    #[test]
    fn custom_project_without_agent_falls_back_to_domain_agent_and_skips_unrelated_first_agent() {
        let agents = test_agents();
        assert_eq!(
            guess_agent_for_domain("frontend", &agents),
            Some("frontend".into())
        );
        assert_eq!(guess_agent_for_domain("qa", &agents), None);

        let custom_projects = vec![
            TeamProjectPreset {
                id: "prj-fe".into(),
                name: "GMZ".into(),
                domain: "frontend".into(),
                working_dir: "D:/GMZ".into(),
                agent_id: "none".into(),
                avatar_id: "frontend_artist".into(),
                model: String::new(),
                notes: String::new(),
                sort_order: 0,
                updated_at: String::new(),
            },
            TeamProjectPreset {
                id: "prj-qa".into(),
                name: "QA".into(),
                domain: "qa".into(),
                working_dir: "D:/QA".into(),
                agent_id: "none".into(),
                avatar_id: "qa_detective".into(),
                model: String::new(),
                notes: String::new(),
                sort_order: 1,
                updated_at: String::new(),
            },
        ];
        let raw_tasks = vec![
            LlmPlannedTask {
                key: "t1".into(),
                project_id: "prj-fe".into(),
                domain: "frontend".into(),
                title: "GMZ".into(),
                description: "前端任务".into(),
                acceptance_criteria: String::new(),
                depends_on: Vec::new(),
            },
            LlmPlannedTask {
                key: "t2".into(),
                project_id: "prj-qa".into(),
                domain: "qa".into(),
                title: "QA".into(),
                description: "验收".into(),
                acceptance_criteria: String::new(),
                depends_on: Vec::new(),
            },
        ];

        let tasks = materialize_planned_tasks(
            &DummyHost,
            "run_1",
            "测试",
            "pipeline",
            "gpt-5",
            Path::new("/workspace"),
            raw_tasks,
            &custom_projects,
            &[],
            &agents,
            &[],
            None,
        );
        assert_eq!(tasks[0].agent_id, "frontend");
        assert_eq!(tasks[0].agent_label, "🎨 前端小白");
        assert_eq!(tasks[1].agent_id, "none");
    }
}
