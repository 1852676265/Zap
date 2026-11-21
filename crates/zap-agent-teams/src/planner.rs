//! `zap-agent-teams` 智能拆单规划器（AI Lead Planner + 领域智能体与项目目录自动预填 + 排障转修复流水线）。

use std::collections::{BTreeMap, HashMap};

use super::prompts::{
    render_template_single_pass, resolve_effective_template, TPL_LEAD_INTERFACE_SPEC,
    TPL_LEAD_PLANNER_PIPELINE, TPL_LEAD_PLANNER_TRIAGE,
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
    path_to_clean_str, truncate_chars_safe, RoleBindingPreset, TeamImage, TeamProjectPreset,
    TeamRun, TeamRunDetail, TeamTask,
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

/// 立即创建 `pipeline` 模式的初始任务单（`stage = "planning"`, `status = "running"`），随后由后台异步跑阶段 1+2。
pub async fn create_pending_pipeline_run(
    host: &dyn TeamHost,
    user_goal: &str,
    lead_model_override: Option<&str>,
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

    let (global_cfg, custom_projects) = tokio::task::spawn_blocking(|| {
        with_team_store(|s| {
            Ok((
                load_config_conn(&s.conn),
                load_custom_projects_conn(&s.conn)?,
            ))
        })
    })
    .await
    .map_err(|e| format!("加载任务配置失败: {e}"))??;

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
            agent_id: p.agent_id.clone(),
            agent_label: String::new(),
            model: p.model.clone(),
            session_id: String::new(),
            custom_prompt: String::new(),
            status: "pending".to_string(),
            attempt: 0,
            attempt_id: String::new(),
            repair_count: 0,
            contract_summary: String::new(),
            last_output_excerpt: String::new(),
            verify_verdict: String::new(),
            verify_feedback: String::new(),
            touched_files: Vec::new(),
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
                "- [{}] {} (项目目录: `{}`, 分工: {})",
                domain_badge_cn(&t.domain),
                if t.project_name.is_empty() {
                    &t.title
                } else {
                    &t.project_name
                },
                t.working_dir,
                t.description
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let fallback = format!(
        "### 前后端统一接口与联调约定\n\n#### 1. 参考执行计划\n{}\n\n#### 2. 已确认数据库表结构\n{}\n\n#### 3. 协作分工\n{}\n\n请后端与各前端严格保持 REST 路径、JSON 字段驼峰命名、字典码值与权限字符串完全一致。",
        if run.execution_plan.is_empty() { "见任务描述" } else { &run.execution_plan },
        if run.sql_schema_draft.is_empty() { "（本次不涉及新建/修改表结构）" } else { &run.sql_schema_draft },
        task_summary
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

    let global_templates = tokio::task::spawn_blocking(|| {
        with_team_store(|s| Ok(load_config_conn(&s.conn).prompt_templates))
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or_default();
    let (template, _) = resolve_effective_template(
        TPL_LEAD_INTERFACE_SPEC,
        None,
        Some(&run.prompt_overrides),
        &global_templates,
    );
    let prompt = render_template_single_pass(
        &template,
        &[
            ("user_goal", &run.user_goal),
            ("execution_plan", &run.execution_plan),
            ("sql_schema_draft", &sql_part),
            ("task_summary", &task_summary),
        ],
    );

    let msgs = vec![build_user_message(&prompt, &run.goal_images)];
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

    match tokio::time::timeout(Duration::from_secs(75), fut).await {
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
pub async fn plan_team_run(
    host: &std::sync::Arc<dyn TeamHost>,
    user_goal: &str,
    mode: &str,
    lead_model_override: Option<&str>,
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

    let llm_plan = call_planner_llm(
        &config,
        &lead_model,
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
            let chosen_aid = if cp.agent_id.trim().is_empty() {
                "none".to_string()
            } else {
                cp.agent_id.trim().to_string()
            };
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
            status: "pending".to_string(),
            attempt: 0,
            attempt_id: String::new(),
            repair_count: 0,
            contract_summary: String::new(),
            last_output_excerpt: String::new(),
            verify_verdict: String::new(),
            verify_feedback: String::new(),
            touched_files: Vec::new(),
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

    let aid = preset
        .map(|p| p.agent_id.trim().to_string())
        .filter(|s| !s.is_empty() && s != "none" && agents.iter().any(|a| a.id == *s))
        .or_else(|| guess_agent_for_domain(domain, agents))
        .unwrap_or_else(|| "none".to_string());

    let mdl = preset
        .map(|p| p.model.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| lead_model.to_string());

    (wd, aid, mdl)
}

fn guess_agent_for_domain(domain: &str, agents: &[NeuralAgent]) -> Option<String> {
    let patterns: &[&str] = match domain {
        "sql" => &["sql", "db", "data", "mysql", "postgres", "数据库", "数仓"],
        "backend" => &[
            "backend", "server", "api", "rust", "java", "go", "后端", "服务",
        ],
        "frontend" => &["frontend", "web", "ui", "vue", "react", "前端", "界面"],
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
    agents.first().map(|a| a.id.clone())
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
        projects
            .iter()
            .take(15)
            .map(|p| format!("- {} ({})", p.name, path_to_clean_str(&p.working_dir)))
            .collect::<Vec<_>>()
            .join("\n")
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

    let msgs = vec![build_user_message(&prompt, images)];
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
        return "定位涉及的真实文件路径、行号与可疑根因，输出结构化排查证据且不修改任何业务代码。"
            .to_string();
    }
    match domain {
        "sql" => "完成表结构/索引/迁移脚本设计，确保字段类型与约束完整，并通过 team_publish_contract 发布准确 DDL 契约。".to_string(),
        "backend" => "严格对齐上游表结构契约实现接口与业务逻辑，完成代码编译/语法校验，并通过 team_publish_contract 发布准确接口请求与响应契约。".to_string(),
        "frontend" => "严格对齐后端接口契约完成页面交互、状态流转与异常处理，确保前端类型与语法构建无报错。".to_string(),
        _ => "按任务描述完成全部改动并通过基础语法或构建检查。".to_string(),
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
