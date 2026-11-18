//! `zap-agent-teams` 智能拆单规划器（AI Lead Planner + 领域智能体与项目目录自动预填 + 排障转修复流水线）。

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use futures::StreamExt;
use serde::Deserialize;
use zap_kernel::message::Message;
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;

use crate::neural_memory::NeuralAgent;
use crate::ProjectInfo;

use super::store::{
    append_event_conn, create_run_with_tasks_conn, load_config_conn, load_role_presets_conn,
    load_run_detail_conn, with_team_store,
};
use super::types::{
    compute_and_assign_waves, domain_badge_cn, new_team_id, normalize_domain, now_ts,
    truncate_chars_safe, RoleBindingPreset, TeamRun, TeamRunDetail, TeamTask,
};

#[derive(Debug, Deserialize)]
struct LlmPlanOutput {
    #[serde(default)]
    title: String,
    #[serde(default)]
    tasks: Vec<LlmPlannedTask>,
}

#[derive(Debug, Deserialize)]
struct LlmPlannedTask {
    #[serde(default)]
    key: String,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    acceptance_criteria: String,
    #[serde(default)]
    depends_on: Vec<String>,
}

/// 根据用户输入的需求（或线上报错），自动拆解跨领域 DAG 任务草稿并预填项目目录与隔离记忆智能体。
pub async fn plan_team_run(
    user_goal: &str,
    mode: &str,
    lead_model_override: Option<&str>,
    current_working_dir: &Path,
    source_triage_id: Option<&str>,
    inherited_bindings: Option<&HashMap<String, (String, String, String)>>,
) -> Result<TeamRunDetail, String> {
    let goal = user_goal.trim();
    if goal.is_empty() {
        return Err("请输入需要拆解的任务目标或故障现象".to_string());
    }

    let norm_mode = if mode.trim().eq_ignore_ascii_case("triage") {
        "triage"
    } else {
        "pipeline"
    };

    let (global_cfg, presets, agents, projects) = tokio::task::spawn_blocking(|| {
        let (cfg, presets) =
            with_team_store(|s| Ok((load_config_conn(&s.conn), load_role_presets_conn(&s.conn)?)))?;
        let agents = crate::neural_memory::list_neural_agents().unwrap_or_default();
        let projects = crate::list_projects().unwrap_or_default();
        Ok::<_, String>((cfg, presets, agents, projects))
    })
    .await
    .map_err(|e| format!("加载规划上下文失败: {e}"))??;

    let config = crate::api_config::load_config()?;
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
        goal,
        norm_mode,
        &projects,
    )
    .await
    .unwrap_or_else(|| build_heuristic_plan(goal, norm_mode));

    let run_id = new_team_id("run");
    let now = now_ts();
    let run_title = if llm_plan.title.trim().is_empty() {
        default_run_title(goal, norm_mode)
    } else {
        truncate_chars_safe(llm_plan.title.trim(), 72)
    };

    let raw_tasks = if llm_plan.tasks.is_empty() {
        build_heuristic_plan(goal, norm_mode).tasks
    } else {
        llm_plan.tasks
    };

    let preset_map: HashMap<String, RoleBindingPreset> = presets
        .into_iter()
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
    let fallback_wd = current_working_dir.to_string_lossy().replace('\\', "/");

    for (idx, (tid, item)) in allocated.into_iter().enumerate() {
        let dom = normalize_domain(&item.domain);
        let deps: Vec<String> = if norm_mode == "triage" {
            // 跨库联合会诊模式：第一阶段所有领域智能体并发只读勘察各自仓库
            Vec::new()
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

        let (wd, aid, task_model) = resolve_role_binding(
            &dom,
            goal,
            &fallback_wd,
            &lead_model,
            &preset_map,
            inherited_bindings,
            &agents,
            &projects,
        );

        let agent_label = agents
            .iter()
            .find(|a| a.id == aid)
            .map(|a| a.label.clone())
            .unwrap_or_default();
        let proj_hash = if wd.trim().is_empty() {
            String::new()
        } else {
            crate::hash_path(Path::new(&wd))
        };

        let title = if item.title.trim().is_empty() {
            format!("{} 子任务 #{}", domain_badge_cn(&dom), idx + 1)
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
            run_id: run_id.clone(),
            domain: dom,
            title,
            description: item.description.trim().to_string(),
            acceptance_criteria: acceptance,
            depends_on: deps,
            wave: 1,
            sort_order: idx,
            working_dir: wd,
            project_hash: proj_hash,
            agent_id: aid,
            agent_label,
            model: task_model,
            session_id: String::new(),
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
        // 若模型偶发输出环路依赖，退化为按列表顺序线性或无环依赖再重算波次
        for t in &mut tasks {
            t.depends_on.clear();
        }
        let _ = compute_and_assign_waves(&mut tasks);
    }

    let run = TeamRun {
        id: run_id.clone(),
        title: run_title,
        user_goal: goal.to_string(),
        mode: norm_mode.to_string(),
        status: "draft".to_string(),
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
                "跨库联合排障会诊单"
            } else {
                "研发协作派工单"
            };
            let _ = append_event_conn(
                &store.conn,
                &run.id,
                "run_planned",
                "",
                "lead_planner",
                &format!(
                    "已生成{}草稿（共 {} 个子任务），请核对各领域项目目录与专属智能体后启动执行。",
                    mode_text,
                    tasks.len()
                ),
                "",
            );
            load_run_detail_conn(&store.conn, &run.id, 0)?
                .ok_or_else(|| "创建派工单后读取失败".to_string())
        })
    })
    .await
    .map_err(|e| format!("保存派工草稿失败: {e}"))??;

    Ok(detail)
}

/// 将已完成的「三方跨库联合排障会诊单」一键转化为「跨仓库修复接力派工单」，自动继承项目目录与智能体绑定。
pub async fn convert_triage_to_pipeline(
    triage_run_id: &str,
    current_working_dir: &Path,
) -> Result<TeamRunDetail, String> {
    let tid = triage_run_id.trim().to_string();
    let triage_detail = tokio::task::spawn_blocking(move || {
        with_team_store(|store| load_run_detail_conn(&store.conn, &tid, 0))
    })
    .await
    .map_err(|e| e.to_string())??
    .ok_or_else(|| "未找到对应的排障会诊单".to_string())?;

    let mut inherited: HashMap<String, (String, String, String)> = HashMap::new();
    for t in &triage_detail.tasks {
        inherited.insert(
            normalize_domain(&t.domain),
            (t.working_dir.clone(), t.agent_id.clone(), t.model.clone()),
        );
    }

    let combined_goal = if triage_detail.run.synthesis_report.trim().is_empty() {
        format!("根据排障需求修复问题：\n{}", triage_detail.run.user_goal)
    } else {
        format!(
            "请基于以下跨库联合会诊根因结论落地修复方案：\n\n【原始问题】：\n{}\n\n【联合会诊结论与修复建议】：\n{}",
            triage_detail.run.user_goal,
            truncate_chars_safe(&triage_detail.run.synthesis_report, 3000)
        )
    };

    plan_team_run(
        &combined_goal,
        "pipeline",
        Some(&triage_detail.run.lead_model),
        current_working_dir,
        Some(&triage_detail.run.id),
        Some(&inherited),
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
    projects: &[ProjectInfo],
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

fn guess_project_for_domain(domain: &str, goal: &str, projects: &[ProjectInfo]) -> Option<String> {
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

async fn call_planner_llm(
    config: &zap_config::Config,
    lead_model: &str,
    working_dir: &Path,
    goal: &str,
    mode: &str,
    projects: &[ProjectInfo],
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
    rt.max_tokens = Some(2400);
    let coding_cfg = crate::kernel_runtime::coding_config_from_runtime(&rt);
    let options = ChatOptions {
        max_tokens: Some(2400),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory = crate::runtime_host::coding_provider_factory();
    let sid = format!("team-planner-{}", uuid::Uuid::new_v4());
    let provider = tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&sid)))
        .await
        .ok()?
        .ok()?;

    let known_repos = if projects.is_empty() {
        "(none)".to_string()
    } else {
        projects
            .iter()
            .take(15)
            .map(|p| {
                format!(
                    "- {} ({})",
                    p.name,
                    p.working_dir.to_string_lossy().replace('\\', "/")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mode_instruction = if mode == "triage" {
        "Mode is `triage` (Cross-Repository Joint Troubleshooting). Decompose the incident/bug into 2 to 4 parallel read-only investigation tasks across relevant domains (`sql`, `backend`, `frontend`, `general`) with `depends_on: []` so all domain specialists inspect their own codebases concurrently."
    } else {
        "Mode is `pipeline` (Cross-Repository Feature/Bugfix Delivery). Decompose the user requirement into 1 to 5 focused tasks across ONLY the domains actually needed (`sql`, `backend`, `frontend`, `general`). Use `depends_on` (referencing `key`) whenever a downstream domain relies on an upstream schema or API contract (e.g. `backend` depends on `sql`; `frontend` depends on `backend`). Tasks without dependencies run concurrently in Wave 1."
    };

    let prompt = format!(
        "You are an expert multi-repository engineering lead planner.\n\
        {mode_instruction}\n\
        Write `title`, `description`, and `acceptance_criteria` in the SAME language used by the user.\n\
        Output ONLY a valid JSON object (no markdown fences, no extra prose) matching this exact schema:\n\
        {{\n\
          \"title\": \"Concise overall plan title (<= 36 chars)\",\n\
          \"tasks\": [\n\
            {{\n\
              \"key\": \"t1\",\n\
              \"domain\": \"sql | backend | frontend | general\",\n\
              \"title\": \"Clear task title\",\n\
              \"description\": \"Detailed implementation or investigation instructions, including what contract to publish or consume\",\n\
              \"acceptance_criteria\": \"Concrete verifiable checklist (build/syntax check, contract completeness, edge cases)\",\n\
              \"depends_on\": []\n\
            }}\n\
          ]\n\
        }}\n\n\
        Known Local Projects (for context):\n{known_repos}\n\n\
        User Goal:\n{goal}"
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
        extract_json_object(&raw).and_then(|j| serde_json::from_str::<LlmPlanOutput>(j).ok())
    };

    tokio::time::timeout(Duration::from_secs(40), fut)
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

fn build_heuristic_plan(goal: &str, mode: &str) -> LlmPlanOutput {
    let low = goal.to_ascii_lowercase();
    let has_sql = low.contains("sql")
        || low.contains("表")
        || low.contains("字段")
        || low.contains("数据库")
        || low.contains("schema")
        || low.contains("ddl")
        || low.contains("索引");
    let has_back = low.contains("后端")
        || low.contains("接口")
        || low.contains("api")
        || low.contains("server")
        || low.contains("service")
        || low.contains("controller");
    let has_front = low.contains("前端")
        || low.contains("页面")
        || low.contains("界面")
        || low.contains("按钮")
        || low.contains("表单")
        || low.contains("web")
        || low.contains("ui")
        || low.contains("vue")
        || low.contains("react");

    let mut tasks = Vec::new();
    if mode == "triage" {
        for (idx, (dom, label)) in [
            ("sql", "SQL/数据库层排查数据表结构、慢查询与脏数据线索"),
            ("backend", "后端服务层排查接口路由、业务逻辑与异常堆栈"),
            ("frontend", "前端界面层排查请求传参、响应解包与状态渲染"),
        ]
        .iter()
        .enumerate()
        {
            tasks.push(LlmPlannedTask {
                key: format!("t{}", idx + 1),
                domain: (*dom).to_string(),
                title: (*label).to_string(),
                description: format!(
                    "针对问题「{goal}」，只读勘察当前仓库相关实现并输出关键代码证据与根因判断。"
                ),
                acceptance_criteria: default_acceptance_for_domain(dom, "triage"),
                depends_on: Vec::new(),
            });
        }
        return LlmPlanOutput {
            title: default_run_title(goal, mode),
            tasks,
        };
    }

    let mut prev_key: Option<String> = None;
    if has_sql {
        let k = "t_sql".to_string();
        tasks.push(LlmPlannedTask {
            key: k.clone(),
            domain: "sql".to_string(),
            title: "设计并落地数据库表结构与迁移脚本".to_string(),
            description: format!("围绕需求「{goal}」梳理并编写 SQL 表结构/索引变更，完成后调用 team_publish_contract 发布完整 DDL 契约。"),
            acceptance_criteria: default_acceptance_for_domain("sql", "pipeline"),
            depends_on: Vec::new(),
        });
        prev_key = Some(k);
    }
    if has_back || (has_sql && has_front) {
        let k = "t_backend".to_string();
        tasks.push(LlmPlannedTask {
            key: k.clone(),
            domain: "backend".to_string(),
            title: "实现后端接口、业务逻辑与数据访问层".to_string(),
            description: format!("围绕需求「{goal}」对齐数据库契约实现后端接口与校验逻辑，完成后调用 team_publish_contract 发布请求路径与响应 JSON 契约。"),
            acceptance_criteria: default_acceptance_for_domain("backend", "pipeline"),
            depends_on: prev_key.iter().cloned().collect(),
        });
        prev_key = Some(k);
    }
    if has_front {
        tasks.push(LlmPlannedTask {
            key: "t_frontend".to_string(),
            domain: "frontend".to_string(),
            title: "实现前端页面组件与后端接口联调对接".to_string(),
            description: format!(
                "围绕需求「{goal}」严格按照上游后端发布的 API 契约实现前端页面与交互状态。"
            ),
            acceptance_criteria: default_acceptance_for_domain("frontend", "pipeline"),
            depends_on: prev_key.iter().cloned().collect(),
        });
    }

    if tasks.is_empty() {
        tasks.push(LlmPlannedTask {
            key: "t_general".to_string(),
            domain: "general".to_string(),
            title: "执行核心研发改造任务".to_string(),
            description: goal.to_string(),
            acceptance_criteria: default_acceptance_for_domain("general", "pipeline"),
            depends_on: Vec::new(),
        });
    }

    LlmPlanOutput {
        title: default_run_title(goal, mode),
        tasks,
    }
}
