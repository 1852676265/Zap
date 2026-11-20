//! `zap-agent-teams` 原生团队协作工具集（仅对团队子会话按需挂载，普通聊天零侵入）：
//! - `team_publish_contract`：向上游账本发布结构化 SQL DDL / REST API / TS 类型契约，自动注入下游波次。
//! - `team_ask_peer`：跨仓库只读问询其他职能同事（SQL / Backend / Frontend），检索对方代码库与契约并即时作答。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use zap_kernel::message::Message;
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;
use zap_kernel::tool::{Tool, ToolContext, ToolResult};

use super::store::{
    append_event_conn, load_contracts_for_run_conn, load_run_conn, load_tasks_for_run_conn,
    upsert_contract_conn, with_team_store,
};
use super::types::{domain_badge_cn, normalize_domain, truncate_chars_safe};

#[derive(Debug, Clone)]
pub(crate) struct ActiveTeamSessionCtx {
    pub(crate) run_id: String,
    pub(crate) task_id: String,
    pub(crate) domain: String,
    pub(crate) actor_label: String,
    pub(crate) provider_name: String,
    pub(crate) read_only_mode: bool,
}

fn active_team_sessions() -> &'static Mutex<HashMap<String, ActiveTeamSessionCtx>> {
    static MAP: OnceLock<Mutex<HashMap<String, ActiveTeamSessionCtx>>> = OnceLock::new();
    MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn register_team_session_ctx(session_id: &str, ctx: ActiveTeamSessionCtx) {
    let mut guard = active_team_sessions()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.insert(session_id.trim().to_string(), ctx);
}

pub(crate) fn unregister_team_session_ctx(session_id: &str) {
    let mut guard = active_team_sessions()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.remove(session_id.trim());
}

pub(crate) fn get_team_session_ctx(session_id: &str) -> Option<ActiveTeamSessionCtx> {
    let guard = active_team_sessions()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.get(session_id.trim()).cloned()
}

/// 在 `process_chat_request` 组装 `CodingRuntimeConfig` 时按需挂载团队协作工具。
pub fn apply_session_team_tools(
    runtime: &mut zap_coding::CodingRuntimeConfig,
    session_id: Option<&str>,
) {
    let Some(sid) = session_id.map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };
    let Some(ctx) = get_team_session_ctx(sid) else {
        return;
    };

    if ctx.read_only_mode {
        runtime.mcp = false;
        runtime.shared_mcp_registry = None;
        runtime
            .extra_tools
            .retain(|tool| team_readonly_tool(tool.name()));
        runtime.extra_hooks.push(Arc::new(TeamReadonlyHook));
    }

    if !ctx.read_only_mode
        && !runtime
            .extra_tools
            .iter()
            .any(|t| t.name() == "team_publish_contract")
    {
        runtime
            .extra_tools
            .push(Arc::new(TeamPublishContractTool::new(sid.to_string())) as Arc<dyn Tool>);
    }

    if !runtime
        .extra_tools
        .iter()
        .any(|t| t.name() == "team_ask_peer")
    {
        runtime
            .extra_tools
            .push(Arc::new(TeamAskPeerTool::new(sid.to_string())) as Arc<dyn Tool>);
    }
}

fn team_readonly_tool(name: &str) -> bool {
    matches!(
        name,
        "code_explore" | "grep" | "read" | "glob" | "team_ask_peer" | "memory_recall"
    )
}

struct TeamReadonlyHook;

#[async_trait]
impl zap_kernel::hook::LifecycleHooks for TeamReadonlyHook {
    async fn on_model_response(&self, response: &mut Message) {
        // Enforced independently of customizable system text; deny shell and unknown tools.
        response
            .tool_calls
            .retain(|call| team_readonly_tool(&call.name));
    }
}

// ---------------------------------------------------------------------------
// Tool 1: team_publish_contract
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PublishContractArgs {
    title: String,
    schema_or_api: String,
    #[serde(default)]
    notes: String,
}

pub struct TeamPublishContractTool {
    session_id: String,
}

impl TeamPublishContractTool {
    pub fn new(session_id: String) -> Self {
        Self { session_id }
    }
}

#[async_trait]
impl Tool for TeamPublishContractTool {
    fn name(&self) -> &str {
        "team_publish_contract"
    }

    fn description(&self) -> &str {
        "Publish or update a structured technical contract (SQL DDL schema, REST/GraphQL API route & payload specification, TypeScript interface, or shared constants) so downstream wave teammates receive the exact specification automatically."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Concise title of the contract (e.g. 'RBAC Tables DDL' or 'POST /api/v1/orders Contract')"
                },
                "schema_or_api": {
                    "type": "string",
                    "description": "Exact DDL statements, HTTP method + path + request/response JSON fields, status codes, or shared type definitions"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional implementation notes, auth headers, pagination rules, or edge cases for downstream teammates"
                }
            },
            "required": ["title", "schema_or_api"]
        })
    }

    fn read_only_hint(&self) -> bool {
        false
    }

    async fn execute(&self, args: &str, _ctx: &ToolContext) -> ToolResult {
        let parsed: PublishContractArgs = match serde_json::from_str(args) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    call_id: String::new(),
                    content: format!("Invalid team_publish_contract arguments: {e}"),
                    is_error: true,
                    images: Vec::new(),
                };
            }
        };

        let Some(team_ctx) = get_team_session_ctx(&self.session_id) else {
            return ToolResult {
                call_id: String::new(),
                content: "No active team session context bound to this session.".to_string(),
                is_error: true,
                images: Vec::new(),
            };
        };

        if parsed.title.trim().is_empty() || parsed.schema_or_api.trim().is_empty() {
            return ToolResult {
                call_id: String::new(),
                content: "Both `title` and `schema_or_api` must be non-empty.".to_string(),
                is_error: true,
                images: Vec::new(),
            };
        }

        let res = tokio::task::spawn_blocking(move || {
            with_team_store(|store| {
                let contract = upsert_contract_conn(
                    &store.conn,
                    &team_ctx.run_id,
                    &team_ctx.task_id,
                    &team_ctx.domain,
                    &parsed.title,
                    &parsed.schema_or_api,
                    &parsed.notes,
                )?;
                let summary = format!(
                    "{} 发布了协作契约《{}》",
                    team_ctx.actor_label, contract.title
                );
                let _ = append_event_conn(
                    &store.conn,
                    &team_ctx.run_id,
                    "contract_published",
                    &team_ctx.task_id,
                    &team_ctx.actor_label,
                    &summary,
                    &contract.schema_or_api,
                );
                Ok(contract)
            })
        })
        .await;

        match res {
            Ok(Ok(c)) => ToolResult {
                call_id: String::new(),
                content: format!(
                    "Contract '{}' (id={}) recorded in the shared team ledger. Downstream dependent tasks will receive it automatically.",
                    c.title, c.id
                ),
                is_error: false,
                images: Vec::new(),
            },
            Ok(Err(e)) => ToolResult {
                call_id: String::new(),
                content: format!("Failed to publish contract: {e}"),
                is_error: true,
                images: Vec::new(),
            },
            Err(e) => ToolResult {
                call_id: String::new(),
                content: format!("Contract task join error: {e}"),
                is_error: true,
                images: Vec::new(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tool 2: team_ask_peer
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AskPeerArgs {
    target_domain: String,
    question: String,
    #[serde(default)]
    search_hint: String,
}

pub struct TeamAskPeerTool {
    session_id: String,
}

impl TeamAskPeerTool {
    pub fn new(session_id: String) -> Self {
        Self { session_id }
    }
}

#[async_trait]
impl Tool for TeamAskPeerTool {
    fn name(&self) -> &str {
        "team_ask_peer"
    }

    fn description(&self) -> &str {
        "Ask a cross-repository teammate (sql, backend, frontend, or general) a focused technical question to inspect their codebase or published contracts in read-only mode (e.g. verify exact field names, response envelope format, or table columns)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "target_domain": {
                    "type": "string",
                    "description": "Target teammate domain ('sql', 'backend', 'frontend', 'general') or exact task id"
                },
                "question": {
                    "type": "string",
                    "description": "Specific technical question to verify against the peer repository and contracts"
                },
                "search_hint": {
                    "type": "string",
                    "description": "Optional identifier, route path, table name, or filename keyword to locate relevant files in the peer repository"
                }
            },
            "required": ["target_domain", "question"]
        })
    }

    fn read_only_hint(&self) -> bool {
        true
    }

    async fn execute(&self, args: &str, _ctx: &ToolContext) -> ToolResult {
        let parsed: AskPeerArgs = match serde_json::from_str(args) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    call_id: String::new(),
                    content: format!("Invalid team_ask_peer arguments: {e}"),
                    is_error: true,
                    images: Vec::new(),
                };
            }
        };

        let Some(team_ctx) = get_team_session_ctx(&self.session_id) else {
            return ToolResult {
                call_id: String::new(),
                content: "No active team session context bound to this session.".to_string(),
                is_error: true,
                images: Vec::new(),
            };
        };

        let target_raw = parsed.target_domain.trim().to_string();
        let norm_target = normalize_domain(&target_raw);
        let question = parsed.question.trim().to_string();
        let search_hint = parsed.search_hint.trim().to_string();

        if question.is_empty() {
            return ToolResult {
                call_id: String::new(),
                content: "`question` must be non-empty.".to_string(),
                is_error: true,
                images: Vec::new(),
            };
        }

        let ctx_for_lookup = team_ctx.clone();
        let q_for_log = question.clone();
        let target_domain_lookup = norm_target.clone();
        let lookup = tokio::task::spawn_blocking(move || {
            with_team_store(|store| {
                let run = load_run_conn(&store.conn, &ctx_for_lookup.run_id)?
                    .ok_or_else(|| "Active team run no longer exists".to_string())?;
                let tasks = load_tasks_for_run_conn(&store.conn, &ctx_for_lookup.run_id)?;
                let contracts = load_contracts_for_run_conn(&store.conn, &ctx_for_lookup.run_id)?;
                let peer_task = tasks
                    .iter()
                    .find(|t| t.id == target_raw)
                    .or_else(|| {
                        tasks.iter().find(|t| {
                            t.domain == target_domain_lookup && t.id != ctx_for_lookup.task_id
                        })
                    })
                    .or_else(|| tasks.iter().find(|t| t.domain == target_domain_lookup))
                    .cloned();

                let target_badge = peer_task
                    .as_ref()
                    .map(|t| {
                        if t.agent_label.is_empty() {
                            domain_badge_cn(&t.domain).to_string()
                        } else {
                            format!("{} ({})", domain_badge_cn(&t.domain), t.agent_label)
                        }
                    })
                    .unwrap_or_else(|| domain_badge_cn(&target_domain_lookup).to_string());

                let _ = append_event_conn(
                    &store.conn,
                    &ctx_for_lookup.run_id,
                    "peer_question",
                    &ctx_for_lookup.task_id,
                    &ctx_for_lookup.actor_label,
                    &format!(
                        "{} 向 {} 发起跨库问询：{}",
                        ctx_for_lookup.actor_label,
                        target_badge,
                        truncate_chars_safe(&q_for_log, 90)
                    ),
                    &q_for_log,
                );
                Ok((run, peer_task, contracts, target_badge))
            })
        })
        .await;

        let (run, peer_task, contracts, target_badge) = match lookup {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                return ToolResult {
                    call_id: String::new(),
                    content: format!("Peer lookup failed: {e}"),
                    is_error: true,
                    images: Vec::new(),
                };
            }
            Err(e) => {
                return ToolResult {
                    call_id: String::new(),
                    content: format!("Peer lookup join error: {e}"),
                    is_error: true,
                    images: Vec::new(),
                };
            }
        };

        let peer_dir = peer_task
            .as_ref()
            .map(|t| PathBuf::from(t.working_dir.trim()))
            .filter(|p| !p.as_os_str().is_empty() && p.is_dir());

        let hint_owned = search_hint.clone();
        let q_owned = question.clone();
        let peer_dir_clone = peer_dir.clone();
        let code_evidence = tokio::task::spawn_blocking(move || {
            if let Some(dir) = peer_dir_clone {
                scan_peer_repo_evidence(&dir, &hint_owned, &q_owned)
            } else {
                "No local working directory bound for this peer task.".to_string()
            }
        })
        .await
        .unwrap_or_default();

        let relevant_contracts: Vec<String> = contracts
            .into_iter()
            .filter(|c| {
                c.domain == norm_target
                    || peer_task.as_ref().is_some_and(|pt| pt.id == c.from_task_id)
            })
            .map(|c| {
                format!(
                    "### {} ({})\n{}\n{}",
                    c.title, c.domain, c.schema_or_api, c.notes
                )
            })
            .collect();

        let peer_excerpt = peer_task
            .as_ref()
            .map(|t| t.last_output_excerpt.clone())
            .unwrap_or_default();

        let contracts_text = relevant_contracts.join("\n\n");
        let prompt = match super::runner::execution_prompt(
            &run,
            super::prompts::TPL_PEER_CONSULTATION,
            Some(&norm_target),
            &[
                ("target_badge", &target_badge),
                ("question", &question),
                ("contracts_text", &contracts_text),
                ("peer_excerpt", &peer_excerpt),
                ("code_evidence", &code_evidence),
            ],
        )
        .await
        {
            Ok(prompt) => prompt,
            Err(e) => {
                return ToolResult {
                    call_id: String::new(),
                    content: format!("Peer template lookup failed: {e}"),
                    is_error: true,
                    images: Vec::new(),
                }
            }
        };
        let answer = synthesize_peer_answer(
            &team_ctx.provider_name,
            peer_dir.as_deref(),
            &prompt,
            &contracts_text,
            &peer_excerpt,
            &code_evidence,
        )
        .await;

        let run_id_log = team_ctx.run_id.clone();
        let task_id_log = peer_task
            .as_ref()
            .map(|t| t.id.clone())
            .unwrap_or_else(|| team_ctx.task_id.clone());
        let badge_log = target_badge.clone();
        let ans_log = answer.clone();
        let _ = tokio::task::spawn_blocking(move || {
            with_team_store(|store| {
                let _ = append_event_conn(
                    &store.conn,
                    &run_id_log,
                    "peer_answer",
                    &task_id_log,
                    &badge_log,
                    &format!("{} 回复了跨库问询", badge_log),
                    &ans_log,
                );
                Ok(())
            })
        })
        .await;

        ToolResult {
            call_id: String::new(),
            content: format!("Peer Consultation Answer from [{target_badge}]:\n{answer}"),
            is_error: false,
            images: Vec::new(),
        }
    }
}

fn scan_peer_repo_evidence(root: &Path, search_hint: &str, question: &str) -> String {
    let mut keywords: Vec<String> = Vec::new();
    for raw in search_hint
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .chain(
            question
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '/'),
        )
    {
        let trimmed = raw.trim();
        if trimmed.len() >= 3
            && !keywords.iter().any(|k| k.eq_ignore_ascii_case(trimmed))
            && !matches!(
                trimmed.to_ascii_lowercase().as_str(),
                "what"
                    | "where"
                    | "which"
                    | "does"
                    | "have"
                    | "with"
                    | "from"
                    | "that"
                    | "this"
                    | "api"
                    | "the"
                    | "and"
                    | "for"
            )
        {
            keywords.push(trimmed.to_string());
            if keywords.len() >= 6 {
                break;
            }
        }
    }

    let mut snippets = Vec::new();
    let mut visited_files = 0usize;
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        if visited_files >= 350 || snippets.len() >= 8 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.')
                || matches!(
                    name.as_str(),
                    "node_modules"
                        | "target"
                        | "dist"
                        | "build"
                        | "vendor"
                        | ".git"
                        | "__pycache__"
                )
            {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(
                ext.as_str(),
                "rs" | "ts"
                    | "tsx"
                    | "js"
                    | "jsx"
                    | "vue"
                    | "py"
                    | "go"
                    | "java"
                    | "sql"
                    | "prisma"
                    | "graphql"
                    | "toml"
                    | "yaml"
                    | "yml"
                    | "json"
            ) {
                continue;
            }
            visited_files += 1;
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.len() > 256 * 1024 {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");

            let file_matched_kw = keywords.iter().find(|kw| {
                rel.to_ascii_lowercase().contains(&kw.to_ascii_lowercase())
                    || content.contains(kw.as_str())
            });

            if let Some(kw) = file_matched_kw {
                let mut matched_lines = Vec::new();
                let lines: Vec<&str> = content.lines().collect();
                for (idx, line) in lines.iter().enumerate() {
                    if line.contains(kw.as_str())
                        || line.to_ascii_lowercase().contains(&kw.to_ascii_lowercase())
                    {
                        let start = idx.saturating_sub(2);
                        let end = (idx + 5).min(lines.len());
                        let block = lines[start..end]
                            .iter()
                            .enumerate()
                            .map(|(off, l)| format!("L{}: {}", start + off + 1, l))
                            .collect::<Vec<_>>()
                            .join("\n");
                        matched_lines.push(block);
                        if matched_lines.len() >= 2 {
                            break;
                        }
                    }
                }
                let preview = if matched_lines.is_empty() {
                    lines.into_iter().take(18).collect::<Vec<_>>().join("\n")
                } else {
                    matched_lines.join("\n---\n")
                };
                snippets.push(format!("File `{rel}` (matched `{kw}`):\n{preview}"));
                if snippets.len() >= 8 {
                    break;
                }
            }
        }
    }

    if snippets.is_empty() {
        format!(
            "Scanned {visited_files} source files in `{}` with keywords {:?}, no exact text hits.",
            root.display(),
            keywords
        )
    } else {
        truncate_chars_safe(&snippets.join("\n\n"), 3200)
    }
}

async fn synthesize_peer_answer(
    provider_name: &str,
    peer_dir: Option<&Path>,
    prompt: &str,
    contracts_text: &str,
    peer_excerpt: &str,
    code_evidence: &str,
) -> String {
    let fallback = format!(
        "Published Contracts:\n{}\n\nRecent Peer Output:\n{}\n\nCodebase Evidence:\n{}",
        if contracts_text.trim().is_empty() {
            "(none yet)"
        } else {
            contracts_text
        },
        if peer_excerpt.trim().is_empty() {
            "(none yet)"
        } else {
            peer_excerpt
        },
        code_evidence
    );

    let Ok(config) = crate::api_config::load_config() else {
        return truncate_chars_safe(&fallback, 2200);
    };
    let wd = peer_dir.unwrap_or_else(|| Path::new("."));
    let prov_opt = (!provider_name.trim().is_empty()).then_some(provider_name.trim());
    let mut rt =
        zap_coding::CodingRuntimeConfig::from_config(&config, wd, prov_opt, None, false, false);
    rt.mcp = false;
    rt.max_tokens = Some(4096);
    let coding_cfg = crate::kernel_runtime::coding_config_from_runtime(&rt);
    let options = ChatOptions {
        max_tokens: Some(4096),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory = crate::runtime_host::coding_provider_factory();
    let ephemeral_sid = format!("team-peer-ask-{}", uuid::Uuid::new_v4());
    let Ok(Ok(provider)) =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&ephemeral_sid))).await
    else {
        return truncate_chars_safe(&fallback, 2200);
    };

    let msgs = vec![Message::user(prompt)];
    let call_fut = async {
        let mut stream = provider.chat_stream(&msgs, &[], &options).await.ok()?;
        let mut text = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(delta) => text.push_str(&delta),
                StreamEvent::Done { .. } | StreamEvent::Error(_) => break,
                _ => {}
            }
        }
        (!text.trim().is_empty()).then_some(text)
    };

    match tokio::time::timeout(Duration::from_secs(25), call_fut).await {
        Ok(Some(ans)) => truncate_chars_safe(&ans, 2200),
        _ => truncate_chars_safe(&fallback, 2200),
    }
}
