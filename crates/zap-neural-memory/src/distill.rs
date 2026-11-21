//! 阶段整理与持久化草稿队列引擎：
//! - 聊天当下 0 负担轻写入冷存档 + `draft_turns` 持久化草稿队列（软件重启不丢）
//! - 5 个自然触发点：静默防抖（90s）/ 攒满 5 轮 / 切换智能体或会话 / 压缩前抢救 / 启动恢复与手动立刻整理
//! - 「明确记住」快车道：检测到“记住/牢记/记下来/以后都”等指令时不等防抖，立刻插队整理落盘
//! - Team 子任务跳过执行期噪音，改由 `team_bridge` 在验收完结时精准回流
//! - 写入前先查出相关旧记忆全文（含完整要点与 ID），强制关闭深度思考以提速省费，失败绝不降级写客套废话

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};
use zap_config::Config;
use zap_kernel::message::{Message, Role};
use zap_kernel::provider::{ChatOptions, ReasoningEffort, ToolChoice};
use zap_kernel::stream::StreamEvent;

use super::reconcile::{apply_fallback_distill_conn, apply_llm_distill_conn, DistillOutput};
use super::redact::redact_secrets;
use super::retrieval::query_nodes_ranked_conn;
use super::search::{
    analyze_query, clean_user_prompt, is_slash_command, is_trivial_turn, ProjectScope,
};
use super::store::{
    bump_draft_retry_conn, delete_draft_turns_conn, enqueue_draft_turn_conn, get_cached_config,
    get_config_conn, load_agents_conn, load_all_pending_draft_groups_conn,
    load_open_tasks_scoped_conn, load_pending_drafts_for_scope_conn, load_profiles_scoped_conn,
    load_raw_turn_conn, load_recent_nodes_scoped_conn, prune_retention_conn,
    resolve_agent_for_session_cached, upsert_raw_turn_conn, with_read, with_write,
};
use super::team_bridge::{is_team_managed_session, lazy_sync_team_deliveries_if_changed};
use super::types::{
    normalize_project_key, normalize_project_name, now_ts_minute, DraftTurn, NeuralAgent,
    NeuralNode, NeuralTask, ProfileFact, RawTurn, DISTILL_TIMEOUT, DRAFT_BATCH_TRIGGER_TURNS,
    DRAFT_DEBOUNCE_SECS,
};

fn agent_distill_lock(agent_id: &str) -> Arc<tokio::sync::Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> = OnceLock::new();
    let map = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
    Arc::clone(
        guard
            .entry(agent_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
    )
}

fn debounce_gen_map() -> &'static Mutex<HashMap<(String, String), u64>> {
    static GENS: OnceLock<Mutex<HashMap<(String, String), u64>>> = OnceLock::new();
    GENS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_debounce_token(agent_id: &str, session_id: &str) -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let tok = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut guard = debounce_gen_map().lock().unwrap_or_else(|e| e.into_inner());
    guard.insert((agent_id.to_string(), session_id.to_string()), tok);
    tok
}

fn is_current_debounce_token(agent_id: &str, session_id: &str, tok: u64) -> bool {
    let guard = debounce_gen_map().lock().unwrap_or_else(|e| e.into_inner());
    guard.get(&(agent_id.to_string(), session_id.to_string())) == Some(&tok)
}

/// 仅当用户输入本身明确表达长期记住/规则/偏好意图时，才允许自动提炼写入 `pin = true` 并走立刻整理快车道。
pub(crate) fn user_explicitly_requests_pin(user_text: &str) -> bool {
    let low = user_text.to_lowercase();
    const PIN_INTENTS: &[&str] = &[
        "记住",
        "牢记",
        "记下来",
        "帮我记",
        "永远",
        "以后都",
        "每次都",
        "必须遵守",
        "核心规范",
        "置顶",
        "always remember",
        "remember this",
        "note this down",
        "pin this",
        "never forget",
        "from now on",
        "strict rule",
    ];
    PIN_INTENTS.iter().any(|p| low.contains(p))
}

fn parse_distill_json(raw: &str) -> Option<DistillOutput> {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<DistillOutput>(trimmed) {
        return Some(v);
    }
    let stripped = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if let Ok(v) = serde_json::from_str::<DistillOutput>(stripped) {
        return Some(v);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end > start {
        serde_json::from_str::<DistillOutput>(&trimmed[start..=end]).ok()
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) fn extract_last_turn_payload(
    messages: &[Message],
) -> Option<(String, String, Vec<String>)> {
    extract_last_turn_payload_with_idx(messages).map(|(_, u, a, f)| (u, a, f))
}

fn extract_last_turn_payload_with_idx(
    messages: &[Message],
) -> Option<(usize, String, String, Vec<String>)> {
    let last_user_idx = messages
        .iter()
        .rposition(|m| m.role == Role::User && !m.synthetic)?;
    let user_text = clean_user_prompt(&messages[last_user_idx].text);
    if user_text.is_empty() {
        return None;
    }

    let mut ai_parts: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    for msg in &messages[last_user_idx + 1..] {
        if msg.role == Role::Assistant && !msg.synthetic {
            let t = msg.text.trim();
            if !t.is_empty() {
                ai_parts.push(t.to_string());
            }
            for call in &msg.tool_calls {
                if let Ok(v) = serde_json::from_str::<Value>(&call.arguments) {
                    for key in ["path", "file_path", "target_file"] {
                        if let Some(p) = v.get(key).and_then(|x| x.as_str()) {
                            let clean = p.trim();
                            if !clean.is_empty() && !files.contains(&clean.to_string()) {
                                files.push(clean.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    let ai_text = ai_parts.join("\n\n");
    if ai_text.trim().is_empty() {
        return None;
    }
    Some((last_user_idx, user_text, ai_text, files))
}

/// 生成跨进程稳定的确定性回合 ID，用于防止多事件路径或重放重复触发同一回合。
pub(crate) fn deterministic_turn_id(
    agent_id: &str,
    session_id: &str,
    last_user_idx: usize,
    user_text: &str,
    ai_text: &str,
) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    };
    feed(agent_id.as_bytes());
    feed(session_id.as_bytes());
    feed(&last_user_idx.to_le_bytes());
    feed(user_text.as_bytes());
    feed(ai_text.as_bytes());
    format!("turn_{h:016x}")
}

/// 截断长文本时同时保留头部上下文与末尾最终结论，避免丢失回合收尾结果。
pub(crate) fn excerpt_head_tail(text: &str, head_chars: usize, tail_chars: usize) -> String {
    let total = text.chars().count();
    if total <= head_chars + tail_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(head_chars).collect();
    let tail_start_char = total.saturating_sub(tail_chars);
    let tail: String = text.chars().skip(tail_start_char).collect();
    format!("{head}\n...[truncated middle]...\n{tail}")
}

/// 进程启动恢复一次：把上次软件退出前尚未整理完的持久化草稿异步接着整理完毕。
pub(crate) fn spawn_boot_draft_recovery_once() {
    static RECOVERED: AtomicBool = AtomicBool::new(false);
    if cfg!(test) || RECOVERED.swap(true, Ordering::AcqRel) {
        return;
    }
    let Ok(rt) = tokio::runtime::Handle::try_current() else {
        return;
    };
    rt.spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let _ = tokio::task::spawn_blocking(lazy_sync_team_deliveries_if_changed).await;
        let _ = flush_all_pending_drafts_async(None).await;
    });
}

/// 回合结束入口（0 阻塞）：
/// 1. 跳过 Team 子会话（改由 `team_bridge` 在验收完结时回流）与通用模式（`none`）；
/// 2. 轻量写入冷层 `raw_turns` + 持久化草稿队列 `draft_turns`（<1ms，重启不丢）；
/// 3. 命中“明确记住”或攒满 5 轮立刻触发阶段整理，否则重置 90s 静默防抖定时器。
pub fn maybe_capture_turn_from_snapshot(
    session_id: &str,
    working_dir: &Path,
    messages: &[Message],
    provider_name: &str,
) {
    let sid_trimmed = session_id.trim();
    if sid_trimmed.is_empty() || is_team_managed_session(sid_trimmed) {
        return;
    }
    let Ok(rt_handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let Some((last_user_idx, user_text_raw, ai_text_raw, touched_files)) =
        extract_last_turn_payload_with_idx(messages)
    else {
        return;
    };
    if is_slash_command(&user_text_raw)
        || is_trivial_turn(&user_text_raw, &ai_text_raw, &touched_files)
    {
        return;
    }

    let sid = sid_trimmed.to_string();
    let wd_owned = working_dir.to_path_buf();
    let provider_owned = provider_name.to_string();
    let explicit_remember = user_explicitly_requests_pin(&user_text_raw);

    rt_handle.spawn(async move {
        let sid_c = sid.clone();
        let wd_c = wd_owned.clone();
        let prov_c = provider_owned.clone();

        let enqueued = tokio::task::spawn_blocking(move || {
            let user_text = redact_secrets(&user_text_raw);
            let ai_text = redact_secrets(&ai_text_raw);
            let cfg = get_cached_config()?;
            if !cfg.timeline_auto_capture || cfg.distill_mode == "off" {
                return Ok::<_, String>(None);
            }
            let Some(agent) = resolve_agent_for_session_cached(&sid_c)? else {
                return Ok::<_, String>(None);
            };
            let turn_id =
                deterministic_turn_id(&agent.id, &sid_c, last_user_idx, &user_text, &ai_text);
            let ts = now_ts_minute();
            let project = normalize_project_name(&wd_c);
            let project_key = normalize_project_key(&wd_c);
            let wd_str = wd_c.to_string_lossy().into_owned();

            let pending_count = with_write(|store| {
                if load_raw_turn_conn(&store.conn, &turn_id)
                    .ok()
                    .flatten()
                    .is_some()
                {
                    return Ok(0usize);
                }
                upsert_raw_turn_conn(
                    &store.conn,
                    &RawTurn {
                        turn_id: turn_id.clone(),
                        ts: ts.clone(),
                        agent: agent.id.clone(),
                        project: project.clone(),
                        project_key: project_key.clone(),
                        working_dir: wd_str.clone(),
                        session_id: sid_c.clone(),
                        user_text: user_text.clone(),
                        ai_text: ai_text.clone(),
                        touched_files: touched_files.clone(),
                    },
                )?;
                if cfg.distill_mode == "rule_only" || cfg.distill_mode == "auto_silent" {
                    apply_fallback_distill_conn(
                        &store.conn,
                        &agent.id,
                        &sid_c,
                        &turn_id,
                        &ts,
                        &project,
                        &wd_str,
                        &user_text,
                        &ai_text,
                        &touched_files,
                        "rule_only",
                    )?;
                    prune_retention_conn(&store.conn, &agent.id);
                    return Ok(0usize);
                }
                enqueue_draft_turn_conn(
                    &store.conn,
                    &DraftTurn {
                        id: 0,
                        turn_id,
                        session_id: sid_c.clone(),
                        agent_id: agent.id.clone(),
                        project,
                        project_key,
                        working_dir: wd_str,
                        user_text,
                        ai_text,
                        touched_files,
                        provider_name: prov_c,
                        explicit_remember,
                        retry_count: 0,
                        created_at: ts,
                    },
                )?;
                let list = load_pending_drafts_for_scope_conn(&store.conn, &agent.id, &sid_c)?;
                Ok(list.len())
            })?;
            Ok(Some((agent.id, pending_count)))
        })
        .await;

        let Ok(Ok(Some((agent_id, pending_count)))) = enqueued else {
            return;
        };
        if pending_count == 0 {
            return;
        }

        if explicit_remember || pending_count >= DRAFT_BATCH_TRIGGER_TURNS {
            let reason = if explicit_remember {
                "explicit_remember"
            } else {
                "batch_threshold"
            };
            let _ = next_debounce_token(&agent_id, &sid);
            consolidate_scope_drafts(&agent_id, &sid, reason).await;
        } else {
            let tok = next_debounce_token(&agent_id, &sid);
            let aid_t = agent_id.clone();
            let sid_t = sid.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(DRAFT_DEBOUNCE_SECS)).await;
                if is_current_debounce_token(&aid_t, &sid_t, tok) {
                    consolidate_scope_drafts(&aid_t, &sid_t, "idle_debounce").await;
                }
            });
        }
    });
}

/// 在切换智能体、切换会话或触发上下文压缩前，立刻把该会话尚未整理的草稿异步落盘抄正本。
pub fn flush_session_drafts_async(session_id: &str, agent_hint: Option<&str>) {
    let sid = session_id.trim();
    if sid.is_empty() || cfg!(test) {
        return;
    }
    let Ok(rt) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let sid_owned = sid.to_string();
    let aid_owned = agent_hint.map(|s| s.trim().to_string());
    rt.spawn(async move {
        let groups: Vec<(String, String)> = tokio::task::spawn_blocking(move || {
            with_read(|conn, _| {
                if let Some(ref aid) = aid_owned {
                    if !aid.is_empty() && aid != "none" {
                        let pending = load_pending_drafts_for_scope_conn(conn, aid, &sid_owned)?;
                        return Ok(if pending.is_empty() {
                            Vec::new()
                        } else {
                            vec![(aid.clone(), sid_owned.clone())]
                        });
                    }
                }
                let all = load_all_pending_draft_groups_conn(conn, None)?;
                Ok(all.into_iter().filter(|(_, s)| s == &sid_owned).collect())
            })
        })
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();

        for (aid, s) in groups {
            let _ = next_debounce_token(&aid, &s);
            consolidate_scope_drafts(&aid, &s, "session_boundary").await;
        }
    });
}

/// 立即整理所有（或指定智能体）的未落盘草稿（供手动按钮 `neural:flush-drafts` 与启动恢复调用）。
pub(crate) async fn flush_all_pending_drafts_async(
    agent_filter: Option<&str>,
) -> Result<Value, String> {
    let filter_owned = agent_filter
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "all")
        .map(str::to_string);
    let groups = tokio::task::spawn_blocking(move || {
        with_read(|conn, _| load_all_pending_draft_groups_conn(conn, filter_owned.as_deref()))
    })
    .await
    .map_err(|e| e.to_string())??;

    let mut flushed_batches = 0usize;
    let mut flushed_turns = 0usize;
    for (aid, sid) in groups {
        let _ = next_debounce_token(&aid, &sid);
        let cnt = consolidate_scope_drafts(&aid, &sid, "manual_flush").await;
        if cnt > 0 {
            flushed_batches += 1;
            flushed_turns += cnt;
        }
    }
    Ok(json!({
        "success": true,
        "flushedBatches": flushed_batches,
        "flushedTurns": flushed_turns,
    }))
}

/// 对单个 `(agent_id, session_id)` 的全部待整理草稿执行一次完整的「先查旧记忆全文 → 融合/作废/新增 → 原子落库」。
async fn consolidate_scope_drafts(agent_id: &str, session_id: &str, trigger_reason: &str) -> usize {
    let lock = agent_distill_lock(agent_id);
    let _serial = lock.lock().await;

    let aid_q = agent_id.to_string();
    let sid_q = session_id.to_string();
    let loaded = tokio::task::spawn_blocking(move || {
        with_read(|conn, _| {
            let drafts = load_pending_drafts_for_scope_conn(conn, &aid_q, &sid_q)?;
            if drafts.is_empty() {
                return Ok(None);
            }
            let cfg = get_config_conn(conn);
            let agents = load_agents_conn(conn)?;
            let Some(agent) = agents.into_iter().find(|a| a.id == aid_q) else {
                return Ok(None);
            };
            let last = drafts.last().expect("non-empty drafts");
            let pkey = last.project_key.clone();
            let pname = last.project.clone();

            let open_tasks = load_open_tasks_scoped_conn(conn, &aid_q, &pkey, &pname, 12)?;
            let profiles = load_profiles_scoped_conn(conn, &aid_q, &pkey, &pname, 24)?;

            let mut combined_query = String::new();
            for d in &drafts {
                combined_query.push_str(&d.user_text);
                combined_query.push(' ');
                for f in &d.touched_files {
                    combined_query.push_str(f);
                    combined_query.push(' ');
                }
            }
            let plan = analyze_query(&combined_query, &[]);
            let scope = if pkey.is_empty() && pname.is_empty() {
                None
            } else {
                Some(ProjectScope {
                    key: pkey.as_str(),
                    name: pname.as_str(),
                })
            };
            let mut candidate_nodes: Vec<NeuralNode> =
                query_nodes_ranked_conn(conn, &aid_q, &plan, scope, 8)
                    .into_iter()
                    .map(|rn| rn.node)
                    .collect();
            let mut seen_nids: HashSet<String> =
                candidate_nodes.iter().map(|n| n.id.clone()).collect();
            for recent in load_recent_nodes_scoped_conn(conn, &aid_q, &pkey, &pname, 10) {
                if candidate_nodes.len() >= 12 {
                    break;
                }
                if seen_nids.insert(recent.id.clone()) {
                    candidate_nodes.push(recent);
                }
            }

            Ok(Some((
                cfg,
                agent,
                drafts,
                open_tasks,
                profiles,
                candidate_nodes,
            )))
        })
    })
    .await;

    let Ok(Ok(Some((cfg, agent, drafts, open_tasks, profiles, existing_nodes)))) = loaded else {
        return 0;
    };

    let draft_ids: Vec<i64> = drafts.iter().map(|d| d.id).collect();
    let turn_count = drafts.len();
    let allow_pin = drafts.iter().any(|d| d.explicit_remember);
    let last_draft = drafts.last().expect("non-empty drafts").clone();

    let mut merged_files: Vec<String> = Vec::new();
    for d in &drafts {
        for f in &d.touched_files {
            if !merged_files.contains(f) {
                merged_files.push(f.clone());
            }
        }
    }

    let Ok(config_loaded) = Config::load(&Config::default_path()) else {
        tracing::warn!("neural_memory: failed to load config during draft consolidation");
        return 0;
    };

    let distill_res = run_batch_llm_distill(
        &config_loaded,
        &last_draft.provider_name,
        &cfg.distill_model,
        Path::new(&last_draft.working_dir),
        &agent,
        &last_draft.project,
        &open_tasks,
        &profiles,
        &existing_nodes,
        &drafts,
        &merged_files,
    )
    .await;

    match distill_res {
        Some(parsed) => {
            let aid_w = agent.id.clone();
            let sid_w = session_id.to_string();
            let ids_w = draft_ids.clone();
            let should_record = parsed.should_record;
            let commit_res = tokio::task::spawn_blocking(move || {
                with_write(|store| {
                    if should_record {
                        apply_llm_distill_conn(
                            &mut store.conn,
                            &aid_w,
                            &sid_w,
                            &last_draft.turn_id,
                            &last_draft.created_at,
                            &last_draft.project,
                            &last_draft.working_dir,
                            &last_draft.user_text,
                            &last_draft.ai_text,
                            &merged_files,
                            parsed,
                            allow_pin,
                        )?;
                        prune_retention_conn(&store.conn, &aid_w);
                    }
                    delete_draft_turns_conn(&store.conn, &ids_w);
                    Ok(())
                })
            })
            .await;

            if let Ok(Ok(())) = commit_res {
                tracing::info!(
                    agent = %agent_id,
                    session = %session_id,
                    turns = turn_count,
                    recorded = should_record,
                    trigger = trigger_reason,
                    "neural_memory: consolidated draft batch"
                );
                return turn_count;
            }
            0
        }
        None => {
            // 绝不降级把 AI 第一句客套废话写进热库：允许有限重试，超限后清空草稿（冷层 raw_turns 已安全归档）
            let max_retries = drafts.iter().map(|d| d.retry_count).max().unwrap_or(0);
            let ids_w = draft_ids;
            let _ = tokio::task::spawn_blocking(move || {
                with_write(|store| {
                    if max_retries >= 2 {
                        delete_draft_turns_conn(&store.conn, &ids_w);
                    } else {
                        bump_draft_retry_conn(&store.conn, &ids_w);
                    }
                    Ok(())
                })
            })
            .await;
            tracing::warn!(
                agent = %agent_id,
                session = %session_id,
                turns = turn_count,
                retries = max_retries,
                "neural_memory: batch LLM consolidation failed; skipped noisy fallback write"
            );
            0
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_batch_llm_distill(
    config: &Config,
    provider_name: &str,
    distill_model: &str,
    working_dir: &Path,
    agent: &NeuralAgent,
    project: &str,
    open_tasks: &[NeuralTask],
    profiles: &[ProfileFact],
    existing_nodes: &[NeuralNode],
    drafts: &[DraftTurn],
    touched_files: &[String],
) -> Option<DistillOutput> {
    let custom_model = distill_model.trim();
    let provider_override = (!provider_name.trim().is_empty()).then_some(provider_name.trim());
    let mut runtime =
        if !custom_model.is_empty() && config.resolve_model(Some(custom_model)).is_ok() {
            zap_coding::CodingRuntimeConfig::from_config(
                config,
                working_dir,
                Some(custom_model),
                None,
                false,
                false,
            )
        } else {
            let mut rt = zap_coding::CodingRuntimeConfig::from_config(
                config,
                working_dir,
                provider_override,
                None,
                false,
                false,
            );
            if !custom_model.is_empty() {
                rt.model = custom_model.to_string();
            }
            rt
        };
    runtime.mcp = false;
    runtime.max_tokens = Some(2600);

    let coding_cfg = runtime.agent_config();
    // 强制关闭高档位深度思考：结构化 JSON 整理不需要长时间思维链，速度提升 3~5 倍且大幅省费
    let options = ChatOptions {
        max_tokens: Some(2600),
        temperature: None,
        tool_choice: ToolChoice::None,
        reasoning_effort: Some(ReasoningEffort::Low),
        ..coding_cfg.chat_options.clone()
    };

    let factory: Arc<dyn zap_coding::CodingProviderFactory> = Arc::new(
        zap_coding::DefaultCodingProviderFactory::new(zap_auth::JEIKCODE_USER_AGENT),
    );
    let ephemeral_sid = format!("neural-distill-{}", uuid::Uuid::new_v4());
    let provider =
        tokio::task::spawn_blocking(move || factory.build(&coding_cfg, Some(&ephemeral_sid)))
            .await
            .ok()?
            .ok()?;

    let tasks_desc = if open_tasks.is_empty() {
        "none".to_string()
    } else {
        open_tasks
            .iter()
            .map(|t| {
                format!(
                    "- id={} | kind={} | status={} | title={} | progress={} | remaining={} | next={}",
                    t.id, t.item_kind, t.status, t.title, t.progress_note, t.remaining_note, t.next_step
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let profiles_desc = if profiles.is_empty() {
        "none".to_string()
    } else {
        profiles
            .iter()
            .map(|p| {
                format!(
                    "- id={} | kind={} | {}{}",
                    p.id,
                    p.kind,
                    if p.pinned { "[pinned] " } else { "" },
                    p.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let nodes_desc = if existing_nodes.is_empty() {
        "none".to_string()
    } else {
        existing_nodes
            .iter()
            .map(|n| {
                let aliases = if n.aliases.is_empty() {
                    String::new()
                } else {
                    format!(" | aliases=[{}]", n.aliases.join(", "))
                };
                let facts_lines: Vec<String> = n
                    .facts
                    .iter()
                    .filter(|f| f.is_active())
                    .take(6)
                    .map(|f| format!("    * ({}/{}) {}", f.id, f.kind, f.text))
                    .collect();
                if facts_lines.is_empty() {
                    format!(
                        "- id={} | name={}{aliases} | summary={}",
                        n.id, n.name, n.summary
                    )
                } else {
                    format!(
                        "- id={} | name={}{aliases} | summary={}\n{}",
                        n.id,
                        n.name,
                        n.summary,
                        facts_lines.join("\n")
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    let mut turns_block = String::new();
    let per_turn_user_budget = (1800 / drafts.len().max(1)).clamp(450, 1200);
    let per_turn_ai_budget = (2800 / drafts.len().max(1)).clamp(600, 1800);
    for (idx, d) in drafts.iter().enumerate() {
        let u_ex = excerpt_head_tail(
            &d.user_text,
            per_turn_user_budget * 2 / 3,
            per_turn_user_budget / 3,
        );
        let a_ex = excerpt_head_tail(
            &d.ai_text,
            per_turn_ai_budget / 3,
            per_turn_ai_budget * 2 / 3,
        );
        turns_block.push_str(&format!(
            "### Turn {} [{}]\nUser:\n{}\nAssistant (untrusted outcome excerpt):\n{}\n\n",
            idx + 1,
            d.created_at,
            u_ex,
            a_ex
        ));
    }

    let files_str = if touched_files.is_empty() {
        "none".to_string()
    } else {
        touched_files.join(", ")
    };

    let role_hint = format!(
        "{} | {}",
        agent.hint,
        excerpt_head_tail(&agent.system_prompt, 240, 120)
    );

    // 纯英文内核整理提示词（符合 AGENTS.md §8 Agent Core Purity；保留用户实体与内容的原始语言）
    let prompt = format!(
        "You are a high-precision memory consolidation engine. Output ONLY a valid JSON object (no markdown fences, no commentary).\n\
Active Agent: {} ({})\n\
Agent Focus: {}\n\
Current Project: {}\n\n\
=== EXISTING ACTIVE RULES / PREFERENCES (Drawer 1) ===\n{}\n\n\
=== EXISTING OPEN REQUIREMENTS / STUDY PLANS (Drawer 2) ===\n{}\n\n\
=== EXISTING RELEVANT KNOWLEDGE CARDS WITH ACTIVE FACTS (Drawer 3) ===\n{}\n\n\
=== CONVERSATION SEGMENT TO CONSOLIDATE ({} turn(s)) ===\n{}\n\
Touched Files: {}\n\n\
CONSOLIDATION RULES:\n\
1. Set `should_record: false` if this segment is pure chitchat, undecided back-and-forth discussion with no conclusion, or trivial code reading with no durable lesson.\n\
2. Adapt naturally to the agent's role: for study/learning agents capture mastery progress, concepts understood, sticking points, and next study plan (`item_kind: \"study\"`); for engineering agents capture requirement progress, finalized decisions, pitfalls/fixes, SQL business conventions, and API contracts.\n\
3. REUSE existing cards via `target_node_id` whenever the topic matches an existing card or alias. NEVER overwrite prior knowledge: use `fact_ops` (`add`, `merge` with existing `fact_id`, or `supersede` with obsolete `fact_id`) and write a re-synthesized `summary` covering ALL active facts.\n\
4. When a new decision contradicts an existing rule or fact, explicitly `supersede` the old ID so old and new conclusions never collide.\n\
5. For every card, provide 4-8 bilingual colloquial `search_tags` (synonyms, Chinese/English aliases, common ways the user might ask about it later) and exact identifiers/file names/table names/error codes in facts.\n\
6. Produce at most ONE concise milestone `timeline` entry summarizing the real outcome of this segment.\n\n\
Return JSON matching this exact schema:\n\
{{\n\
  \"should_record\": true/false,\n\
  \"profile_actions\": [{{\"action\": \"add|update|supersede|remove\", \"target_id\": \"existing profile id or empty\", \"text\": \"self-contained durable rule/preference\", \"pin\": true/false, \"kind\": \"rule|preference|domain_convention|study_pref\"}}],\n\
  \"task_actions\": [{{\"action\": \"create|update|complete\", \"task_id\": \"existing task id or empty\", \"title\": \"concise title\", \"item_kind\": \"requirement|study|bugfix|general\", \"progress_note\": \"completed part\", \"remaining_note\": \"remaining work\", \"next_step\": \"concrete next action or tomorrow study topic\"}}],\n\
  \"node_actions\": [{{\n\
    \"target_node_id\": \"existing node id to update or empty for new card\",\n\
    \"name\": \"canonical card title (2-24 chars/words)\",\n\
    \"aliases\": [\"alternative name\"],\n\
    \"search_tags\": [\"bilingual synonym\", \"colloquial query phrase\"],\n\
    \"summary\": \"synthesized overview across all active facts of this card\",\n\
    \"mastery\": 60-95,\n\
    \"status\": \"active|review|solid\",\n\
    \"fact_ops\": [{{\"action\": \"add|merge|supersede\", \"fact_id\": \"existing fact id or empty\", \"kind\": \"fact|decision|pitfall|contract|study_note\", \"text\": \"self-contained atomic point\"}}],\n\
    \"related_nodes\": [{{\"name\": \"existing related card name\", \"relation\": \"depends_on|causes|part_of|related\"}}]\n\
  }}],\n\
  \"timeline\": {{\"user_goal\": \"one-line goal\", \"summary\": \"concise milestone summary\", \"ai_outcome\": \"verified outcome\", \"kind\": \"work|study|milestone\", \"keywords\": [\"keyword\", \"bilingual alias\"]}}\n\
}}",
        agent.label,
        agent.id,
        role_hint,
        if project.is_empty() { "global" } else { project },
        profiles_desc,
        tasks_desc,
        nodes_desc,
        drafts.len(),
        turns_block,
        files_str,
    );

    let fut = async move {
        let msgs = [
            Message::system("You are a precise JSON-only memory consolidation engine."),
            Message::user(prompt),
        ];
        let mut stream = provider.chat_stream(&msgs, &[], &options).await.ok()?;
        let mut raw = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta(delta) => raw.push_str(&delta),
                StreamEvent::Done { .. } | StreamEvent::Error(_) => break,
                _ => {}
            }
        }
        parse_distill_json(&raw)
    };

    tokio::time::timeout(DISTILL_TIMEOUT, fut)
        .await
        .ok()
        .flatten()
}
