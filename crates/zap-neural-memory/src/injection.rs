//! 静态会话前缀注入（含极薄大脑目录）+ `pre_request` 瞬态动态记忆 Hook（WAL 读写分离、4 关精准检索、耗时打点）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

use async_trait::async_trait;
use rusqlite::{params, Connection};
use zap_capabilities::reminder::{is_strong_reminder, is_system_reminder, system_reminder};
use zap_kernel::hook::{LifecycleHooks, TurnCtx};
use zap_kernel::message::{Message, Role};
use zap_kernel::tool::Tool;

use super::distill::spawn_boot_draft_recovery_once;
use super::recall_tool::MemoryRecallTool;
use super::retrieval::{
    expand_top_node_neighbors_conn, query_nodes_ranked_conn, query_raw_turns_cold_conn,
    query_tasks_ranked_conn, query_timeline_ranked_conn,
};
use super::search::{
    analyze_query, clean_user_prompt, detect_mentioned_project, is_slash_command, ProjectScope,
};
use super::store::{
    bind_fresh_session_agent_if_unset_conn, build_thin_memory_directory_conn,
    get_thin_directory_cached, has_any_agents_conn, list_known_projects_cached,
    list_known_projects_conn, load_open_tasks_scoped_conn, load_profiles_scoped_conn,
    load_raw_turn_conn, map_timeline_row, record_memory_hits_conn,
    resolve_agent_for_session_cached, resolve_agent_for_session_conn, with_read, with_write,
    KnownProject,
};
use super::team_bridge::{
    format_team_overview_line, list_recent_team_runs_for_project, lookup_team_run_overview,
};
use super::types::{
    normalize_project_key, normalize_project_name, NeuralNode, ProfileFact, RawTurn, TimelineEvent,
    MAX_L3_EVIDENCE_CHARS,
};

const MAX_SESSION_STATIC_STATES: usize = 256;

#[derive(Debug, Clone)]
struct SessionStaticState {
    initial_agent_id: String,
    active_agent_id: String,
    frozen_profile_ids: HashSet<String>,
    known_profile_ids: HashSet<String>,
    frozen_profiles: Vec<(String, String)>,
    last_seq: u64,
}

#[derive(Default)]
struct SessionStaticCache {
    entries: HashMap<String, SessionStaticState>,
    seq: u64,
}

static SESSION_STATIC_STATES: OnceLock<Mutex<SessionStaticCache>> = OnceLock::new();

pub(crate) fn record_session_static_state(
    session_id: &str,
    agent_id: &str,
    frozen_profiles: Vec<(String, String)>,
) {
    let sid = session_id.trim();
    if sid.is_empty() {
        return;
    }
    let frozen_profile_ids: HashSet<String> =
        frozen_profiles.iter().map(|(id, _)| id.clone()).collect();
    let known_profile_ids = frozen_profile_ids.clone();
    let cache = SESSION_STATIC_STATES.get_or_init(|| Mutex::new(SessionStaticCache::default()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard.seq = guard.seq.saturating_add(1);
    let seq = guard.seq;
    let preserved_initial = guard
        .entries
        .get(sid)
        .map(|existing| existing.initial_agent_id.clone())
        .unwrap_or_else(|| agent_id.to_string());
    guard.entries.insert(
        sid.to_string(),
        SessionStaticState {
            initial_agent_id: preserved_initial,
            active_agent_id: agent_id.to_string(),
            frozen_profile_ids,
            known_profile_ids,
            frozen_profiles,
            last_seq: seq,
        },
    );
    if guard.entries.len() > MAX_SESSION_STATIC_STATES {
        let evict_n = guard.entries.len() - MAX_SESSION_STATIC_STATES;
        let mut by_age: Vec<(u64, String)> = guard
            .entries
            .iter()
            .map(|(k, v)| (v.last_seq, k.clone()))
            .collect();
        by_age.sort_by_key(|(s, _)| *s);
        for (_, k) in by_age.into_iter().take(evict_n) {
            guard.entries.remove(&k);
        }
    }
}

fn record_session_known_profile_ids(session_id: &str, known_ids: &HashSet<String>) {
    let sid = session_id.trim();
    if sid.is_empty() || known_ids.is_empty() {
        return;
    }
    let cache = SESSION_STATIC_STATES.get_or_init(|| Mutex::new(SessionStaticCache::default()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = guard.entries.get_mut(sid) {
        entry.known_profile_ids.extend(known_ids.iter().cloned());
    }
}

fn get_session_static_state(session_id: &str) -> Option<SessionStaticState> {
    let sid = session_id.trim();
    if sid.is_empty() {
        return None;
    }
    let cache = SESSION_STATIC_STATES.get_or_init(|| Mutex::new(SessionStaticCache::default()));
    let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard.entries.get(sid).cloned()
}

/// 当会话在运行期切换了智能体（含切换回 `none`）时，刷新该会话的当前生效智能体与 L1 档案基线快照。
fn sync_session_switched_agent_state(
    session_id: &str,
    active_agent_id: &str,
    frozen_profiles: Vec<(String, String)>,
    all_scoped_profile_ids: HashSet<String>,
) {
    let sid = session_id.trim();
    if sid.is_empty() {
        return;
    }
    let frozen_profile_ids: HashSet<String> =
        frozen_profiles.iter().map(|(id, _)| id.clone()).collect();
    let mut known_profile_ids = frozen_profile_ids.clone();
    known_profile_ids.extend(all_scoped_profile_ids);
    let cache = SESSION_STATIC_STATES.get_or_init(|| Mutex::new(SessionStaticCache::default()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    guard.seq = guard.seq.saturating_add(1);
    let seq = guard.seq;
    if let Some(entry) = guard.entries.get_mut(sid) {
        entry.active_agent_id = active_agent_id.to_string();
        entry.frozen_profile_ids = frozen_profile_ids;
        entry.known_profile_ids = known_profile_ids;
        entry.frozen_profiles = frozen_profiles;
        entry.last_seq = seq;
    } else {
        guard.entries.insert(
            sid.to_string(),
            SessionStaticState {
                initial_agent_id: active_agent_id.to_string(),
                active_agent_id: active_agent_id.to_string(),
                frozen_profile_ids,
                known_profile_ids,
                frozen_profiles,
                last_seq: seq,
            },
        );
    }
}

fn format_profile_line(f: &ProfileFact) -> String {
    let pin = if f.pinned { "[pinned] " } else { "" };
    let proj = if f.project.is_empty() {
        String::new()
    } else {
        format!("[{}] ", f.project)
    };
    format!("- {pin}{proj}{}", f.text)
}

fn format_profile_lines(facts: &[ProfileFact]) -> Vec<String> {
    facts.iter().map(format_profile_line).collect()
}

pub(crate) fn build_static_system_block_with_ids_conn(
    conn: &Connection,
    session_id: &str,
    current_project_key: &str,
    current_project_name: &str,
    is_switched_override: bool,
) -> Result<(Option<String>, Vec<(String, String)>), String> {
    let Some(agent) = resolve_agent_for_session_conn(conn, session_id) else {
        return Ok((None, Vec::new()));
    };
    let facts = load_profiles_scoped_conn(
        conn,
        &agent.id,
        current_project_key,
        current_project_name,
        18,
    )?;
    let frozen_profiles: Vec<(String, String)> = facts
        .iter()
        .map(|f| (f.id.clone(), format_profile_line(f)))
        .collect();

    let mut parts = Vec::new();
    if is_switched_override {
        parts.push(format!(
            "[Neural Agent Switched: {} ({})]\nThis supersedes the initial neural agent persona and memory scope above for this turn and all subsequent turns. Cite only this active agent's memory.",
            agent.label, agent.id
        ));
    } else {
        parts.push(format!(
            "[Initial Neural Agent: {} ({})]\nSession memory isolation is active for this agent; cite only this agent's memory and do not mix other agent contexts unless superseded by a later [Neural Agent Switched] directive.",
            agent.label, agent.id
        ));
    }
    if !agent.system_prompt.trim().is_empty() {
        parts.push(format!(
            "[Agent System Prompt]\n{}",
            agent.system_prompt.trim()
        ));
    }
    if !facts.is_empty() {
        let lines = format_profile_lines(&facts);
        parts.push(format!("[L1 Core Agent Profile]\n{}", lines.join("\n")));
    }
    // 第 1 关：极薄大脑目录（仅在开场静态前缀挂载，0ms 告诉模型有哪些活跃需求/学习计划和知识卡片索引）
    if !is_switched_override {
        let thin_dir = build_thin_memory_directory_conn(
            conn,
            &agent.id,
            current_project_key,
            current_project_name,
        )?;
        if !thin_dir.trim().is_empty() {
            parts.push(thin_dir);
        }
    }
    Ok((Some(parts.join("\n\n")), frozen_profiles))
}

/// 静态会话前缀块（纯英文标签，命中 Prompt Cache，走独立只读连接）。
pub fn build_static_system_block(session_id: &str) -> Option<String> {
    with_read(|conn, _| {
        let (blk, _) = build_static_system_block_with_ids_conn(conn, session_id, "", "", false)?;
        Ok(blk)
    })
    .ok()
    .flatten()
}

/// 新建 Fresh 会话分配好 `session_id` 后补记静态智能体归属与已冻结档案快照（若 `bind_session_id` 尚未登记）。
pub async fn note_fresh_session_static_agent(session_id: &str, working_dir: &Path) {
    let sid = session_id.trim();
    if sid.is_empty() || get_session_static_state(sid).is_some() {
        return;
    }
    let sid_owned = sid.to_string();
    let proj_name = normalize_project_name(working_dir);
    let proj_key = normalize_project_key(working_dir);
    let sid_for_db = sid_owned.clone();
    let res = tokio::task::spawn_blocking(move || {
        with_read(|conn, _| {
            let agent_id = resolve_agent_for_session_conn(conn, &sid_for_db)
                .map(|a| a.id)
                .unwrap_or_else(|| "none".to_string());
            let frozen_profiles = if agent_id != "none" {
                load_profiles_scoped_conn(conn, &agent_id, &proj_key, &proj_name, 18)?
                    .into_iter()
                    .map(|f| {
                        let line = format_profile_line(&f);
                        (f.id, line)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Ok((agent_id, frozen_profiles))
        })
    })
    .await;
    if let Ok(Ok((agent_id, frozen_profiles))) = res {
        record_session_static_state(&sid_owned, &agent_id, frozen_profiles);
    }
}

/// 由 `apply_session_static_system_append` 返回给调用方的绑定句柄，用于在 Fresh 会话创建后绑定 `session_id`。
#[derive(Clone)]
pub struct NeuralMemoryBinding {
    recall_tool: Option<Arc<MemoryRecallTool>>,
    hook: Arc<NeuralMemoryHook>,
    was_fresh: bool,
    initial_agent_id: String,
    frozen_profiles: Vec<(String, String)>,
    initial_known_profile_ids: HashSet<String>,
}

impl NeuralMemoryBinding {
    pub fn bind_session_id(&self, session_id: &str) {
        let sid = session_id.trim();
        if let Some(tool) = &self.recall_tool {
            tool.bind_session_id(sid);
        }
        self.hook.bind_session_id(sid);
        if !sid.is_empty() {
            record_session_static_state(sid, &self.initial_agent_id, self.frozen_profiles.clone());
            record_session_known_profile_ids(sid, &self.initial_known_profile_ids);
            if self.was_fresh {
                let sid_owned = sid.to_string();
                let aid_owned = self.initial_agent_id.clone();
                if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    rt.spawn_blocking(move || {
                        let _ = with_write(|store| {
                            bind_fresh_session_agent_if_unset_conn(
                                &store.conn,
                                &sid_owned,
                                &aid_owned,
                            )
                        });
                    });
                } else {
                    let _ = with_write(|store| {
                        bind_fresh_session_agent_if_unset_conn(&store.conn, &sid_owned, &aid_owned)
                    });
                }
            }
        }
    }
}

/// 挂载智能体静态系统前缀、`memory_recall` 工具与 `NeuralMemoryHook`（`pre_request` 瞬态注入）。
pub async fn apply_session_static_system_append(
    runtime: &mut zap_coding::CodingRuntimeConfig,
    session_id: Option<&str>,
) -> NeuralMemoryBinding {
    spawn_boot_draft_recovery_once();
    let sid_opt = session_id
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let sid = sid_opt.clone().unwrap_or_default();
    let current_proj_name = normalize_project_name(&runtime.working_dir);
    let current_proj_key = normalize_project_key(&runtime.working_dir);

    let hook = Arc::new(NeuralMemoryHook::new(
        sid_opt.clone(),
        runtime.working_dir.clone(),
    ));
    runtime
        .extra_hooks
        .push(Arc::clone(&hook) as Arc<dyn LifecycleHooks>);

    let sid_for_db = sid.clone();
    let pkey_dir = current_proj_key.clone();
    let pname_dir = current_proj_name.clone();
    let store_res = tokio::task::spawn_blocking(move || {
        let agent = resolve_agent_for_session_cached(&sid_for_db)?;
        let (should_mount_tool, mut block, frozen_profiles, known_ids) = with_read(|conn, _| {
            let should_mount_tool = agent.is_some() || has_any_agents_conn(conn);
            let (block, frozen_profiles) = build_static_system_block_with_ids_conn(
                conn,
                &sid_for_db,
                &current_proj_key,
                &current_proj_name,
                false,
            )?;
            let known_ids: HashSet<String> = if let Some(ref ag) = agent {
                load_profiles_scoped_conn(conn, &ag.id, &current_proj_key, &current_proj_name, 64)?
                    .into_iter()
                    .map(|p| p.id)
                    .collect()
            } else {
                HashSet::new()
            };
            Ok((should_mount_tool, block, frozen_profiles, known_ids))
        })?;
        // 预热极薄大脑目录缓存（写时失效），后续新会话 0ms 复用
        if let Some(ref ag) = agent {
            if let Ok(dir) = get_thin_directory_cached(&ag.id, &pkey_dir, &pname_dir) {
                if !dir.trim().is_empty() && block.as_deref().is_some_and(|b| !b.contains(&dir)) {
                    if let Some(ref mut b) = block {
                        b.push_str("\n\n");
                        b.push_str(&dir);
                    }
                }
            }
        }
        let resolved_aid = agent.as_ref().map(|a| a.id.as_str()).unwrap_or("none");
        if !sid_for_db.is_empty() {
            let _ = with_write(|store| {
                bind_fresh_session_agent_if_unset_conn(&store.conn, &sid_for_db, resolved_aid)
            });
        }
        Ok::<_, String>((
            agent.map(|a| a.id),
            should_mount_tool,
            block,
            frozen_profiles,
            known_ids,
        ))
    })
    .await;

    let Ok(Ok((agent_id, should_mount_tool, block_opt, frozen_profiles, known_ids))) = store_res
    else {
        return NeuralMemoryBinding {
            recall_tool: None,
            hook,
            was_fresh: sid.is_empty(),
            initial_agent_id: "none".to_string(),
            frozen_profiles: Vec::new(),
            initial_known_profile_ids: HashSet::new(),
        };
    };

    let recall_tool = if should_mount_tool {
        let tool = Arc::new(MemoryRecallTool::new(sid_opt));
        if !runtime.extra_tools.iter().any(|t| t.name() == tool.name()) {
            runtime.extra_tools.push(Arc::clone(&tool) as Arc<dyn Tool>);
        }
        Some(tool)
    } else {
        None
    };

    let resolved_agent_id = agent_id.unwrap_or_else(|| "none".to_string());
    if !sid.is_empty() {
        record_session_static_state(&sid, &resolved_agent_id, frozen_profiles.clone());
        record_session_known_profile_ids(&sid, &known_ids);
    }

    if let Some(block) = block_opt {
        match runtime.extra_system_append.as_mut() {
            Some(existing) if !existing.trim().is_empty() => {
                existing.push_str("\n\n");
                existing.push_str(&block);
            }
            _ => {
                runtime.extra_system_append = Some(block);
            }
        }
    }

    NeuralMemoryBinding {
        recall_tool,
        hook,
        was_fresh: sid.is_empty(),
        initial_agent_id: resolved_agent_id,
        frozen_profiles,
        initial_known_profile_ids: known_ids,
    }
}

/// 瞬态动态记忆 Hook：在每次发往模型前的 `pre_request` 中走只读连接极速计算相关记忆并作为 `synthetic_user` 尾注。
pub struct NeuralMemoryHook {
    session_id: Arc<RwLock<Option<String>>>,
    working_dir: PathBuf,
    turn_cache: Mutex<Option<(u64, u64, Option<String>)>>,
}

impl NeuralMemoryHook {
    pub fn new(session_id: Option<String>, working_dir: PathBuf) -> Self {
        Self {
            session_id: Arc::new(RwLock::new(
                session_id
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            )),
            working_dir,
            turn_cache: Mutex::new(None),
        }
    }

    pub fn bind_session_id(&self, session_id: &str) {
        let trimmed = session_id.trim();
        if trimmed.is_empty() {
            return;
        }
        if let Ok(mut guard) = self.session_id.write() {
            *guard = Some(trimmed.to_string());
        }
    }

    fn resolved_session_id(&self, ctx: &TurnCtx) -> String {
        if let Some(sid) = ctx.session_id.as_deref() {
            let trimmed = sid.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        self.session_id
            .read()
            .ok()
            .and_then(|g| g.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl LifecycleHooks for NeuralMemoryHook {
    async fn pre_request(&self, messages: &mut Vec<Message>, ctx: &TurnCtx) {
        // 提取最近 3 条真实用户消息：最新一条为主问题，前 1~2 条供短句追问借用主语主题词
        let real_user_msgs: Vec<String> = messages
            .iter()
            .rev()
            .filter(|m| {
                m.role == Role::User
                    && !m.synthetic
                    && !is_system_reminder(&m.text)
                    && !is_strong_reminder(&m.text)
            })
            .take(3)
            .map(|m| m.text.clone())
            .collect();
        let Some(latest_user_raw) = real_user_msgs.first().cloned() else {
            return;
        };
        let prior_user_turns: Vec<String> = real_user_msgs
            .iter()
            .skip(1)
            .rev()
            .map(|s| clean_user_prompt(s))
            .filter(|s| !s.is_empty())
            .collect();

        // 同回合第 2..N 轮工具循环直接复用缓存的瞬态提醒，0 次 SQL 查询
        if ctx.turn_id > 0 {
            let cached = self
                .turn_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .and_then(|(tid, epoch, val)| {
                    (*tid == ctx.turn_id && *epoch == ctx.cache_epoch).then(|| val.clone())
                });
            if let Some(reminder_opt) = cached {
                if let Some(reminder) = reminder_opt {
                    messages.push(Message::synthetic_user(reminder));
                }
                return;
            }
        }

        let sid = self.resolved_session_id(ctx);
        let wd = self.working_dir.clone();
        let reminder_opt = tokio::task::spawn_blocking(move || {
            build_dynamic_turn_reminder_with_context(&sid, &latest_user_raw, &prior_user_turns, &wd)
        })
        .await
        .ok()
        .flatten();

        if ctx.turn_id > 0 {
            let mut guard = self.turn_cache.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some((ctx.turn_id, ctx.cache_epoch, reminder_opt.clone()));
        }

        if let Some(reminder) = reminder_opt {
            messages.push(Message::synthetic_user(reminder));
        }
    }
}

fn record_hits_async(node_ids: Vec<String>, profile_ids: Vec<String>) {
    if (node_ids.is_empty() && profile_ids.is_empty()) || cfg!(test) {
        return;
    }
    if let Ok(rt) = tokio::runtime::Handle::try_current() {
        rt.spawn_blocking(move || {
            let _ = with_write(|store| {
                record_memory_hits_conn(&store.conn, &node_ids, &profile_ids);
                Ok(())
            });
        });
    }
}

/// 构造动态单轮 `<system-reminder>`（向后兼容旧签名）。
pub fn build_dynamic_turn_reminder(
    session_id: &str,
    user_text: &str,
    working_dir: &Path,
) -> Option<String> {
    build_dynamic_turn_reminder_with_context(session_id, user_text, &[], working_dir)
}

pub fn build_dynamic_turn_reminder_with_context(
    session_id: &str,
    user_text: &str,
    prior_user_turns: &[String],
    working_dir: &Path,
) -> Option<String> {
    let clean_q = clean_user_prompt(user_text);
    if clean_q.is_empty() || is_slash_command(&clean_q) {
        return None;
    }
    let started = Instant::now();

    // 生产热路径：先查进程级热缓存；通用模式下直接 0 SQL 短路返回
    let cached_agent = match resolve_agent_for_session_cached(session_id) {
        Ok(ag) => ag,
        Err(err) => {
            tracing::warn!(error = %err, "neural_memory: failed to resolve cached agent");
            return None;
        }
    };
    let Some(agent) = cached_agent else {
        if let Some(st) = get_session_static_state(session_id) {
            if st.initial_agent_id != "none" || st.active_agent_id != "none" {
                sync_session_switched_agent_state(session_id, "none", Vec::new(), HashSet::new());
                return Some(system_reminder(
                    "[Neural Agent Switched: none]\nNeural memory isolation is now disabled for this session. Do not rely on the initial or previous neural agent persona above.",
                ));
            }
        }
        return None;
    };
    let known_projects = list_known_projects_cached(&agent.id).unwrap_or_default();

    let res = with_read(|conn, _| {
        build_dynamic_turn_reminder_inner(
            conn,
            session_id,
            &clean_q,
            prior_user_turns,
            working_dir,
            Some((agent, known_projects)),
        )
    });
    match res {
        Ok((reminder_opt, hit_nodes, hit_profiles)) => {
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            tracing::debug!(
                session_id = session_id,
                elapsed_ms = format_args!("{elapsed_ms:.2}"),
                injected = reminder_opt.is_some(),
                hit_nodes = hit_nodes.len(),
                "neural_memory: pre_request retrieval completed"
            );
            record_hits_async(hit_nodes, hit_profiles);
            reminder_opt
        }
        Err(err) => {
            tracing::warn!(error = %err, "neural_memory: failed to build dynamic reminder");
            None
        }
    }
}

#[cfg(test)]
pub(crate) fn build_dynamic_turn_reminder_conn(
    conn: &Connection,
    session_id: &str,
    clean_q: &str,
    working_dir: &Path,
) -> Result<Option<String>, String> {
    let (rem, _, _) =
        build_dynamic_turn_reminder_inner(conn, session_id, clean_q, &[], working_dir, None)?;
    Ok(rem)
}

#[cfg(test)]
pub(crate) fn build_dynamic_turn_reminder_with_context_conn(
    conn: &Connection,
    session_id: &str,
    clean_q: &str,
    prior_user_turns: &[String],
    working_dir: &Path,
) -> Result<Option<String>, String> {
    let (rem, _, _) = build_dynamic_turn_reminder_inner(
        conn,
        session_id,
        clean_q,
        prior_user_turns,
        working_dir,
        None,
    )?;
    Ok(rem)
}

fn format_node_card_block(
    n: &NeuralNode,
    include_superseded: bool,
    relation_tag: Option<&str>,
) -> String {
    let p = if n.project.is_empty() {
        String::new()
    } else {
        format!("[{}] ", n.project)
    };
    let rel = relation_tag
        .map(|r| format!("[linked:{r}] "))
        .unwrap_or_default();
    let mut header = format!(
        "- {rel}{p}{} (status={}, mastery={}%): {}",
        n.name, n.status, n.mastery, n.summary
    );
    let extra_facts: Vec<String> = n
        .facts
        .iter()
        .filter(|f| include_superseded || f.is_active())
        .filter(|f| f.text.trim() != n.summary.trim())
        .take(3)
        .map(|f| {
            let st = if f.is_active() { "" } else { "[superseded] " };
            format!("  * {st}[{}] {}", f.kind, f.text)
        })
        .collect();
    if !extra_facts.is_empty() {
        header.push('\n');
        header.push_str(&extra_facts.join("\n"));
    }
    header
}

fn build_dynamic_turn_reminder_inner(
    conn: &Connection,
    session_id: &str,
    clean_q: &str,
    prior_user_turns: &[String],
    working_dir: &Path,
    preloaded: Option<(super::types::NeuralAgent, Vec<KnownProject>)>,
) -> Result<(Option<String>, Vec<String>, Vec<String>), String> {
    let proj_name = normalize_project_name(working_dir);
    let proj_key = normalize_project_key(working_dir);
    let static_state = get_session_static_state(session_id);

    let (agent, known_projects) = match preloaded {
        Some((ag, projs)) => (ag, projs),
        None => {
            let agent_opt = resolve_agent_for_session_conn(conn, session_id);
            let Some(ag) = agent_opt else {
                if let Some(st) = static_state {
                    if st.initial_agent_id != "none" || st.active_agent_id != "none" {
                        sync_session_switched_agent_state(
                            session_id,
                            "none",
                            Vec::new(),
                            HashSet::new(),
                        );
                        return Ok((
                            Some(system_reminder(
                                "[Neural Agent Switched: none]\nNeural memory isolation is now disabled for this session. Do not rely on the initial or previous neural agent persona above.",
                            )),
                            Vec::new(),
                            Vec::new(),
                        ));
                    }
                }
                return Ok((None, Vec::new(), Vec::new()));
            };
            let projs = list_known_projects_conn(conn, &ag.id);
            (ag, projs)
        }
    };
    let mentioned_project =
        detect_mentioned_project(clean_q, &known_projects, &proj_name, &proj_key);

    let (target_key, target_name) = match &mentioned_project {
        Some(mp) => (mp.key.as_str(), mp.name.as_str()),
        None => (proj_key.as_str(), proj_name.as_str()),
    };
    let target_scope = if target_key.is_empty() && target_name.is_empty() {
        None
    } else {
        Some(ProjectScope {
            key: target_key,
            name: target_name,
        })
    };

    let plan = analyze_query(clean_q, prior_user_turns);
    let mut sections: Vec<String> = Vec::new();
    let mut hit_node_ids: Vec<String> = Vec::new();
    let mut hit_profile_ids: Vec<String> = Vec::new();

    // 1. 支持单会话内任意多次切换智能体 + 增量同步本场新学/撤回的 L1 档案
    let differs_from_initial = static_state
        .as_ref()
        .is_some_and(|st| st.initial_agent_id != agent.id);
    let agent_just_changed = static_state
        .as_ref()
        .is_none_or(|st| st.active_agent_id != agent.id);

    let current_profiles = load_profiles_scoped_conn(conn, &agent.id, &proj_key, &proj_name, 64)?;
    let current_ids: HashSet<String> = current_profiles.iter().map(|p| p.id.clone()).collect();

    if differs_from_initial {
        let (static_blk_opt, switched_frozen) =
            build_static_system_block_with_ids_conn(conn, session_id, &proj_key, &proj_name, true)?;
        if let Some(static_blk) = static_blk_opt {
            sections.push(static_blk);
        }
        if agent_just_changed {
            sync_session_switched_agent_state(
                session_id,
                &agent.id,
                switched_frozen,
                current_ids.clone(),
            );
        }
    } else if agent_just_changed {
        let (_, switched_frozen) = build_static_system_block_with_ids_conn(
            conn, session_id, &proj_key, &proj_name, false,
        )?;
        sync_session_switched_agent_state(
            session_id,
            &agent.id,
            switched_frozen,
            current_ids.clone(),
        );
    }

    if !agent_just_changed {
        let active_state = get_session_static_state(session_id);
        let known_ids = active_state.as_ref().map(|st| &st.known_profile_ids);
        let newly_learned: Vec<ProfileFact> = current_profiles
            .into_iter()
            .filter(|f| known_ids.is_none_or(|set| !set.contains(&f.id)))
            .take(8)
            .collect();
        if !newly_learned.is_empty() {
            for f in &newly_learned {
                hit_profile_ids.push(f.id.clone());
            }
            let lines = format_profile_lines(&newly_learned);
            sections.push(format!(
                "[Newly Learned L1 Profile Facts]\n{}",
                lines.join("\n")
            ));
        }
        if let Some(st) = active_state.as_ref() {
            let retracted: Vec<String> = st
                .frozen_profiles
                .iter()
                .filter(|(id, _)| !current_ids.contains(id))
                .map(|(_, line)| line.clone())
                .collect();
            if !retracted.is_empty() {
                sections.push(format!(
                    "[Retracted L1 Profile Facts (no longer valid, do not follow)]\n{}",
                    retracted.join("\n")
                ));
            }
        }
    }

    // 2. 抽屉 2（需求与学习进度本 + Team 跨端状态透视）
    let mut displayed_tasks =
        load_open_tasks_scoped_conn(conn, &agent.id, target_key, target_name, 8)?;
    if plan.wants_progress {
        let mut seen_tids: HashSet<String> = displayed_tasks.iter().map(|t| t.id.clone()).collect();
        for rt in query_tasks_ranked_conn(conn, &agent.id, &plan, target_scope, 6) {
            if seen_tids.insert(rt.task.id.clone()) {
                displayed_tasks.push(rt.task);
            }
        }
    }
    let mut team_run_ids_to_check: Vec<String> = Vec::new();
    if !displayed_tasks.is_empty() {
        let mut lines = Vec::new();
        for t in &displayed_tasks {
            let p = if t.project.is_empty() {
                String::new()
            } else {
                format!("[{}] ", t.project)
            };
            let rem = if !t.next_step.trim().is_empty() {
                format!(" | next: {}", t.next_step.trim())
            } else if !t.remaining_note.trim().is_empty() {
                format!(" | remaining: {}", t.remaining_note.trim())
            } else if !t.progress_note.trim().is_empty() {
                format!(" | progress: {}", t.progress_note.trim())
            } else {
                String::new()
            };
            if !t.team_run_id.trim().is_empty() && !team_run_ids_to_check.contains(&t.team_run_id) {
                team_run_ids_to_check.push(t.team_run_id.clone());
            }
            lines.push(format!(
                "- ({}/{}, id={}) {p}{}{rem}",
                t.item_kind, t.status, t.id, t.title
            ));
        }
        sections.push(format!("[L1 Pinned Open Tasks]\n{}", lines.join("\n")));
    }

    let mut team_overview_lines: Vec<String> = Vec::new();
    let mut seen_runs: HashSet<String> = HashSet::new();
    for rid in &team_run_ids_to_check {
        if seen_runs.insert(rid.clone()) {
            if let Some(ov) = lookup_team_run_overview(rid) {
                team_overview_lines.push(format_team_overview_line(&ov));
            }
        }
    }
    if plan.wants_progress && team_overview_lines.len() < 3 {
        for ov in list_recent_team_runs_for_project(working_dir, 3) {
            if seen_runs.insert(ov.run_id.clone()) {
                team_overview_lines.push(format_team_overview_line(&ov));
            }
        }
    }
    if !team_overview_lines.is_empty() {
        sections.push(format!(
            "[Linked Agent Teams Cross-Role Status]\n{}",
            team_overview_lines.join("\n")
        ));
    }

    // 3. 抽屉 3（知识与经验活卡片）：精确路牌 + 分列加权 BM25 + 多因子重排 + 及格线宁缺毋滥 + 1-hop 邻居
    let mut ranked_nodes = query_nodes_ranked_conn(conn, &agent.id, &plan, target_scope, 3);
    if ranked_nodes.is_empty() && target_scope.is_some() && mentioned_project.is_none() {
        ranked_nodes = query_nodes_ranked_conn(conn, &agent.id, &plan, None, 2)
            .into_iter()
            .filter(|rn| rn.anchor_hits > 0 || rn.score >= 28.0)
            .collect();
    }
    if !ranked_nodes.is_empty() {
        let mut lines: Vec<String> = Vec::new();
        for rn in &ranked_nodes {
            hit_node_ids.push(rn.node.id.clone());
            lines.push(format_node_card_block(
                &rn.node,
                plan.wants_superseded,
                None,
            ));
        }
        for (neighbor, rel) in expand_top_node_neighbors_conn(conn, &agent.id, &ranked_nodes, 1) {
            lines.push(format_node_card_block(
                &neighbor,
                plan.wants_superseded,
                Some(&rel),
            ));
        }
        sections.push(format!("[L2 Knowledge Nodes]\n{}", lines.join("\n")));
    }

    // 4. 抽屉 4（每日精华大事记）：日期范围直达 + 综合加权检索及格线
    let mut timeline_hits: Vec<TimelineEvent> = Vec::new();
    let mut seen_tl_ids: HashSet<String> = HashSet::new();
    if let Some(ref dr) = plan.date_range {
        if let Ok(mut stmt) = conn.prepare_cached(
            "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                    user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by,
                    kind, team_run_id, node_ids_json
             FROM timeline
             WHERE agent_id = ?1 AND ts >= ?2 AND ts <= ?3
               AND (?4 = '' OR project_key = ?4 OR (project_key = '' AND LOWER(project) = LOWER(?5)) OR (project_key = '' AND project = ''))
             ORDER BY ts DESC LIMIT 6",
        ) {
            if let Ok(rows) = stmt.query_map(
                params![agent.id, dr.start, dr.end, target_key, target_name],
                map_timeline_row,
            ) {
                for ev in rows.flatten() {
                    if seen_tl_ids.insert(ev.id.clone()) {
                        timeline_hits.push(ev);
                    }
                }
            }
        }
    }
    if timeline_hits.len() < 3 {
        for rt in query_timeline_ranked_conn(
            conn,
            &agent.id,
            &plan,
            target_scope,
            plan.date_range.as_ref(),
            3,
        ) {
            if timeline_hits.len() >= 3 {
                break;
            }
            if seen_tl_ids.insert(rt.event.id.clone()) {
                timeline_hits.push(rt.event);
            }
        }
    }
    if !timeline_hits.is_empty() {
        let header = match &plan.date_range {
            Some(dr) => format!("[L2 Period Index: {}]", dr.label),
            None => "[L2 Timeline Index]".to_string(),
        };
        let lines: Vec<String> = timeline_hits
            .iter()
            .map(|ev| {
                let p = if ev.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", ev.project)
                };
                let out = if ev.ai_outcome.is_empty() {
                    String::new()
                } else {
                    format!(" -> {}", ev.ai_outcome)
                };
                format!("- [{}] {p}{}{out}", ev.ts, ev.summary)
            })
            .collect();
        sections.push(format!("{header}\n{}", lines.join("\n")));
    }

    // 5. 抽屉 5（冷仓库原话）：仅显式索要原文/堆栈或跨项目点名时按需检索（零 FTS 膨胀）
    if plan.wants_verbatim || mentioned_project.is_some() {
        let mut raw_hits: Vec<RawTurn> = Vec::new();
        let mut seen_turns: HashSet<String> = HashSet::new();
        let l3_scope = mentioned_project
            .as_ref()
            .map(|mp| ProjectScope {
                key: mp.key.as_str(),
                name: mp.name.as_str(),
            })
            .or(target_scope);

        for ev in timeline_hits.iter().take(2) {
            if !ev.turn_id.is_empty() && seen_turns.insert(ev.turn_id.clone()) {
                if let Ok(Some(r)) = load_raw_turn_conn(conn, &ev.turn_id) {
                    raw_hits.push(r);
                }
            }
        }
        if raw_hits.len() < 2 {
            let mut terms = plan.anchors.clone();
            terms.extend(plan.core_terms.iter().take(3).cloned());
            for r in query_raw_turns_cold_conn(
                conn,
                &agent.id,
                &terms,
                l3_scope,
                plan.date_range.as_ref(),
                2,
            ) {
                if seen_turns.insert(r.turn_id.clone()) {
                    raw_hits.push(r);
                    if raw_hits.len() >= 2 {
                        break;
                    }
                }
            }
        }
        if !raw_hits.is_empty() {
            let mut total_chars = 0usize;
            let mut blocks = Vec::new();
            for r in raw_hits {
                let u: String = r.user_text.chars().take(700).collect();
                let a: String = r.ai_text.chars().take(1200).collect();
                let blk = format!(
                    "--- [{}] [{}] ---\nUser: {u}\nAssistant: {a}",
                    r.ts, r.project
                );
                total_chars += blk.chars().count();
                if total_chars > MAX_L3_EVIDENCE_CHARS && !blocks.is_empty() {
                    break;
                }
                blocks.push(blk);
            }
            if !blocks.is_empty() {
                sections.push(format!(
                    "[L3 Verbatim Raw Evidence]\n{}",
                    blocks.join("\n\n")
                ));
            }
        }
    }

    // 6. 工具提示
    if !known_projects.is_empty() && (!sections.is_empty() || mentioned_project.is_some()) {
        let mut names: Vec<String> = known_projects
            .into_iter()
            .filter(|kp| !kp.name.is_empty())
            .map(|kp| kp.name)
            .collect();
        names.sort();
        names.dedup();
        let proj_list = if names.is_empty() {
            String::new()
        } else {
            format!(" Known projects in memory: {}.", names.join(", "))
        };
        sections.push(format!(
            "[Memory Recall Tool]\nCall `memory_recall` whenever you need full card facts, superseded history, cross-project records (`project=\"*\"`), or verbatim transcripts (`include_raw=true`).{proj_list}"
        ));
    }

    if sections.is_empty() {
        return Ok((None, hit_node_ids, hit_profile_ids));
    }
    Ok((
        Some(system_reminder(&sections.join("\n\n"))),
        hit_node_ids,
        hit_profile_ids,
    ))
}
