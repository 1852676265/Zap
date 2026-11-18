//! 静态会话前缀注入 + `pre_request` 瞬态动态记忆 Hook（不污染对话历史与 `.snapshot`）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use rusqlite::{params, Connection};
use zap_capabilities::reminder::{is_strong_reminder, is_system_reminder, system_reminder};
use zap_kernel::hook::{LifecycleHooks, TurnCtx};
use zap_kernel::message::{Message, Role};
use zap_kernel::tool::Tool;

use super::recall_tool::MemoryRecallTool;
use super::search::{
    build_fts_queries, clean_user_prompt, detect_mentioned_project, is_slash_command,
    parse_temporal_range, query_nodes_fts_conn, query_raw_turns_fts_conn, query_timeline_fts_conn,
    should_inject_l3_verbatim, ProjectScope,
};
use super::store::{
    bind_fresh_session_agent_if_unset_conn, has_any_agents_conn, list_known_projects_conn,
    load_open_tasks_scoped_conn, load_profiles_scoped_conn, load_raw_turn_conn,
    resolve_agent_for_session_conn, with_store,
};
use super::types::{
    normalize_project_key, normalize_project_name, NeuralNode, ProfileFact, RawTurn, TimelineEvent,
    MAX_L3_EVIDENCE_CHARS,
};

const MAX_SESSION_STATIC_STATES: usize = 256;

#[derive(Debug, Clone)]
struct SessionStaticState {
    initial_agent_id: String,
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

static SESSION_STATIC_STATES: std::sync::OnceLock<Mutex<SessionStaticCache>> =
    std::sync::OnceLock::new();

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
    guard.entries.insert(
        sid.to_string(),
        SessionStaticState {
            initial_agent_id: agent_id.to_string(),
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
    Ok((Some(parts.join("\n\n")), frozen_profiles))
}

/// 静态会话前缀块（纯英文标签，命中 Prompt Cache）。
pub fn build_static_system_block(session_id: &str) -> Option<String> {
    with_store(|store| {
        let (blk, _) =
            build_static_system_block_with_ids_conn(&store.conn, session_id, "", "", false)?;
        Ok(blk)
    })
    .ok()
    .flatten()
}

/// 新建 Fresh 会话分配好 `session_id` 后补记静态智能体归属与已冻结档案快照（若 `bind_session_id` 尚未登记）。
pub fn note_fresh_session_static_agent(session_id: &str, working_dir: &Path) {
    let sid = session_id.trim();
    if sid.is_empty() || get_session_static_state(sid).is_some() {
        return;
    }
    let proj_name = normalize_project_name(working_dir);
    let proj_key = normalize_project_key(working_dir);
    let res = with_store(|store| {
        let agent_id = resolve_agent_for_session_conn(&store.conn, sid)
            .map(|a| a.id)
            .unwrap_or_else(|| "none".to_string());
        let frozen_profiles = if agent_id != "none" {
            load_profiles_scoped_conn(&store.conn, &agent_id, &proj_key, &proj_name, 18)?
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
    });
    if let Ok((agent_id, frozen_profiles)) = res {
        record_session_static_state(sid, &agent_id, frozen_profiles);
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
            if self.was_fresh && self.initial_agent_id != "none" {
                let sid_owned = sid.to_string();
                let aid_owned = self.initial_agent_id.clone();
                let _ = with_store(|store| {
                    bind_fresh_session_agent_if_unset_conn(&store.conn, &sid_owned, &aid_owned)
                });
            }
        }
    }
}

/// 挂载智能体静态系统前缀、`memory_recall` 工具（仅在存在可用智能体或已激活时挂载）与 `NeuralMemoryHook`（`pre_request` 瞬态注入）。
pub fn apply_session_static_system_append(
    runtime: &mut zap_coding::CodingRuntimeConfig,
    session_id: Option<&str>,
) -> NeuralMemoryBinding {
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

    let store_res = with_store(|store| {
        let agent = resolve_agent_for_session_conn(&store.conn, &sid);
        let should_mount_tool = agent.is_some() || has_any_agents_conn(&store.conn);
        let (block, frozen_profiles) = build_static_system_block_with_ids_conn(
            &store.conn,
            &sid,
            &current_proj_key,
            &current_proj_name,
            false,
        )?;
        let known_ids: HashSet<String> = if let Some(ref ag) = agent {
            load_profiles_scoped_conn(
                &store.conn,
                &ag.id,
                &current_proj_key,
                &current_proj_name,
                64,
            )?
            .into_iter()
            .map(|p| p.id)
            .collect()
        } else {
            HashSet::new()
        };
        Ok((
            agent.map(|a| a.id),
            should_mount_tool,
            block,
            frozen_profiles,
            known_ids,
        ))
    });

    let Ok((agent_id, should_mount_tool, block_opt, frozen_profiles, known_ids)) = store_res else {
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

/// 瞬态动态记忆 Hook：在每次发往模型前的 `pre_request` 中计算/读取记忆并作为 `synthetic_user` 尾注，
/// 既保持 Prompt Cache 增量前缀安全，又**绝不写入 `Conversation` 与 `.snapshot`**，根除多轮膨胀。
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
        // 提取本轮最后一个真实用户输入
        let Some(latest_user_raw) = messages
            .iter()
            .rev()
            .find(|m| {
                m.role == Role::User
                    && !m.synthetic
                    && !is_system_reminder(&m.text)
                    && !is_strong_reminder(&m.text)
            })
            .map(|m| m.text.clone())
        else {
            return;
        };

        // 同回合第 2..N 轮工具循环直接复用缓存的瞬态提醒，避免重复读 SQLite
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
            build_dynamic_turn_reminder(&sid, &latest_user_raw, &wd)
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

/// 构造动态单轮 `<system-reminder>`（全部使用英文框架指令，严格相关性过滤，支持热切换与本场新学 L1 档案）。
pub fn build_dynamic_turn_reminder(
    session_id: &str,
    user_text: &str,
    working_dir: &Path,
) -> Option<String> {
    let clean_q = clean_user_prompt(user_text);
    if clean_q.is_empty() || is_slash_command(&clean_q) {
        return None;
    }
    match with_store(|store| {
        build_dynamic_turn_reminder_conn(&store.conn, session_id, &clean_q, working_dir)
    }) {
        Ok(res) => res,
        Err(err) => {
            tracing::warn!(error = %err, "neural_memory: failed to build dynamic reminder");
            None
        }
    }
}

pub(crate) fn build_dynamic_turn_reminder_conn(
    conn: &Connection,
    session_id: &str,
    clean_q: &str,
    working_dir: &Path,
) -> Result<Option<String>, String> {
    let proj_name = normalize_project_name(working_dir);
    let proj_key = normalize_project_key(working_dir);
    let static_state = get_session_static_state(session_id);

    let agent_opt = resolve_agent_for_session_conn(conn, session_id);
    let Some(agent) = agent_opt else {
        // 如果会话最初挂载了某个智能体，中途被切换为 `none`，需发一次解除隔离提示
        if let Some(st) = static_state {
            if st.initial_agent_id != "none" {
                return Ok(Some(system_reminder(
                    "[Neural Agent Switched: none]\nNeural memory isolation is now disabled for this session. Do not rely on the initial neural agent persona above.",
                )));
            }
        }
        return Ok(None);
    };

    let known_projects = list_known_projects_conn(conn, &agent.id);
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

    let mut sections: Vec<String> = Vec::new();

    // 1. 若会话中途切换了智能体，发瞬态接管指令（替换开头初始人设）
    let switched = static_state
        .as_ref()
        .is_some_and(|st| st.initial_agent_id != agent.id);
    if switched {
        if let (Some(static_blk), _) =
            build_static_system_block_with_ids_conn(conn, session_id, &proj_key, &proj_name, true)?
        {
            sections.push(static_blk);
        }
    } else {
        // 未切换智能体时：
        // 1) 把本场会话中刚学到、尚未进入静态前缀的新 L1 档案增量注入
        // 2) 把开场冻结后又被删除/否定的档案发一次撤回提示，避免继续生效
        let current_profiles =
            load_profiles_scoped_conn(conn, &agent.id, &proj_key, &proj_name, 64)?;
        let current_ids: HashSet<String> = current_profiles.iter().map(|p| p.id.clone()).collect();
        let known_ids = static_state.as_ref().map(|st| &st.known_profile_ids);
        let newly_learned: Vec<ProfileFact> = current_profiles
            .into_iter()
            .filter(|f| known_ids.is_none_or(|set| !set.contains(&f.id)))
            .take(8)
            .collect();
        if !newly_learned.is_empty() {
            let lines = format_profile_lines(&newly_learned);
            sections.push(format!(
                "[Newly Learned L1 Profile Facts]\n{}",
                lines.join("\n")
            ));
        }
        if let Some(st) = static_state.as_ref() {
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

    // 2. L1 未完成任务（SQL 作用域下推，只查当前/全局或显式点名项目）
    let open_tasks = load_open_tasks_scoped_conn(conn, &agent.id, target_key, target_name, 8)?;
    if !open_tasks.is_empty() {
        let mut lines = Vec::new();
        for t in open_tasks {
            let tag = if t.status == "stale" {
                "stale"
            } else {
                "in_progress"
            };
            let p = if t.project.is_empty() {
                String::new()
            } else {
                format!("[{}] ", t.project)
            };
            let rem = if t.remaining_note.is_empty() {
                String::new()
            } else {
                format!(" | remaining: {}", t.remaining_note)
            };
            lines.push(format!("- ({tag}, id={}) {p}{}{rem}", t.id, t.title));
        }
        sections.push(format!("[L1 Pinned Open Tasks]\n{}", lines.join("\n")));
    }

    // 3. L2 知识节点：严格按 FTS5 相关性命中，不再用无关最近节点凑数
    let (fts_strict, fts_broad) = build_fts_queries(clean_q);
    let mut matched_nodes: Vec<NeuralNode> = Vec::new();
    let mut seen_node_ids: HashSet<String> = HashSet::new();

    for fq in [fts_strict.as_deref(), fts_broad.as_deref()]
        .into_iter()
        .flatten()
    {
        if matched_nodes.len() >= 6 {
            break;
        }
        if let Some(ref mp) = mentioned_project {
            let mp_scope = ProjectScope {
                key: mp.key.as_str(),
                name: mp.name.as_str(),
            };
            for n in query_nodes_fts_conn(conn, &agent.id, fq, Some(mp_scope), 6) {
                if seen_node_ids.insert(n.id.clone()) {
                    matched_nodes.push(n);
                }
            }
        }
        let mut hits = query_nodes_fts_conn(conn, &agent.id, fq, target_scope, 6);
        if hits.is_empty() && target_scope.is_some() && mentioned_project.is_none() {
            // 当前项目未命中时，允许严格 AND 检索跨项目强相关节点
            if fts_strict.as_deref() == Some(fq) {
                hits = query_nodes_fts_conn(conn, &agent.id, fq, None, 4);
            }
        }
        for n in hits {
            if matched_nodes.len() >= 6 {
                break;
            }
            if seen_node_ids.insert(n.id.clone()) {
                matched_nodes.push(n);
            }
        }
    }
    if !matched_nodes.is_empty() {
        let lines: Vec<String> = matched_nodes
            .iter()
            .map(|n| {
                let p = if n.project.is_empty() {
                    String::new()
                } else {
                    format!("[{}] ", n.project)
                };
                format!(
                    "- {p}{} (status={}, mastery={}%): {}",
                    n.name, n.status, n.mastery, n.summary
                )
            })
            .collect();
        sections.push(format!("[L2 Knowledge Nodes]\n{}", lines.join("\n")));
    }

    // 4. L2 时间线索引（日期范围感知 + FTS5 BM25，仅命中时注入）
    let date_range = parse_temporal_range(clean_q);
    let mut timeline_hits: Vec<TimelineEvent> = Vec::new();
    let mut seen_tl_ids: HashSet<String> = HashSet::new();

    if let Some(ref dr) = date_range {
        if let Ok(mut stmt) = conn.prepare(
            "SELECT id, ts, agent_id, project, project_key, working_dir, session_id, turn_id, raw_file,
                    user_goal, summary, ai_outcome, keywords_json, touched_files_json, distilled_by
             FROM timeline
             WHERE agent_id = ?1 AND ts >= ?2 AND ts <= ?3
               AND (?4 = '' OR project_key = ?4 OR (project_key = '' AND LOWER(project) = LOWER(?5)))
             ORDER BY ts DESC LIMIT 8",
        ) {
            if let Ok(rows) = stmt.query_map(
                params![agent.id, dr.start, dr.end, target_key, target_name],
                |row| {
                    let kw_str: String = row.get(12)?;
                    let tf_str: String = row.get(13)?;
                    Ok(TimelineEvent {
                        id: row.get(0)?,
                        ts: row.get(1)?,
                        agent: row.get(2)?,
                        project: row.get(3)?,
                        project_key: row.get(4)?,
                        working_dir: row.get(5)?,
                        session_id: row.get(6)?,
                        turn_id: row.get(7)?,
                        raw_file: row.get(8)?,
                        user_goal: row.get(9)?,
                        summary: row.get(10)?,
                        ai_outcome: row.get(11)?,
                        keywords: serde_json::from_str(&kw_str).unwrap_or_default(),
                        touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
                        distilled_by: row.get(14)?,
                    })
                },
            ) {
                for ev in rows.flatten() {
                    if seen_tl_ids.insert(ev.id.clone()) {
                        timeline_hits.push(ev);
                    }
                }
            }
        }
    }
    for fq in [fts_strict.as_deref(), fts_broad.as_deref()]
        .into_iter()
        .flatten()
    {
        if timeline_hits.len() >= 6 {
            break;
        }
        let hits =
            query_timeline_fts_conn(conn, &agent.id, fq, target_scope, date_range.as_ref(), 6);
        for ev in hits {
            if timeline_hits.len() >= 6 {
                break;
            }
            if seen_tl_ids.insert(ev.id.clone()) {
                timeline_hits.push(ev);
            }
        }
    }
    if !timeline_hits.is_empty() {
        let header = match &date_range {
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

    // 5. L3 原文仅在显式索要原文/堆栈，或精确点名跨项目且 FTS 命中时注入
    if should_inject_l3_verbatim(clean_q) || mentioned_project.is_some() {
        let mut raw_hits: Vec<RawTurn> = Vec::new();
        let mut seen_turns: HashSet<String> = HashSet::new();
        let l3_scope = mentioned_project
            .as_ref()
            .map(|mp| ProjectScope {
                key: mp.key.as_str(),
                name: mp.name.as_str(),
            })
            .or(target_scope);

        for fq in [fts_strict.as_deref(), fts_broad.as_deref()]
            .into_iter()
            .flatten()
        {
            if raw_hits.len() >= 2 {
                break;
            }
            for r in query_raw_turns_fts_conn(conn, &agent.id, fq, l3_scope, date_range.as_ref(), 3)
            {
                if seen_turns.insert(r.turn_id.clone()) {
                    raw_hits.push(r);
                }
                if raw_hits.len() >= 2 {
                    break;
                }
            }
        }
        if raw_hits.is_empty() && should_inject_l3_verbatim(clean_q) {
            for ev in timeline_hits.iter().take(2) {
                if !ev.turn_id.is_empty() && seen_turns.insert(ev.turn_id.clone()) {
                    if let Ok(Some(r)) = load_raw_turn_conn(conn, &ev.turn_id) {
                        raw_hits.push(r);
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

    // 6. 记忆检索工具提示（仅在确实有记忆项目或本次命中数据时提供一行简洁引导）
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
            "[Memory Recall Tool]\nCall `memory_recall` whenever you need past requirements, decisions, tasks, or verbatim transcripts (across projects via `project=\"*\"` or `project=\"<name>\"`).{proj_list}"
        ));
    }

    if sections.is_empty() {
        return Ok(None);
    }
    Ok(Some(system_reminder(&sections.join("\n\n"))))
}
