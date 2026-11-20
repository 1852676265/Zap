//! 写入期「先查后合」事务引擎（Reconcile）：
//! - 抽屉 1（profiles）：支持新增、融合更新、作废替代（`superseded_by` + `invalid_at`）、显式移除（保护置顶）
//! - 抽屉 2（tasks）：支持工作需求与学习计划（`item_kind`、`next_step`、模糊标题合并与 14 天 stale 标记）
//! - 抽屉 3（nodes）：按 ID / 名称 / 别名精准归并，内部维护多条独立 `NodeFact`（`add` / `merge` / `supersede`），
//!   绝不盲覆盖旧总结，同时更新 `aliases`、`search_tags` 与有类型 `links`
//! - 抽屉 4（timeline）：每批次只落 1 条高纯度大事记并关联卡片 ID
//! - 抽屉 5（raw_turns）：同事务无损归档（无 FTS 膨胀）

use std::collections::HashSet;

use chrono::NaiveDate;
use rusqlite::{params, Connection, TransactionBehavior};
use serde::Deserialize;

use super::search::{escape_like_pattern, is_noise_node_name, normalize_node_key};
use super::store::{
    delete_node_with_indexes_conn, insert_timeline_conn, load_nodes_conn, load_profiles_conn,
    load_tasks_conn, upsert_link_conn, upsert_node_conn, upsert_profile_conn, upsert_raw_turn_conn,
    upsert_task_conn,
};
use super::types::{
    default_fact_kind, default_journal_kind, default_link_relation, default_mastery,
    default_node_status, default_profile_kind, default_task_item_kind, default_windows, new_id,
    normalize_project_key_str, now_ts_minute, NeuralNode, NeuralTask, NodeEvent, NodeFact,
    ProfileFact, RawTurn, TimelineEvent,
};

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillOutput {
    #[serde(default)]
    pub(crate) should_record: bool,
    /// 旧字段兼容：简单新增/置顶规矩
    #[serde(default)]
    pub(crate) profile_updates: Vec<DistillProfileInput>,
    /// 旧字段兼容：按关键词删除非置顶规矩
    #[serde(default)]
    pub(crate) profile_removes: Vec<String>,
    /// v3 结构化规矩操作：支持 `add` | `update` | `supersede` | `remove`
    #[serde(default)]
    pub(crate) profile_actions: Vec<DistillProfileAction>,
    #[serde(default)]
    pub(crate) task_actions: Vec<DistillTaskAction>,
    #[serde(default)]
    pub(crate) node_actions: Vec<DistillNodeAction>,
    #[serde(default)]
    pub(crate) timeline: Option<DistillTimeline>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum DistillProfileInput {
    Obj(DistillProfileObj),
    Str(String),
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillProfileObj {
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) pin: bool,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillProfileAction {
    /// `"add"` | `"update"` | `"supersede"` | `"remove"`
    #[serde(default)]
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) target_id: String,
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) pin: bool,
    #[serde(default = "default_profile_kind")]
    pub(crate) kind: String,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillTaskAction {
    #[serde(default)]
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) task_id: String,
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) progress_note: String,
    #[serde(default)]
    pub(crate) remaining_note: String,
    #[serde(default = "default_task_item_kind")]
    pub(crate) item_kind: String,
    #[serde(default)]
    pub(crate) next_step: String,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillNodeFactOp {
    /// `"add"` | `"merge"` | `"supersede"`
    #[serde(default)]
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) fact_id: String,
    #[serde(default = "default_fact_kind")]
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) text: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum DistillLinkTarget {
    Typed(DistillTypedLink),
    Name(String),
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillTypedLink {
    pub(crate) name: String,
    #[serde(default = "default_link_relation")]
    pub(crate) relation: String,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillNodeAction {
    /// 命中已有卡片时直接回传 `target_node_id`，彻底避免换名分裂
    #[serde(default)]
    pub(crate) target_node_id: String,
    #[serde(default)]
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) summary: String,
    #[serde(default = "default_mastery")]
    pub(crate) mastery: i64,
    #[serde(default = "default_node_status")]
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) related_nodes: Vec<DistillLinkTarget>,
    /// 卡片别名列表（换种说法也归到同一张卡）
    #[serde(default)]
    pub(crate) aliases: Vec<String>,
    /// 双语/大白话检索标签（路线 1 核心）
    #[serde(default)]
    pub(crate) search_tags: Vec<String>,
    /// 卡片内部独立要点增量演进操作
    #[serde(default)]
    pub(crate) fact_ops: Vec<DistillNodeFactOp>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DistillTimeline {
    #[serde(default)]
    pub(crate) user_goal: String,
    #[serde(default)]
    pub(crate) summary: String,
    #[serde(default)]
    pub(crate) ai_outcome: String,
    #[serde(default)]
    pub(crate) keywords: Vec<String>,
    #[serde(default = "default_journal_kind")]
    pub(crate) kind: String,
}

/// 生成用于短句去重/任务合并的词袋与双字切片集合（忽略标点与空白，大小写不敏感）。
fn similarity_tokens(text: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    let low = text.trim().to_lowercase();
    for word in low
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= 2)
    {
        set.insert(word.to_string());
    }
    let non_ascii: Vec<char> = low
        .chars()
        .filter(|c| !c.is_ascii() && !c.is_whitespace() && !c.is_ascii_punctuation())
        .collect();
    if non_ascii.len() >= 2 {
        for win in non_ascii.windows(2) {
            set.insert(win.iter().collect());
        }
    } else if non_ascii.len() == 1 {
        set.insert(non_ascii[0].to_string());
    }
    set
}

pub(crate) fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let sa = similarity_tokens(a);
    let sb = similarity_tokens(b);
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn parse_ymd_prefix(ts: &str) -> Option<NaiveDate> {
    let head: String = ts.trim().chars().take(10).collect();
    if head.len() < 10 {
        return None;
    }
    NaiveDate::parse_from_str(&head, "%Y-%m-%d").ok()
}

fn merge_unique_strings(existing: &mut Vec<String>, incoming: &[String], cap: usize) {
    let mut seen: HashSet<String> = existing.iter().map(|s| s.to_lowercase()).collect();
    for item in incoming {
        let trimmed = item.trim();
        if trimmed.chars().count() < 2 {
            continue;
        }
        if seen.insert(trimmed.to_lowercase()) {
            existing.push(trimmed.to_string());
            if existing.len() >= cap {
                break;
            }
        }
    }
}

fn synthesize_node_summary(
    existing_summary: &str,
    facts: &[NodeFact],
    incoming_summary: &str,
) -> String {
    let inc = incoming_summary.trim();
    let active: Vec<&NodeFact> = facts.iter().filter(|f| f.is_active()).collect();
    if active.len() <= 1 {
        if !inc.is_empty() {
            return inc.to_string();
        }
        return active
            .first()
            .map(|f| f.text.clone())
            .unwrap_or_else(|| existing_summary.to_string());
    }
    // 如果模型返回的新 summary 已经较完整（>= 旧 summary 一半长度且含核心信息），直接使用；否则由生效要点拼接保底，绝不丢失历史
    if !inc.is_empty()
        && (existing_summary.trim().is_empty()
            || inc.chars().count() * 2 >= existing_summary.chars().count())
    {
        return inc.to_string();
    }
    active
        .iter()
        .take(4)
        .map(|f| {
            let t = f.text.trim();
            if t.chars().count() > 110 {
                format!("{}...", t.chars().take(110).collect::<String>())
            } else {
                t.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("；")
}

/// 将一批已整理好的 LLM 输出原子落库到 5 个抽屉（支持新增、融合演进、作废替代，绝不盲覆盖）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_llm_distill_conn(
    conn: &mut Connection,
    agent_id: &str,
    session_id: &str,
    turn_id: &str,
    ts: &str,
    project: &str,
    working_dir: &str,
    user_text: &str,
    ai_text: &str,
    touched_files: &[String],
    out: DistillOutput,
    allow_pin: bool,
) -> Result<(), String> {
    let project_key = normalize_project_key_str(working_dir);
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    // 0. 抽屉 5：冷仓库原文无损归档（无 FTS 膨胀）
    if !user_text.trim().is_empty() || !ai_text.trim().is_empty() {
        upsert_raw_turn_conn(
            &tx,
            &RawTurn {
                turn_id: turn_id.to_string(),
                ts: ts.to_string(),
                agent: agent_id.to_string(),
                project: project.to_string(),
                project_key: project_key.clone(),
                working_dir: working_dir.to_string(),
                session_id: session_id.to_string(),
                user_text: user_text.to_string(),
                ai_text: ai_text.to_string(),
                touched_files: touched_files.to_vec(),
            },
        )?;
    }

    // 1. 抽屉 1（profiles）：先处理旧版 profile_removes（强制保护 pinned=1 且限定作用域）
    for rem in &out.profile_removes {
        let r = rem.trim();
        if r.chars().count() >= 3 {
            let escaped = escape_like_pattern(r);
            let like_pat = format!("%{escaped}%");
            let victims: Vec<(i64, String)> = if !project_key.is_empty() {
                tx.prepare(
                    "SELECT rowid, id FROM profiles
                     WHERE agent_id = ?1 AND pinned = 0 AND text LIKE ?2 ESCAPE '\\'
                       AND (project_key = ?3 OR (project_key = '' AND LOWER(project) = LOWER(?4)))",
                )
                .ok()
                .and_then(|mut s| {
                    s.query_map(params![agent_id, like_pat, project_key, project], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                    })
                    .ok()
                    .map(|rows| rows.flatten().collect())
                })
                .unwrap_or_default()
            } else if !project.is_empty() {
                tx.prepare(
                    "SELECT rowid, id FROM profiles
                     WHERE agent_id = ?1 AND pinned = 0 AND text LIKE ?2 ESCAPE '\\'
                       AND LOWER(project) = LOWER(?3)",
                )
                .ok()
                .and_then(|mut s| {
                    s.query_map(params![agent_id, like_pat, project], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                    })
                    .ok()
                    .map(|rows| rows.flatten().collect())
                })
                .unwrap_or_default()
            } else {
                tx.prepare(
                    "SELECT rowid, id FROM profiles
                     WHERE agent_id = ?1 AND pinned = 0 AND text LIKE ?2 ESCAPE '\\'
                       AND project_key = '' AND project = ''",
                )
                .ok()
                .and_then(|mut s| {
                    s.query_map(params![agent_id, like_pat], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                    })
                    .ok()
                    .map(|rows| rows.flatten().collect())
                })
                .unwrap_or_default()
            };
            for (rid, pid) in victims {
                let _ = tx.execute("DELETE FROM profiles_fts WHERE rowid = ?1", params![rid]);
                let _ = tx.execute(
                    "DELETE FROM memory_anchors WHERE target_kind = 'profile' AND target_id = ?1",
                    params![pid],
                );
                let _ = tx.execute("DELETE FROM profiles WHERE id = ?1", params![pid]);
            }
        }
    }

    let mut existing_profiles = load_profiles_conn(&tx, Some(agent_id))?
        .get(agent_id)
        .cloned()
        .unwrap_or_default();
    let same_scope = |p: &ProfileFact| -> bool {
        if !project_key.is_empty() {
            p.project_key == project_key
                || (p.project_key.is_empty() && p.project.eq_ignore_ascii_case(project))
        } else {
            p.project_key.is_empty() && p.project.eq_ignore_ascii_case(project)
        }
    };

    // 1b. 处理 v3 结构化 profile_actions（支持 update / supersede / remove / add）
    for act in out.profile_actions {
        let text = act.text.trim().to_string();
        let tid = act.target_id.trim();
        let want_pin = act.pin && allow_pin;
        let kind = if act.kind.trim().is_empty() {
            default_profile_kind()
        } else {
            act.kind.trim().to_string()
        };

        match act.action.trim() {
            "remove" => {
                if let Some(pos) = existing_profiles
                    .iter()
                    .position(|p| p.id == tid && !p.pinned && same_scope(p))
                {
                    let mut old = existing_profiles.remove(pos);
                    old.status = "superseded".into();
                    old.invalid_at = ts.to_string();
                    upsert_profile_conn(&tx, agent_id, &old)?;
                }
            }
            "supersede" => {
                if text.is_empty() {
                    continue;
                }
                let new_pf = ProfileFact {
                    id: new_id("pf"),
                    text,
                    project: project.to_string(),
                    project_key: project_key.clone(),
                    pinned: want_pin,
                    updated_at: ts.to_string(),
                    source_turn_id: turn_id.to_string(),
                    kind,
                    status: "active".into(),
                    ..Default::default()
                };
                if let Some(pos) = existing_profiles
                    .iter()
                    .position(|p| p.id == tid && (!p.pinned || allow_pin))
                {
                    let mut old = existing_profiles.remove(pos);
                    old.status = "superseded".into();
                    old.superseded_by = new_pf.id.clone();
                    old.invalid_at = ts.to_string();
                    upsert_profile_conn(&tx, agent_id, &old)?;
                }
                upsert_profile_conn(&tx, agent_id, &new_pf)?;
                existing_profiles.insert(0, new_pf);
            }
            "update" => {
                if text.is_empty() {
                    continue;
                }
                if let Some(pos) = existing_profiles.iter().position(|p| p.id == tid) {
                    let mut merged = existing_profiles[pos].clone();
                    merged.text = text;
                    merged.pinned = merged.pinned || want_pin;
                    merged.kind = kind;
                    merged.updated_at = ts.to_string();
                    merged.source_turn_id = turn_id.to_string();
                    upsert_profile_conn(&tx, agent_id, &merged)?;
                    existing_profiles[pos] = merged;
                    continue;
                }
                let fact = ProfileFact {
                    id: new_id("pf"),
                    text,
                    project: project.to_string(),
                    project_key: project_key.clone(),
                    pinned: want_pin,
                    updated_at: ts.to_string(),
                    source_turn_id: turn_id.to_string(),
                    kind,
                    ..Default::default()
                };
                upsert_profile_conn(&tx, agent_id, &fact)?;
                existing_profiles.insert(0, fact);
            }
            _ => {
                if text.is_empty() {
                    continue;
                }
                upsert_or_merge_profile_in_tx(
                    &tx,
                    agent_id,
                    project,
                    &project_key,
                    turn_id,
                    ts,
                    text,
                    want_pin,
                    kind,
                    &mut existing_profiles,
                )?;
            }
        }
    }

    // 1c. 兼容旧版 profile_updates
    for item in out.profile_updates {
        let (text, raw_pin) = match item {
            DistillProfileInput::Obj(o) => (o.text.trim().to_string(), o.pin),
            DistillProfileInput::Str(s) => (s.trim().to_string(), false),
        };
        if text.is_empty() {
            continue;
        }
        upsert_or_merge_profile_in_tx(
            &tx,
            agent_id,
            project,
            &project_key,
            turn_id,
            ts,
            text,
            raw_pin && allow_pin,
            default_profile_kind(),
            &mut existing_profiles,
        )?;
    }

    // 2. 抽屉 2（tasks）：需求与学习进度本（支持 item_kind / next_step / 模糊标题合并 / 14 天 stale）
    let mut all_tasks = load_tasks_conn(&tx, Some(agent_id))?;
    let task_same_scope = |t: &NeuralTask| -> bool {
        project_key.is_empty()
            || t.project_key == project_key
            || (t.project_key.is_empty() && t.project.eq_ignore_ascii_case(project))
    };
    for act in out.task_actions {
        let title = act.title.trim();
        let tid = act.task_id.trim();
        let matched_idx = if !tid.is_empty() {
            all_tasks.iter().position(|t| t.id == tid)
        } else if !title.is_empty() {
            all_tasks
                .iter()
                .position(|t| {
                    t.status != "done" && t.title.eq_ignore_ascii_case(title) && task_same_scope(t)
                })
                .or_else(|| {
                    all_tasks
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.status != "done" && task_same_scope(t))
                        .map(|(idx, t)| (idx, jaccard_similarity(&t.title, title)))
                        .filter(|(_, sim)| *sim >= 0.65)
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(idx, _)| idx)
                })
        } else {
            None
        };

        match act.action.as_str() {
            "complete" => {
                if let Some(idx) = matched_idx {
                    let mut task = all_tasks[idx].clone();
                    task.status = "done".into();
                    if !act.progress_note.trim().is_empty() {
                        task.progress_note = act.progress_note.trim().to_string();
                    }
                    task.remaining_note.clear();
                    task.next_step.clear();
                    task.done_at = ts.to_string();
                    task.updated_at = ts.to_string();
                    if !task.source_turn_ids.contains(&turn_id.to_string()) {
                        task.source_turn_ids.push(turn_id.to_string());
                    }
                    upsert_task_conn(&tx, &task)?;
                    all_tasks[idx] = task;
                }
            }
            "create" | "update" => {
                if let Some(idx) = matched_idx {
                    let mut task = all_tasks[idx].clone();
                    task.status = "in_progress".into();
                    if !act.progress_note.trim().is_empty() {
                        task.progress_note = act.progress_note.trim().to_string();
                    }
                    if !act.remaining_note.trim().is_empty() {
                        task.remaining_note = act.remaining_note.trim().to_string();
                    }
                    if !act.next_step.trim().is_empty() {
                        task.next_step = act.next_step.trim().to_string();
                    }
                    if !act.item_kind.trim().is_empty() {
                        task.item_kind = act.item_kind.trim().to_string();
                    }
                    for f in touched_files {
                        if !task.touched_files.contains(f) {
                            task.touched_files.push(f.clone());
                        }
                    }
                    if !task.source_turn_ids.contains(&turn_id.to_string()) {
                        task.source_turn_ids.push(turn_id.to_string());
                    }
                    task.updated_at = ts.to_string();
                    upsert_task_conn(&tx, &task)?;
                    all_tasks[idx] = task;
                } else if !title.is_empty() {
                    let task = NeuralTask {
                        id: new_id("task"),
                        agent: agent_id.to_string(),
                        project: project.to_string(),
                        project_key: project_key.clone(),
                        working_dir: working_dir.to_string(),
                        title: title.to_string(),
                        progress_note: act.progress_note.trim().to_string(),
                        remaining_note: act.remaining_note.trim().to_string(),
                        touched_files: touched_files.to_vec(),
                        session_id: session_id.to_string(),
                        source_turn_ids: vec![turn_id.to_string()],
                        status: "in_progress".into(),
                        created_at: ts.to_string(),
                        updated_at: ts.to_string(),
                        done_at: String::new(),
                        item_kind: if act.item_kind.trim().is_empty() {
                            default_task_item_kind()
                        } else {
                            act.item_kind.trim().to_string()
                        },
                        next_step: act.next_step.trim().to_string(),
                        team_run_id: String::new(),
                        team_task_id: String::new(),
                    };
                    upsert_task_conn(&tx, &task)?;
                    all_tasks.push(task);
                }
            }
            _ => {}
        }
    }

    if let Some(now_day) = parse_ymd_prefix(ts) {
        for task in &mut all_tasks {
            if task.status == "in_progress" && task_same_scope(task) {
                if let Some(last_day) = parse_ymd_prefix(&task.updated_at) {
                    if (now_day - last_day).num_days() > 14 {
                        task.status = "stale".into();
                        upsert_task_conn(&tx, task)?;
                    }
                }
            }
        }
    }

    // 3. 抽屉 3（nodes）：知识与经验活卡片（按 ID / 名称 / 别名精准归并 + 独立要点演进，绝不覆写失忆）
    let mut all_nodes = load_nodes_conn(&tx, Some(agent_id))?;
    let mut touched_node_ids = Vec::new();
    let mut pending_relations: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for na in out.node_actions {
        let name = na.name.trim();
        let target_id = na.target_node_id.trim();
        if target_id.is_empty() && is_noise_node_name(name) {
            continue;
        }
        let norm_key = normalize_node_key(name);
        let alias_keys: HashSet<String> = na
            .aliases
            .iter()
            .map(|a| normalize_node_key(a))
            .filter(|k| !k.is_empty())
            .collect();

        let existing_pos = if !target_id.is_empty() {
            all_nodes.iter().position(|n| n.id == target_id)
        } else {
            None
        }
        .or_else(|| {
            all_nodes.iter().position(|n| {
                let same_proj = project_key.is_empty()
                    || n.project_key == project_key
                    || (n.project_key.is_empty() && n.project.eq_ignore_ascii_case(project));
                if !same_proj {
                    return false;
                }
                let nk = normalize_node_key(&n.name);
                (!norm_key.is_empty() && nk == norm_key)
                    || (!norm_key.is_empty()
                        && n.aliases.iter().any(|a| normalize_node_key(a) == norm_key))
                    || alias_keys.contains(&nk)
            })
        });

        let mut node = match existing_pos {
            Some(idx) => all_nodes[idx].clone(),
            None => NeuralNode {
                id: new_id(agent_id),
                name: name.to_string(),
                cat: agent_id.to_string(),
                project: project.to_string(),
                project_key: project_key.clone(),
                status: if na.status.trim().is_empty() {
                    "active".into()
                } else {
                    na.status.clone()
                },
                mastery: na.mastery.clamp(20, 100),
                tier: 0,
                windows: default_windows(),
                summary: String::new(),
                events: Vec::new(),
                updated_at: ts.to_string(),
                ..Default::default()
            },
        };

        // 如果命中已有卡片且模型给了不同叫法，自动把新名字收入别名列表
        if !name.is_empty()
            && normalize_node_key(name) != normalize_node_key(&node.name)
            && !is_noise_node_name(name)
        {
            merge_unique_strings(&mut node.aliases, &[name.to_string()], 10);
        }
        merge_unique_strings(&mut node.aliases, &na.aliases, 10);
        merge_unique_strings(&mut node.search_tags, &na.search_tags, 16);

        // 先确保老节点的旧 summary 已被收编为一条 active fact，绝不丢历史
        if node.facts.is_empty() && !node.summary.trim().is_empty() {
            node.facts.push(NodeFact {
                id: new_id("nf"),
                kind: "fact".into(),
                text: node.summary.trim().to_string(),
                status: "active".into(),
                updated_at: node.updated_at.clone(),
                ..Default::default()
            });
        }

        // 应用独立要点演进操作（add / merge / supersede）
        if !na.fact_ops.is_empty() {
            for fop in na.fact_ops {
                let text = fop.text.trim();
                if text.is_empty() {
                    continue;
                }
                let fkind = if fop.kind.trim().is_empty() {
                    default_fact_kind()
                } else {
                    fop.kind.trim().to_string()
                };
                match fop.action.trim() {
                    "supersede" => {
                        let new_f = NodeFact {
                            id: new_id("nf"),
                            kind: fkind,
                            text: text.to_string(),
                            status: "active".into(),
                            source_turn_id: turn_id.to_string(),
                            updated_at: ts.to_string(),
                            ..Default::default()
                        };
                        if let Some(old) = node
                            .facts
                            .iter_mut()
                            .find(|f| f.id == fop.fact_id.trim() && f.is_active())
                        {
                            old.status = "superseded".into();
                            old.superseded_by = new_f.id.clone();
                            old.invalid_at = ts.to_string();
                        }
                        node.facts.insert(0, new_f);
                    }
                    "merge" | "update" => {
                        if let Some(old) = node
                            .facts
                            .iter_mut()
                            .find(|f| f.id == fop.fact_id.trim() && f.is_active())
                        {
                            old.text = text.to_string();
                            old.kind = fkind;
                            old.source_turn_id = turn_id.to_string();
                            old.updated_at = ts.to_string();
                        } else {
                            upsert_or_merge_node_fact(&mut node.facts, fkind, text, turn_id, ts);
                        }
                    }
                    _ => {
                        upsert_or_merge_node_fact(&mut node.facts, fkind, text, turn_id, ts);
                    }
                }
            }
        } else if !na.summary.trim().is_empty() {
            // 兼容旧版仅传 summary 的结构：追加/融合为新要点，绝不丢弃既有要点
            upsert_or_merge_node_fact(
                &mut node.facts,
                "fact".into(),
                na.summary.trim(),
                turn_id,
                ts,
            );
        }
        node.facts.truncate(24);

        let old_summary = node.summary.clone();
        node.summary = synthesize_node_summary(&old_summary, &node.facts, &na.summary);
        node.status = if na.status.trim().is_empty() {
            node.status
        } else {
            na.status
        };
        node.mastery = na.mastery.clamp(20, 100);
        node.updated_at = ts.to_string();

        let event_line = if !na.summary.trim().is_empty() {
            na.summary.trim().to_string()
        } else {
            node.summary.clone()
        };
        if !event_line.is_empty() && node.events.first().is_none_or(|e| e.text != event_line) {
            node.events.insert(
                0,
                NodeEvent {
                    ts: ts.to_string(),
                    text: event_line,
                },
            );
            node.events.truncate(12);
        }

        upsert_node_conn(&tx, &node)?;
        if let Some(idx) = existing_pos {
            all_nodes[idx] = node.clone();
        } else {
            all_nodes.push(node.clone());
        }
        if !touched_node_ids.contains(&node.id) {
            touched_node_ids.push(node.id.clone());
        }
        if !na.related_nodes.is_empty() {
            let rels: Vec<(String, String)> = na
                .related_nodes
                .into_iter()
                .map(|r| match r {
                    DistillLinkTarget::Typed(tl) => (tl.name, tl.relation),
                    DistillLinkTarget::Name(n) => (n, default_link_relation()),
                })
                .collect();
            pending_relations.push((node.id, rels));
        }
    }

    for (src_id, rels) in pending_relations {
        for (rel_name, rel_kind) in rels {
            let rel_key = normalize_node_key(&rel_name);
            if rel_key.is_empty() {
                continue;
            }
            if let Some(target) = all_nodes.iter().find(|n| {
                n.id != src_id
                    && (normalize_node_key(&n.name) == rel_key
                        || n.aliases.iter().any(|a| normalize_node_key(a) == rel_key))
            }) {
                upsert_link_conn(&tx, &src_id, &target.id, &rel_kind);
            }
        }
    }
    for pair in touched_node_ids.windows(2) {
        if pair[0] != pair[1] {
            upsert_link_conn(&tx, &pair[0], &pair[1], "related");
        }
    }

    // 4. 抽屉 4（timeline）：每日精华大事记（每批次仅落 1 条高纯度里程碑，并挂接卡片 ID）
    if let Some(tl) = out.timeline {
        let sum = tl.summary.trim();
        let goal = tl.user_goal.trim();
        if !sum.is_empty() || !goal.is_empty() {
            insert_timeline_conn(
                &tx,
                &TimelineEvent {
                    id: new_id("tl"),
                    ts: ts.to_string(),
                    agent: agent_id.to_string(),
                    project: project.to_string(),
                    project_key,
                    working_dir: working_dir.to_string(),
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    raw_file: "sqlite".into(),
                    user_goal: if goal.is_empty() {
                        user_text.chars().take(60).collect()
                    } else {
                        goal.to_string()
                    },
                    summary: if sum.is_empty() {
                        ai_text.chars().take(120).collect()
                    } else {
                        sum.to_string()
                    },
                    ai_outcome: tl.ai_outcome.trim().to_string(),
                    keywords: tl.keywords,
                    touched_files: touched_files.to_vec(),
                    distilled_by: "llm".into(),
                    kind: if tl.kind.trim().is_empty() {
                        default_journal_kind()
                    } else {
                        tl.kind
                    },
                    team_run_id: String::new(),
                    node_ids: touched_node_ids,
                },
            )?;
        }
    }

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn upsert_or_merge_profile_in_tx(
    tx: &Connection,
    agent_id: &str,
    project: &str,
    project_key: &str,
    turn_id: &str,
    ts: &str,
    text: String,
    want_pin: bool,
    kind: String,
    existing_profiles: &mut Vec<ProfileFact>,
) -> Result<(), String> {
    let same_scope = |p: &ProfileFact| -> bool {
        if !project_key.is_empty() {
            p.project_key == project_key
                || (p.project_key.is_empty() && p.project.eq_ignore_ascii_case(project))
        } else {
            p.project_key.is_empty() && p.project.eq_ignore_ascii_case(project)
        }
    };
    if let Some(pos) = existing_profiles
        .iter()
        .position(|p| p.text == text && same_scope(p))
    {
        if want_pin && !existing_profiles[pos].pinned {
            existing_profiles[pos].pinned = true;
            existing_profiles[pos].updated_at = ts.to_string();
            existing_profiles[pos].source_turn_id = turn_id.to_string();
            upsert_profile_conn(tx, agent_id, &existing_profiles[pos])?;
        }
        return Ok(());
    }
    let similar_pos = existing_profiles
        .iter()
        .enumerate()
        .filter(|(_, p)| same_scope(p))
        .map(|(idx, p)| (idx, jaccard_similarity(&p.text, &text)))
        .filter(|(_, sim)| *sim >= 0.78)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx);

    if let Some(pos) = similar_pos {
        let mut merged = existing_profiles[pos].clone();
        merged.text = text;
        merged.pinned = merged.pinned || want_pin;
        merged.updated_at = ts.to_string();
        merged.source_turn_id = turn_id.to_string();
        upsert_profile_conn(tx, agent_id, &merged)?;
        existing_profiles[pos] = merged;
        return Ok(());
    }

    let fact = ProfileFact {
        id: new_id("pf"),
        text,
        project: project.to_string(),
        project_key: project_key.to_string(),
        pinned: want_pin,
        updated_at: ts.to_string(),
        source_turn_id: turn_id.to_string(),
        kind,
        ..Default::default()
    };
    upsert_profile_conn(tx, agent_id, &fact)?;
    existing_profiles.insert(0, fact);
    Ok(())
}

fn upsert_or_merge_node_fact(
    facts: &mut Vec<NodeFact>,
    kind: String,
    text: &str,
    turn_id: &str,
    ts: &str,
) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    if let Some(existing) = facts.iter_mut().find(|f| {
        f.is_active() && (f.text == trimmed || jaccard_similarity(&f.text, trimmed) >= 0.80)
    }) {
        existing.text = trimmed.to_string();
        existing.updated_at = ts.to_string();
        if !turn_id.is_empty() {
            existing.source_turn_id = turn_id.to_string();
        }
        return;
    }
    facts.insert(
        0,
        NodeFact {
            id: new_id("nf"),
            kind,
            text: trimmed.to_string(),
            status: "active".into(),
            source_turn_id: turn_id.to_string(),
            updated_at: ts.to_string(),
            ..Default::default()
        },
    );
}

/// 规则兜底仅在用户显式开启 `rule_only` 模式或单元测试中调用；`llm_auto` 失败绝不调用此函数写垃圾。
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_fallback_distill_conn(
    conn: &Connection,
    agent_id: &str,
    session_id: &str,
    turn_id: &str,
    ts: &str,
    project: &str,
    working_dir: &str,
    user_text: &str,
    ai_text: &str,
    touched_files: &[String],
    mode: &str,
) -> Result<(), String> {
    let goal: String = user_text
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .chars()
        .take(60)
        .collect();
    if is_noise_node_name(&goal) {
        return Ok(());
    }
    let outcome: String = ai_text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
        .chars()
        .take(120)
        .collect();

    insert_timeline_conn(
        conn,
        &TimelineEvent {
            id: new_id("tl"),
            ts: ts.to_string(),
            agent: agent_id.to_string(),
            project: project.to_string(),
            project_key: normalize_project_key_str(working_dir),
            working_dir: working_dir.to_string(),
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            raw_file: "sqlite".into(),
            user_goal: goal.clone(),
            summary: goal,
            ai_outcome: outcome,
            keywords: Vec::new(),
            touched_files: touched_files.to_vec(),
            distilled_by: mode.to_string(),
            ..Default::default()
        },
    )?;
    Ok(())
}

/// 老记忆一键体检整理（供 `neural:consolidate-legacy` 手动触发）：
/// - 合并同项目下名称规范化高度重复的碎片节点，保留全部要点并把旧名收编进 `aliases`；
/// - 为所有节点/任务/规矩/时间线重建分列 FTS5 与精确路牌索引。
pub(crate) fn consolidate_legacy_nodes_conn(
    conn: &mut Connection,
    agent_filter: Option<&str>,
) -> Result<(usize, usize), String> {
    let nodes = load_nodes_conn(conn, agent_filter)?;
    let mut merged_count = 0usize;
    let mut visited: HashSet<String> = HashSet::new();

    for i in 0..nodes.len() {
        let base = &nodes[i];
        if visited.contains(&base.id) {
            continue;
        }
        let base_key = normalize_node_key(&base.name);
        let mut combined = base.clone();
        let mut duplicates: Vec<NeuralNode> = Vec::new();

        for other in nodes.iter().skip(i + 1) {
            if visited.contains(&other.id) || other.cat != base.cat {
                continue;
            }
            let same_proj = base.project_key == other.project_key
                && base.project.eq_ignore_ascii_case(&other.project);
            if !same_proj {
                continue;
            }
            let other_key = normalize_node_key(&other.name);
            let is_dup = (!base_key.is_empty() && base_key == other_key)
                || jaccard_similarity(&base.name, &other.name) >= 0.78;
            if is_dup {
                visited.insert(other.id.clone());
                duplicates.push(other.clone());
            }
        }

        if !duplicates.is_empty() {
            if combined.facts.is_empty() && !combined.summary.trim().is_empty() {
                combined.facts.push(NodeFact {
                    id: new_id("nf"),
                    kind: "fact".into(),
                    text: combined.summary.trim().to_string(),
                    status: "active".into(),
                    updated_at: combined.updated_at.clone(),
                    ..Default::default()
                });
            }
            for dup in duplicates {
                if normalize_node_key(&dup.name) != base_key {
                    merge_unique_strings(&mut combined.aliases, &[dup.name.clone()], 10);
                }
                merge_unique_strings(&mut combined.aliases, &dup.aliases, 10);
                merge_unique_strings(&mut combined.search_tags, &dup.search_tags, 16);
                for f in dup.facts {
                    if f.is_active() {
                        upsert_or_merge_node_fact(
                            &mut combined.facts,
                            f.kind,
                            &f.text,
                            &f.source_turn_id,
                            &f.updated_at,
                        );
                    }
                }
                if !dup.summary.trim().is_empty() {
                    upsert_or_merge_node_fact(
                        &mut combined.facts,
                        "fact".into(),
                        &dup.summary,
                        "",
                        &dup.updated_at,
                    );
                }
                combined.hit_count += dup.hit_count;
                combined.mastery = combined.mastery.max(dup.mastery);
                delete_node_with_indexes_conn(conn, &dup.id)?;
                merged_count += 1;
            }
            combined.facts.truncate(24);
            combined.summary =
                synthesize_node_summary(&combined.summary, &combined.facts, &combined.summary);
            combined.updated_at = now_ts_minute();
            upsert_node_conn(conn, &combined)?;
        }
    }

    super::schema::rebuild_all_v3_indexes_conn(conn)?;
    let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
    let total_nodes = load_nodes_conn(conn, agent_filter)
        .map(|v| v.len())
        .unwrap_or(0);
    Ok((merged_count, total_nodes))
}
