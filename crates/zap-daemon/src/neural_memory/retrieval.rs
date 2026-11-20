//! 多通道极速精准检索引擎：
//! - 通道 1：精确路牌 B-Tree 直达（文件/表名/接口/符号/错误码/Team 单号）
//! - 通道 2：分列加权 FTS5 BM25（标题/别名/标签权重 5~10 倍于正文）
//! - 综合重排：路牌分 + BM25 归一化 + 词项覆盖率 + 标题/别名/标签精确加分 + 时间衰减 + 使用率热度
//! - 宁缺毋滥硬及格线截断 + 1-hop 脑图邻居扩展 + 冷层原始对话按需检索

use std::collections::HashMap;

use chrono::{Local, NaiveDate};
use rusqlite::Connection;

use super::search::{escape_like_pattern, DateRange, ProjectScope, QueryPlan};
use super::store::{
    load_neighbor_nodes_conn, load_nodes_by_ids_conn, load_profiles_by_ids_conn,
    load_tasks_by_ids_conn, load_timeline_by_ids_conn, lookup_anchor_hits_conn, map_node_row,
    map_timeline_row,
};
use super::types::{NeuralNode, NeuralTask, ProfileFact, RawTurn, TimelineEvent};

pub(crate) const MIN_NODE_SCORE: f64 = 16.0;
pub(crate) const MIN_TIMELINE_SCORE: f64 = 14.0;
pub(crate) const MIN_TASK_SCORE: f64 = 12.0;
pub(crate) const MIN_PROFILE_SCORE: f64 = 12.0;

#[derive(Debug, Clone)]
pub(crate) struct RankedNode {
    pub(crate) node: NeuralNode,
    pub(crate) score: f64,
    pub(crate) anchor_hits: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct RankedTimeline {
    pub(crate) event: TimelineEvent,
    pub(crate) score: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct RankedTask {
    pub(crate) task: NeuralTask,
    pub(crate) score: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct RankedProfile {
    pub(crate) profile: ProfileFact,
    pub(crate) score: f64,
}

fn scope_matches(project_key: &str, project_name: &str, scope: Option<ProjectScope<'_>>) -> i8 {
    let Some(sc) = scope else {
        return 0;
    };
    if !sc.key.is_empty() && !project_key.is_empty() && project_key == sc.key {
        return 2;
    }
    if project_key.is_empty()
        && !sc.name.is_empty()
        && !project_name.is_empty()
        && project_name.eq_ignore_ascii_case(sc.name)
    {
        return 2;
    }
    if project_key.is_empty() && project_name.is_empty() {
        return 1;
    }
    -1
}

fn recency_multiplier(ts: &str, floor: f64, half_life_days: f64) -> f64 {
    let head: String = ts.trim().chars().take(10).collect();
    let Ok(day) = NaiveDate::parse_from_str(&head, "%Y-%m-%d") else {
        return 0.92;
    };
    let today = Local::now().date_naive();
    let age_days = (today - day).num_days().max(0) as f64;
    let decay = (-age_days * std::f64::consts::LN_2 / half_life_days).exp();
    floor + (1.0 - floor) * decay
}

fn normalize_bm25(raw_bm25: f64, max_pts: f64) -> f64 {
    // SQLite FTS5 bm25() 返回越小（越负）越相关的值
    let pos = (-raw_bm25).max(0.0);
    max_pts * (pos / (pos + 6.0))
}

fn append_project_scope_sql(
    conditions: &mut Vec<String>,
    dyn_params: &mut Vec<rusqlite::types::Value>,
    param_idx: &mut usize,
    alias: &str,
    scope: Option<ProjectScope<'_>>,
) {
    let Some(sc) = scope else {
        return;
    };
    if !sc.key.is_empty() && !sc.name.is_empty() {
        let p_key = *param_idx;
        let p_name = *param_idx + 1;
        *param_idx += 2;
        conditions.push(format!(
            "(({alias}.project_key != '' AND {alias}.project_key = ?{p_key}) OR ({alias}.project_key = '' AND LOWER({alias}.project) = LOWER(?{p_name})) OR ({alias}.project_key = '' AND {alias}.project = ''))"
        ));
        dyn_params.push(rusqlite::types::Value::from(sc.key.to_string()));
        dyn_params.push(rusqlite::types::Value::from(sc.name.to_string()));
    } else if !sc.key.is_empty() {
        let p_key = *param_idx;
        *param_idx += 1;
        conditions.push(format!(
            "({alias}.project_key = ?{p_key} OR ({alias}.project_key = '' AND {alias}.project = ''))"
        ));
        dyn_params.push(rusqlite::types::Value::from(sc.key.to_string()));
    } else if !sc.name.is_empty() {
        let p_name = *param_idx;
        *param_idx += 1;
        conditions.push(format!(
            "(LOWER({alias}.project) = LOWER(?{p_name}) OR ({alias}.project_key = '' AND {alias}.project = ''))"
        ));
        dyn_params.push(rusqlite::types::Value::from(sc.name.to_string()));
    }
}

fn fts_nodes_candidates(
    conn: &Connection,
    agent_id: &str,
    fts_query: &str,
    scope: Option<ProjectScope<'_>>,
    limit: usize,
) -> Vec<(NeuralNode, f64)> {
    let mut conditions = vec![
        "n.agent_id = ?1".to_string(),
        "nodes_fts MATCH ?2".to_string(),
    ];
    let mut dyn_params: Vec<rusqlite::types::Value> = vec![
        rusqlite::types::Value::from(agent_id.to_string()),
        rusqlite::types::Value::from(fts_query.to_string()),
    ];
    let mut param_idx = 3usize;
    append_project_scope_sql(&mut conditions, &mut dyn_params, &mut param_idx, "n", scope);
    dyn_params.push(rusqlite::types::Value::from(limit as i64));

    let sql = format!(
        "SELECT n.id, n.agent_id, n.name, n.project, n.project_key, n.status, n.mastery, n.tier,
                n.windows_json, n.summary, n.events_json, n.updated_at, n.aliases_json,
                n.search_tags_json, n.facts_json, n.hit_count, n.last_used_at,
                bm25(nodes_fts, 10.0, 7.0, 4.0, 1.0) AS rank_score
         FROM nodes_fts
         JOIN nodes n ON n.rowid = nodes_fts.rowid
         WHERE {}
         ORDER BY rank_score ASC
         LIMIT ?{param_idx}",
        conditions.join(" AND ")
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return Vec::new();
    };
    stmt.query_map(rusqlite::params_from_iter(dyn_params), |row| {
        let node = map_node_row(row)?;
        let bm25_val: f64 = row.get(17).unwrap_or(0.0);
        Ok((node, bm25_val))
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

/// 抽屉 3 多通道检索 + 综合打分重排 + 及格线截断（宁缺毋滥）。
pub(crate) fn query_nodes_ranked_conn(
    conn: &Connection,
    agent_id: &str,
    plan: &QueryPlan,
    scope: Option<ProjectScope<'_>>,
    limit: usize,
) -> Vec<RankedNode> {
    // node_id -> (node, best_bm25_opt, anchor_hits, hit_strict)
    let mut cands: HashMap<String, (NeuralNode, Option<f64>, usize, bool)> = HashMap::new();

    // 通道 1：精确路牌直达
    if !plan.anchors.is_empty() {
        let anchor_hits = lookup_anchor_hits_conn(conn, agent_id, &plan.anchors, "node");
        if !anchor_hits.is_empty() {
            let ids: Vec<String> = anchor_hits.keys().cloned().collect();
            for node in load_nodes_by_ids_conn(conn, agent_id, &ids) {
                if scope.is_some() && scope_matches(&node.project_key, &node.project, scope) < 0 {
                    continue;
                }
                let hits = *anchor_hits.get(&node.id).unwrap_or(&1);
                cands.insert(node.id.clone(), (node, None, hits, true));
            }
        }
    }

    // 通道 2：分列加权 FTS5（strict 优先 + broad 补充候选池）
    if let Some(ref fq) = plan.fts_strict {
        for (node, rank) in fts_nodes_candidates(conn, agent_id, fq, scope, 16) {
            cands
                .entry(node.id.clone())
                .and_modify(|e| {
                    e.1 = Some(e.1.map_or(rank, |old| old.min(rank)));
                    e.3 = true;
                })
                .or_insert((node, Some(rank), 0, true));
        }
    }
    if let Some(ref fq) = plan.fts_broad {
        for (node, rank) in fts_nodes_candidates(conn, agent_id, fq, scope, 24) {
            cands
                .entry(node.id.clone())
                .and_modify(|e| {
                    e.1 = Some(e.1.map_or(rank, |old| old.min(rank)));
                })
                .or_insert((node, Some(rank), 0, false));
        }
    }

    let mut ranked: Vec<RankedNode> = Vec::new();
    for (_, (node, bm25_opt, anchor_hits, hit_strict)) in cands {
        let name_low = node.name.to_lowercase();
        let alias_low = node.aliases.join(" ").to_lowercase();
        let tags_low = node.search_tags.join(" ").to_lowercase();
        let active_facts_low = node
            .facts
            .iter()
            .filter(|f| plan.wants_superseded || f.is_active())
            .map(|f| f.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let body_low = format!(
            "{} {} {} {} {}",
            name_low,
            alias_low,
            tags_low,
            node.summary.to_lowercase(),
            active_facts_low
        );

        let anchor_pts = (anchor_hits as f64).min(3.0) * 34.0;
        let bm25_pts = bm25_opt.map(|r| normalize_bm25(r, 26.0)).unwrap_or(0.0);

        let mut title_hits = 0usize;
        let mut tag_hits = 0usize;
        let mut core_hits = 0usize;
        for t in &plan.core_terms {
            if name_low.contains(t) || alias_low.contains(t) {
                title_hits += 1;
                core_hits += 1;
            } else if tags_low.contains(t) {
                tag_hits += 1;
                core_hits += 1;
            } else if body_low.contains(t) {
                core_hits += 1;
            }
        }
        let expanded_hits = plan
            .expanded_terms
            .iter()
            .filter(|t| body_low.contains(t.as_str()))
            .count();
        let ctx_hits = plan
            .context_terms
            .iter()
            .filter(|t| body_low.contains(t.as_str()))
            .count();

        let title_pts = (title_hits as f64).min(3.0) * 14.0 + (tag_hits as f64).min(3.0) * 9.0;
        let strict_bonus = if hit_strict { 8.0 } else { 0.0 };
        let expanded_pts = (expanded_hits as f64).min(3.0) * 5.0;
        let ctx_pts = (ctx_hits as f64).min(2.0) * 6.5;

        let coverage = if !plan.core_terms.is_empty() {
            core_hits as f64 / plan.core_terms.len() as f64
        } else if !plan.context_terms.is_empty() {
            ctx_hits as f64 / plan.context_terms.len() as f64
        } else {
            1.0
        };

        let mut score = anchor_pts + bm25_pts + title_pts + strict_bonus + expanded_pts + ctx_pts;
        // 覆盖率惩罚：无路牌、无标题/标签命中且在长问题里只撞中极少孤词时，压低分数防止噪声混入
        if anchor_hits == 0 && title_hits == 0 && tag_hits == 0 {
            if plan.core_terms.len() >= 3 && core_hits <= 1 && expanded_hits == 0 {
                score *= 0.45;
            } else if !plan.core_terms.is_empty() && core_hits == 0 && expanded_hits == 0 {
                score *= 0.25;
            }
        } else {
            score += coverage * 10.0;
        }

        // 时间衰减 + 使用率热度 + 项目亲和度
        let floor = if node.status == "solid" { 0.94 } else { 0.84 };
        score *= recency_multiplier(&node.updated_at, floor, 45.0);
        score += (1.0 + node.hit_count.max(0) as f64).ln() * 2.5;
        match scope_matches(&node.project_key, &node.project, scope) {
            2 => score += 6.0,
            1 => score += 1.5,
            _ => {}
        }

        if score >= MIN_NODE_SCORE {
            ranked.push(RankedNode {
                node,
                score,
                anchor_hits,
            });
        }
    }

    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.node.hit_count.cmp(&a.node.hit_count))
            .then_with(|| b.node.updated_at.cmp(&a.node.updated_at))
    });
    ranked.truncate(limit);
    ranked
}

/// 1-hop 脑图邻居扩展：仅为高置信 Top-1 节点带出最多 1~2 个强关联邻居。
pub(crate) fn expand_top_node_neighbors_conn(
    conn: &Connection,
    agent_id: &str,
    ranked: &[RankedNode],
    max_neighbors: usize,
) -> Vec<(NeuralNode, String)> {
    let Some(top) = ranked.first() else {
        return Vec::new();
    };
    if top.score < 26.0 || max_neighbors == 0 {
        return Vec::new();
    }
    let existing: std::collections::HashSet<&str> =
        ranked.iter().map(|r| r.node.id.as_str()).collect();
    load_neighbor_nodes_conn(conn, agent_id, &top.node.id, max_neighbors + 2)
        .into_iter()
        .filter(|(n, _)| !existing.contains(n.id.as_str()))
        .take(max_neighbors)
        .collect()
}

fn fts_timeline_candidates(
    conn: &Connection,
    agent_id: &str,
    fts_query: &str,
    scope: Option<ProjectScope<'_>>,
    date_range: Option<&DateRange>,
    limit: usize,
) -> Vec<(TimelineEvent, f64)> {
    let mut conditions = vec![
        "t.agent_id = ?1".to_string(),
        "timeline_fts MATCH ?2".to_string(),
    ];
    let mut dyn_params: Vec<rusqlite::types::Value> = vec![
        rusqlite::types::Value::from(agent_id.to_string()),
        rusqlite::types::Value::from(fts_query.to_string()),
    ];
    let mut param_idx = 3usize;
    append_project_scope_sql(&mut conditions, &mut dyn_params, &mut param_idx, "t", scope);
    if let Some(dr) = date_range {
        let ps = param_idx;
        let pe = param_idx + 1;
        param_idx += 2;
        conditions.push(format!("t.ts >= ?{ps}"));
        conditions.push(format!("t.ts <= ?{pe}"));
        dyn_params.push(rusqlite::types::Value::from(dr.start.clone()));
        dyn_params.push(rusqlite::types::Value::from(dr.end.clone()));
    }
    dyn_params.push(rusqlite::types::Value::from(limit as i64));

    let sql = format!(
        "SELECT t.id, t.ts, t.agent_id, t.project, t.project_key, t.working_dir, t.session_id, t.turn_id,
                t.raw_file, t.user_goal, t.summary, t.ai_outcome, t.keywords_json, t.touched_files_json,
                t.distilled_by, t.kind, t.team_run_id, t.node_ids_json,
                bm25(timeline_fts, 8.0, 5.0, 2.5) AS rank_score
         FROM timeline_fts
         JOIN timeline t ON t.rowid = timeline_fts.rowid
         WHERE {}
         ORDER BY rank_score ASC
         LIMIT ?{param_idx}",
        conditions.join(" AND ")
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return Vec::new();
    };
    stmt.query_map(rusqlite::params_from_iter(dyn_params), |row| {
        let ev = map_timeline_row(row)?;
        let bm25_val: f64 = row.get(18).unwrap_or(0.0);
        Ok((ev, bm25_val))
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

/// 抽屉 4（每日精华大事记）多通道检索 + 综合打分重排 + 及格线截断。
pub(crate) fn query_timeline_ranked_conn(
    conn: &Connection,
    agent_id: &str,
    plan: &QueryPlan,
    scope: Option<ProjectScope<'_>>,
    date_range: Option<&DateRange>,
    limit: usize,
) -> Vec<RankedTimeline> {
    let mut cands: HashMap<String, (TimelineEvent, Option<f64>, usize, bool)> = HashMap::new();

    if !plan.anchors.is_empty() {
        let anchor_hits = lookup_anchor_hits_conn(conn, agent_id, &plan.anchors, "timeline");
        if !anchor_hits.is_empty() {
            let ids: Vec<String> = anchor_hits.keys().cloned().collect();
            for ev in load_timeline_by_ids_conn(conn, agent_id, &ids) {
                if scope.is_some() && scope_matches(&ev.project_key, &ev.project, scope) < 0 {
                    continue;
                }
                if let Some(dr) = date_range {
                    if ev.ts < dr.start || ev.ts > dr.end {
                        continue;
                    }
                }
                let hits = *anchor_hits.get(&ev.id).unwrap_or(&1);
                cands.insert(ev.id.clone(), (ev, None, hits, true));
            }
        }
    }

    if let Some(ref fq) = plan.fts_strict {
        for (ev, rank) in fts_timeline_candidates(conn, agent_id, fq, scope, date_range, 14) {
            cands
                .entry(ev.id.clone())
                .and_modify(|e| {
                    e.1 = Some(e.1.map_or(rank, |old| old.min(rank)));
                    e.3 = true;
                })
                .or_insert((ev, Some(rank), 0, true));
        }
    }
    if let Some(ref fq) = plan.fts_broad {
        for (ev, rank) in fts_timeline_candidates(conn, agent_id, fq, scope, date_range, 20) {
            cands
                .entry(ev.id.clone())
                .and_modify(|e| {
                    e.1 = Some(e.1.map_or(rank, |old| old.min(rank)));
                })
                .or_insert((ev, Some(rank), 0, false));
        }
    }

    let mut ranked = Vec::new();
    for (_, (ev, bm25_opt, anchor_hits, hit_strict)) in cands {
        let kw_low = ev.keywords.join(" ").to_lowercase();
        let files_low = ev.touched_files.join(" ").to_lowercase();
        let hay = format!(
            "{} {} {} {} {}",
            kw_low,
            files_low,
            ev.user_goal.to_lowercase(),
            ev.summary.to_lowercase(),
            ev.ai_outcome.to_lowercase()
        );
        let anchor_pts = (anchor_hits as f64).min(3.0) * 32.0;
        let bm25_pts = bm25_opt.map(|r| normalize_bm25(r, 24.0)).unwrap_or(0.0);
        let mut kw_hits = 0usize;
        let mut core_hits = 0usize;
        for t in &plan.core_terms {
            if kw_low.contains(t) || files_low.contains(t) {
                kw_hits += 1;
                core_hits += 1;
            } else if hay.contains(t) {
                core_hits += 1;
            }
        }
        let expanded_hits = plan
            .expanded_terms
            .iter()
            .filter(|t| hay.contains(t.as_str()))
            .count();

        let mut score = anchor_pts
            + bm25_pts
            + (kw_hits as f64).min(3.0) * 9.0
            + if hit_strict { 7.0 } else { 0.0 }
            + (expanded_hits as f64).min(3.0) * 4.5;

        if anchor_hits == 0 && kw_hits == 0 && plan.core_terms.len() >= 3 && core_hits <= 1 {
            score *= 0.45;
        }
        score *= recency_multiplier(&ev.ts, 0.80, 30.0);
        if scope_matches(&ev.project_key, &ev.project, scope) == 2 {
            score += 4.5;
        }

        if score >= MIN_TIMELINE_SCORE {
            ranked.push(RankedTimeline { event: ev, score });
        }
    }

    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.event.ts.cmp(&a.event.ts))
    });
    ranked.truncate(limit);
    ranked
}

/// 抽屉 2（需求与学习进度本）精准检索（路牌 + 分列 FTS + 进行中加权）。
pub(crate) fn query_tasks_ranked_conn(
    conn: &Connection,
    agent_id: &str,
    plan: &QueryPlan,
    scope: Option<ProjectScope<'_>>,
    limit: usize,
) -> Vec<RankedTask> {
    let mut cands: HashMap<String, (NeuralTask, Option<f64>, usize)> = HashMap::new();

    if !plan.anchors.is_empty() {
        let anchor_hits = lookup_anchor_hits_conn(conn, agent_id, &plan.anchors, "task");
        if !anchor_hits.is_empty() {
            let ids: Vec<String> = anchor_hits.keys().cloned().collect();
            for t in load_tasks_by_ids_conn(conn, agent_id, &ids) {
                if scope.is_some() && scope_matches(&t.project_key, &t.project, scope) < 0 {
                    continue;
                }
                let hits = *anchor_hits.get(&t.id).unwrap_or(&1);
                cands.insert(t.id.clone(), (t, None, hits));
            }
        }
    }

    for fq in [plan.fts_strict.as_deref(), plan.fts_broad.as_deref()]
        .into_iter()
        .flatten()
    {
        let mut conditions = vec![
            "t.agent_id = ?1".to_string(),
            "tasks_fts MATCH ?2".to_string(),
        ];
        let mut dyn_params: Vec<rusqlite::types::Value> = vec![
            rusqlite::types::Value::from(agent_id.to_string()),
            rusqlite::types::Value::from(fq.to_string()),
        ];
        let mut param_idx = 3usize;
        append_project_scope_sql(&mut conditions, &mut dyn_params, &mut param_idx, "t", scope);
        dyn_params.push(rusqlite::types::Value::from(16i64));
        let sql = format!(
            "SELECT t.id, t.agent_id, t.project, t.project_key, t.working_dir, t.title,
                    t.progress_note, t.remaining_note, t.touched_files_json, t.session_id,
                    t.source_turn_ids_json, t.status, t.created_at, t.updated_at, t.done_at,
                    t.item_kind, t.next_step, t.team_run_id, t.team_task_id,
                    bm25(tasks_fts, 10.0, 3.0) AS rank_score
             FROM tasks_fts
             JOIN tasks t ON t.rowid = tasks_fts.rowid
             WHERE {}
             ORDER BY rank_score ASC
             LIMIT ?{param_idx}",
            conditions.join(" AND ")
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            if let Ok(rows) = stmt.query_map(rusqlite::params_from_iter(dyn_params), |r| {
                let touched_str: String = r.get(8)?;
                let turns_str: String = r.get(10)?;
                let task = NeuralTask {
                    id: r.get(0)?,
                    agent: r.get(1)?,
                    project: r.get(2)?,
                    project_key: r.get(3)?,
                    working_dir: r.get(4)?,
                    title: r.get(5)?,
                    progress_note: r.get(6)?,
                    remaining_note: r.get(7)?,
                    touched_files: serde_json::from_str(&touched_str).unwrap_or_default(),
                    session_id: r.get(9)?,
                    source_turn_ids: serde_json::from_str(&turns_str).unwrap_or_default(),
                    status: r.get(11)?,
                    created_at: r.get(12)?,
                    updated_at: r.get(13)?,
                    done_at: r.get(14)?,
                    item_kind: r
                        .get::<_, String>(15)
                        .unwrap_or_else(|_| "requirement".into()),
                    next_step: r.get::<_, String>(16).unwrap_or_default(),
                    team_run_id: r.get::<_, String>(17).unwrap_or_default(),
                    team_task_id: r.get::<_, String>(18).unwrap_or_default(),
                };
                Ok((task, r.get::<_, f64>(19).unwrap_or(0.0)))
            }) {
                for (t, rank) in rows.flatten() {
                    cands
                        .entry(t.id.clone())
                        .and_modify(|e| {
                            e.1 = Some(e.1.map_or(rank, |old| old.min(rank)));
                        })
                        .or_insert((t, Some(rank), 0));
                }
            }
        }
    }

    let mut ranked = Vec::new();
    for (_, (task, bm25_opt, anchor_hits)) in cands {
        let mut score = (anchor_hits as f64).min(3.0) * 30.0
            + bm25_opt.map(|r| normalize_bm25(r, 24.0)).unwrap_or(0.0);
        if task.status != "done" {
            score += 6.0;
        }
        if score >= MIN_TASK_SCORE {
            ranked.push(RankedTask { task, score });
        }
    }
    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.task.updated_at.cmp(&a.task.updated_at))
    });
    ranked.truncate(limit);
    ranked
}

/// 抽屉 1（长期规矩与习惯）精准检索（路牌 + FTS5 + 置顶/热度加权）。
pub(crate) fn query_profiles_ranked_conn(
    conn: &Connection,
    agent_id: &str,
    plan: &QueryPlan,
    scope: Option<ProjectScope<'_>>,
    include_superseded: bool,
    limit: usize,
) -> Vec<RankedProfile> {
    let mut cands: HashMap<String, (ProfileFact, Option<f64>, usize)> = HashMap::new();

    if !plan.anchors.is_empty() {
        let anchor_hits = lookup_anchor_hits_conn(conn, agent_id, &plan.anchors, "profile");
        if !anchor_hits.is_empty() {
            let ids: Vec<String> = anchor_hits.keys().cloned().collect();
            for p in load_profiles_by_ids_conn(conn, agent_id, &ids, include_superseded) {
                if scope.is_some() && scope_matches(&p.project_key, &p.project, scope) < 0 {
                    continue;
                }
                let hits = *anchor_hits.get(&p.id).unwrap_or(&1);
                cands.insert(p.id.clone(), (p, None, hits));
            }
        }
    }

    for fq in [plan.fts_strict.as_deref(), plan.fts_broad.as_deref()]
        .into_iter()
        .flatten()
    {
        let mut conditions = vec![
            "p.agent_id = ?1".to_string(),
            "profiles_fts MATCH ?2".to_string(),
        ];
        if !include_superseded {
            conditions.push("(p.status = '' OR p.status = 'active')".to_string());
        }
        let mut dyn_params: Vec<rusqlite::types::Value> = vec![
            rusqlite::types::Value::from(agent_id.to_string()),
            rusqlite::types::Value::from(fq.to_string()),
        ];
        let mut param_idx = 3usize;
        append_project_scope_sql(&mut conditions, &mut dyn_params, &mut param_idx, "p", scope);
        dyn_params.push(rusqlite::types::Value::from(16i64));
        let sql = format!(
            "SELECT p.id, p.project, p.project_key, p.text, p.pinned, p.updated_at, p.source_turn_id,
                    p.kind, p.status, p.superseded_by, p.invalid_at, p.hit_count, p.last_used_at,
                    bm25(profiles_fts, 6.0) AS rank_score
             FROM profiles_fts
             JOIN profiles p ON p.rowid = profiles_fts.rowid
             WHERE {}
             ORDER BY rank_score ASC
             LIMIT ?{param_idx}",
            conditions.join(" AND ")
        );
        if let Ok(mut stmt) = conn.prepare(&sql) {
            if let Ok(rows) = stmt.query_map(rusqlite::params_from_iter(dyn_params), |r| {
                let pinned_int: i64 = r.get(4)?;
                let pf = ProfileFact {
                    id: r.get(0)?,
                    project: r.get(1)?,
                    project_key: r.get(2)?,
                    text: r.get(3)?,
                    pinned: pinned_int != 0,
                    updated_at: r.get(5)?,
                    source_turn_id: r.get(6)?,
                    kind: r.get::<_, String>(7).unwrap_or_else(|_| "rule".into()),
                    status: r.get::<_, String>(8).unwrap_or_else(|_| "active".into()),
                    superseded_by: r.get::<_, String>(9).unwrap_or_default(),
                    invalid_at: r.get::<_, String>(10).unwrap_or_default(),
                    hit_count: r.get::<_, i64>(11).unwrap_or(0),
                    last_used_at: r.get::<_, String>(12).unwrap_or_default(),
                };
                Ok((pf, r.get::<_, f64>(13).unwrap_or(0.0)))
            }) {
                for (p, rank) in rows.flatten() {
                    cands
                        .entry(p.id.clone())
                        .and_modify(|e| {
                            e.1 = Some(e.1.map_or(rank, |old| old.min(rank)));
                        })
                        .or_insert((p, Some(rank), 0));
                }
            }
        }
    }

    let mut ranked = Vec::new();
    for (_, (profile, bm25_opt, anchor_hits)) in cands {
        let mut score = (anchor_hits as f64).min(3.0) * 30.0
            + bm25_opt.map(|r| normalize_bm25(r, 24.0)).unwrap_or(0.0)
            + (1.0 + profile.hit_count.max(0) as f64).ln() * 2.0;
        if profile.pinned {
            score += 5.0;
        }
        if score >= MIN_PROFILE_SCORE {
            ranked.push(RankedProfile { profile, score });
        }
    }
    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.profile.pinned.cmp(&a.profile.pinned))
    });
    ranked.truncate(limit);
    ranked
}

/// 抽屉 5（冷仓库 `raw_turns`）按需轻量查询：无 FTS 膨胀，仅在显式索要原话/堆栈时按时间/作用域/核心词精准过滤。
pub(crate) fn query_raw_turns_cold_conn(
    conn: &Connection,
    agent_id: &str,
    terms: &[String],
    scope: Option<ProjectScope<'_>>,
    date_range: Option<&DateRange>,
    limit: usize,
) -> Vec<RawTurn> {
    let mut conditions = vec!["r.agent_id = ?1".to_string()];
    let mut dyn_params: Vec<rusqlite::types::Value> =
        vec![rusqlite::types::Value::from(agent_id.to_string())];
    let mut param_idx = 2usize;

    append_project_scope_sql(&mut conditions, &mut dyn_params, &mut param_idx, "r", scope);
    if let Some(dr) = date_range {
        let ps = param_idx;
        let pe = param_idx + 1;
        param_idx += 2;
        conditions.push(format!("r.ts >= ?{ps}"));
        conditions.push(format!("r.ts <= ?{pe}"));
        dyn_params.push(rusqlite::types::Value::from(dr.start.clone()));
        dyn_params.push(rusqlite::types::Value::from(dr.end.clone()));
    }

    let key_terms: Vec<&String> = terms
        .iter()
        .filter(|t| t.chars().count() >= 2)
        .take(3)
        .collect();
    if !key_terms.is_empty() {
        let mut or_parts = Vec::new();
        for t in key_terms {
            let p = param_idx;
            param_idx += 1;
            let pat = format!("%{}%", escape_like_pattern(t));
            or_parts.push(format!(
                "(r.user_text LIKE ?{p} ESCAPE '\\' OR r.ai_text LIKE ?{p} ESCAPE '\\' OR r.touched_files_json LIKE ?{p} ESCAPE '\\')"
            ));
            dyn_params.push(rusqlite::types::Value::from(pat));
        }
        conditions.push(format!("({})", or_parts.join(" OR ")));
    }
    dyn_params.push(rusqlite::types::Value::from(limit as i64));

    let sql = format!(
        "SELECT r.turn_id, r.ts, r.agent_id, r.project, r.project_key, r.working_dir, r.session_id,
                r.user_text, r.ai_text, r.touched_files_json
         FROM raw_turns r
         WHERE {}
         ORDER BY r.ts DESC
         LIMIT ?{param_idx}",
        conditions.join(" AND ")
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return Vec::new();
    };
    stmt.query_map(rusqlite::params_from_iter(dyn_params), |row| {
        let tf_str: String = row.get(9)?;
        Ok(RawTurn {
            turn_id: row.get(0)?,
            ts: row.get(1)?,
            agent: row.get(2)?,
            project: row.get(3)?,
            project_key: row.get(4)?,
            working_dir: row.get(5)?,
            session_id: row.get(6)?,
            user_text: row.get(7)?,
            ai_text: row.get(8)?,
            touched_files: serde_json::from_str(&tf_str).unwrap_or_default(),
        })
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}
