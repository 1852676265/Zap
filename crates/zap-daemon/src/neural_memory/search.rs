//! 文本切词（含 CJK/扩展汉字/日文假名/韩文音节）、时间范围解析与 FTS5 BM25 检索。

use std::collections::HashSet;

use chrono::{Datelike, Duration as ChronoDuration, Local, NaiveDate};
use rusqlite::Connection;
use zap_capabilities::reminder::strip_injected_reminders_for_display;

use super::store::KnownProject;
use super::types::{default_windows, NeuralNode, RawTurn, TimelineEvent};

pub(crate) struct DateRange {
    pub(crate) label: String,
    pub(crate) start: String,
    pub(crate) end: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProjectScope<'a> {
    pub(crate) key: &'a str,
    pub(crate) name: &'a str,
}

/// 判断字符是否属于 CJK 汉字（含扩展区/兼容区）、日文假名或韩文音节。
fn is_cjk_or_kana_hangul(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF   // CJK Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x20000..=0x2EBEF // CJK Extensions B-F
        | 0x3040..=0x309F // Hiragana
        | 0x30A0..=0x30FF // Katakana
        | 0xAC00..=0xD7AF // Hangul Syllables
    )
}

/// 生成供 FTS5 检索的混合文本：保留英文/标识符原词（含拆分子词），同时对东亚文字片段补充 2-gram / 3-gram 空格分词。
pub(crate) fn build_fts_search_text(parts: &[&str]) -> String {
    let joined = parts.join(" ");
    let mut tokens: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for word in joined
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.')
        .filter(|w| w.chars().count() >= 2)
    {
        let low = word.to_lowercase();
        if seen.insert(low.clone()) {
            tokens.push(low.clone());
        }
        for sub in low
            .split(['.', '-', '_'])
            .filter(|s| s.chars().count() >= 2)
        {
            let s = sub.to_string();
            if seen.insert(s.clone()) {
                tokens.push(s);
            }
        }
    }

    let mut cjk_run = Vec::new();
    let flush_cjk = |run: &mut Vec<char>, tokens: &mut Vec<String>, seen: &mut HashSet<String>| {
        if run.len() >= 2 {
            for win in run.windows(2) {
                let g: String = win.iter().collect();
                if seen.insert(g.clone()) {
                    tokens.push(g);
                }
            }
            if run.len() >= 3 {
                for win in run.windows(3) {
                    let g: String = win.iter().collect();
                    if seen.insert(g.clone()) {
                        tokens.push(g);
                    }
                }
            }
        }
        run.clear();
    };
    for ch in joined.chars() {
        if is_cjk_or_kana_hangul(ch) {
            cjk_run.push(ch);
        } else {
            flush_cjk(&mut cjk_run, &mut tokens, &mut seen);
        }
    }
    flush_cjk(&mut cjk_run, &mut tokens, &mut seen);

    if tokens.is_empty() {
        joined
    } else {
        format!("{joined} {}", tokens.join(" "))
    }
}

fn is_fts_stop_word(term: &str) -> bool {
    const STOPS: &[&str] = &[
        "这个",
        "那个",
        "这些",
        "那些",
        "怎么",
        "如何",
        "什么",
        "为什么",
        "为啥",
        "哪里",
        "哪个",
        "可以",
        "能否",
        "能不能",
        "一下",
        "一些",
        "一点",
        "之前",
        "以前",
        "刚才",
        "现在",
        "目前",
        "已经",
        "还是",
        "或者",
        "以及",
        "关于",
        "帮我",
        "帮忙",
        "看看",
        "看下",
        "查下",
        "查询",
        "请问",
        "问题",
        "需求",
        "修改",
        "改动",
        "处理",
        "解决",
        "我们",
        "你们",
        "他们",
        "自己",
        "一个",
        "没有",
        "不是",
        "就是",
        "这样",
        "那样",
        "这里",
        "那里",
        "时候",
        "东西",
        "地方",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "what",
        "how",
        "why",
        "when",
        "where",
        "can",
        "could",
        "would",
        "should",
        "our",
        "your",
        "from",
        "into",
        "about",
        "please",
        "check",
        "look",
        "see",
        "help",
        "need",
        "want",
        "have",
        "has",
        "was",
        "were",
        "are",
    ];
    STOPS.contains(&term)
}

/// 生成高精度优先的两段式 FTS5 查询：`(strict_and_query, broad_or_query)`。
pub(crate) fn build_fts_queries(query: &str) -> (Option<String>, Option<String>) {
    let mut primary_terms: Vec<String> = Vec::new();
    let mut cjk_terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for word in query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.')
        .filter(|w| w.chars().count() >= 2)
    {
        let clean = word.trim_matches('.').to_lowercase();
        if clean.len() < 2 || clean.len() > 48 || is_fts_stop_word(&clean) {
            continue;
        }
        let has_ascii = clean.chars().any(|c| c.is_ascii_alphanumeric());
        if has_ascii && seen.insert(clean.clone()) {
            let escaped = clean.replace('"', "");
            if !escaped.is_empty() {
                primary_terms.push(format!("\"{escaped}\""));
            }
        }
    }

    let mut cjk_run = Vec::new();
    let flush_cjk =
        |run: &mut Vec<char>, cjk_terms: &mut Vec<String>, seen: &mut HashSet<String>| {
            if run.len() >= 2 {
                if run.len() <= 4 {
                    let whole: String = run.iter().collect();
                    if !is_fts_stop_word(&whole) && seen.insert(whole.clone()) {
                        cjk_terms.push(format!("\"{whole}\""));
                    }
                }
                for win in run.windows(2) {
                    let g: String = win.iter().collect();
                    if !is_fts_stop_word(&g) && seen.insert(g.clone()) {
                        cjk_terms.push(format!("\"{g}\""));
                    }
                }
            }
            run.clear();
        };
    for ch in query.chars() {
        if is_cjk_or_kana_hangul(ch) {
            cjk_run.push(ch);
        } else {
            flush_cjk(&mut cjk_run, &mut cjk_terms, &mut seen);
        }
    }
    flush_cjk(&mut cjk_run, &mut cjk_terms, &mut seen);

    let mut all_terms = primary_terms.clone();
    all_terms.extend(cjk_terms);
    if all_terms.is_empty() {
        return (None, None);
    }

    let strict = if primary_terms.len() >= 2 {
        let top: Vec<String> = primary_terms.iter().take(4).cloned().collect();
        Some(top.join(" AND "))
    } else if all_terms.len() >= 2 {
        let top: Vec<String> = all_terms.iter().take(2).cloned().collect();
        Some(top.join(" AND "))
    } else {
        None
    };

    all_terms.truncate(20);
    let broad = Some(all_terms.join(" OR "));
    (strict, broad)
}

/// 剥离前端/后端注入的所有系统提醒块，还原用户的干净原始提问。
pub fn clean_user_prompt(raw: &str) -> String {
    strip_injected_reminders_for_display(raw).trim().to_string()
}

/// 识别斜杠指令（如 `/compact`、`/help`），同时放行 Unix 绝对路径（如 `/etc/nginx/nginx.conf`）。
pub(crate) fn is_slash_command(clean_query: &str) -> bool {
    let first = clean_query
        .trim_start()
        .split_whitespace()
        .next()
        .unwrap_or("");
    first.starts_with('/')
        && first.len() > 1
        && !first[1..].contains('/')
        && first[1..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ':')
}

pub(crate) fn normalize_node_key(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .collect()
}

pub(crate) fn escape_like_pattern(raw: &str) -> String {
    raw.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// 防垃圾守门：过滤纯确认语/寒暄/无实质内容对话。
pub(crate) fn is_trivial_turn(user_text: &str, ai_text: &str, touched_files: &[String]) -> bool {
    if !touched_files.is_empty() {
        return false;
    }
    let trimmed = user_text
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || "，。！？、~～… ".contains(c));
    if trimmed.is_empty() {
        return true;
    }
    let low = trimmed.to_lowercase();
    const NOISE_SET: &[&str] = &[
        "可以",
        "好的",
        "好",
        "好吧",
        "行",
        "嗯",
        "嗯嗯",
        "收到",
        "明白",
        "了解",
        "知道了",
        "谢谢",
        "多谢",
        "感谢",
        "辛苦了",
        "继续",
        "开始",
        "开始吧",
        "没问题",
        "可以的",
        "行吧",
        "ok",
        "okay",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "thanks",
        "thank you",
        "thx",
        "go",
        "go on",
        "continue",
        "done",
        "nice",
        "good",
        "great",
        "1",
        "2",
        "3",
    ];
    if NOISE_SET.contains(&low.as_str()) {
        return true;
    }
    let user_chars = trimmed.chars().count();
    let ai_chars = ai_text.trim().chars().count();
    user_chars <= 4 && ai_chars <= 140
}

pub(crate) fn is_noise_node_name(name: &str) -> bool {
    let trimmed = name
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || "，。！？、~～… ".contains(c));
    if trimmed.is_empty() || trimmed.chars().count() <= 1 {
        return true;
    }
    const NOISE_NAMES: &[&str] = &[
        "可以", "好的", "好", "嗯", "行", "谢谢", "收到", "继续", "ok", "yes", "no",
    ];
    NOISE_NAMES.contains(&trimmed.to_lowercase().as_str())
}

fn is_generic_project_name(name_low: &str) -> bool {
    const GENERIC: &[&str] = &[
        "app", "apps", "api", "web", "src", "test", "tests", "docs", "doc", "core", "main", "data",
        "server", "client", "code", "work", "repo", "home", "dist", "build", "temp", "tmp", "demo",
        "project", "service", "backend", "frontend",
    ];
    name_low.chars().count() < 4 || GENERIC.contains(&name_low)
}

fn is_ascii_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// 词边界感知的跨项目点名检测，避免 `api` / `web` 等短词或普通单词子串误触发跨项目 L3 注入。
pub(crate) fn detect_mentioned_project(
    query: &str,
    known_projects: &[KnownProject],
    current_name: &str,
    current_key: &str,
) -> Option<KnownProject> {
    let q_low = query.to_lowercase();
    for kp in known_projects {
        if !current_key.is_empty() && !kp.key.is_empty() && kp.key == current_key {
            continue;
        }
        if kp.key.is_empty()
            && !current_name.is_empty()
            && kp.name.eq_ignore_ascii_case(current_name)
        {
            continue;
        }
        let p_low = kp.name.trim().to_lowercase();
        if p_low.is_empty() || is_generic_project_name(&p_low) {
            continue;
        }
        // 查找带词边界的匹配位置
        let bytes = q_low.as_bytes();
        let mut search_from = 0usize;
        while let Some(rel) = q_low[search_from..].find(&p_low) {
            let start = search_from + rel;
            let end = start + p_low.len();
            let left_ok = start == 0 || !is_ascii_ident_char(bytes[start - 1]);
            let right_ok = end >= bytes.len() || !is_ascii_ident_char(bytes[end]);
            if left_ok && right_ok {
                return Some(kp.clone());
            }
            search_from = end;
        }
    }
    None
}

fn parse_cn_or_ascii_days(raw: &str) -> Option<i64> {
    let s = raw.trim();
    if let Ok(n) = s.parse::<i64>() {
        return (1..=365).contains(&n).then_some(n);
    }
    match s {
        "一" => Some(1),
        "两" | "二" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        "五" => Some(5),
        "六" => Some(6),
        "七" => Some(7),
        "十" => Some(10),
        "几" => Some(3),
        _ => None,
    }
}

pub(crate) fn parse_temporal_range(query: &str) -> Option<DateRange> {
    let today = Local::now().date_naive();
    let fmt_day = |d: NaiveDate| d.format("%Y-%m-%d").to_string();
    let low = query.to_lowercase();

    if query.contains("今天") || query.contains("今日") || low.contains("today") {
        let s = fmt_day(today);
        return Some(DateRange {
            label: s.clone(),
            start: format!("{s} 00:00"),
            end: format!("{s} 23:59"),
        });
    }
    if query.contains("昨天") || query.contains("昨日") || low.contains("yesterday") {
        let d = today - ChronoDuration::days(1);
        let s = fmt_day(d);
        return Some(DateRange {
            label: "yesterday".into(),
            start: format!("{s} 00:00"),
            end: format!("{s} 23:59"),
        });
    }
    if query.contains("前天") || low.contains("day before yesterday") {
        let d = today - ChronoDuration::days(2);
        let s = fmt_day(d);
        return Some(DateRange {
            label: s.clone(),
            start: format!("{s} 00:00"),
            end: format!("{s} 23:59"),
        });
    }
    if query.contains("本周")
        || query.contains("这周")
        || query.contains("这个星期")
        || low.contains("this week")
    {
        let offset = today.weekday().num_days_from_monday() as i64;
        let mon = today - ChronoDuration::days(offset);
        return Some(DateRange {
            label: "this week".into(),
            start: format!("{} 00:00", fmt_day(mon)),
            end: format!("{} 23:59", fmt_day(today)),
        });
    }
    if query.contains("上周")
        || query.contains("上星期")
        || low.contains("last week")
        || low.contains("past week")
    {
        let offset = today.weekday().num_days_from_monday() as i64;
        let this_mon = today - ChronoDuration::days(offset);
        let last_mon = this_mon - ChronoDuration::days(7);
        let last_sun = this_mon - ChronoDuration::days(1);
        return Some(DateRange {
            label: "last week".into(),
            start: format!("{} 00:00", fmt_day(last_mon)),
            end: format!("{} 23:59", fmt_day(last_sun)),
        });
    }
    if query.contains("近一周") || query.contains("最近一周") || query.contains("最近7天")
    {
        let start = today - ChronoDuration::days(6);
        return Some(DateRange {
            label: "last 7 days".into(),
            start: format!("{} 00:00", fmt_day(start)),
            end: format!("{} 23:59", fmt_day(today)),
        });
    }
    if query.contains("本月") || query.contains("这个月") || low.contains("this month") {
        let start = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?;
        return Some(DateRange {
            label: "this month".into(),
            start: format!("{} 00:00", fmt_day(start)),
            end: format!("{} 23:59", fmt_day(today)),
        });
    }
    if query.contains("上月")
        || query.contains("上个月")
        || low.contains("last month")
        || low.contains("past month")
    {
        let (y, m) = if today.month() == 1 {
            (today.year() - 1, 12)
        } else {
            (today.year(), today.month() - 1)
        };
        let start = NaiveDate::from_ymd_opt(y, m, 1)?;
        let next_month = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?;
        let end = next_month - ChronoDuration::days(1);
        return Some(DateRange {
            label: "last month".into(),
            start: format!("{} 00:00", fmt_day(start)),
            end: format!("{} 23:59", fmt_day(end)),
        });
    }
    if query.contains("今年") || low.contains("this year") {
        return Some(DateRange {
            label: "this year".into(),
            start: format!("{}-01-01 00:00", today.year()),
            end: format!("{}-12-31 23:59", today.year()),
        });
    }
    if query.contains("去年") || low.contains("last year") {
        return Some(DateRange {
            label: "last year".into(),
            start: format!("{}-01-01 00:00", today.year() - 1),
            end: format!("{}-12-31 23:59", today.year() - 1),
        });
    }
    for prefix in ["最近", "过去", "前", "近"] {
        if let Some(pos) = query.find(prefix) {
            let rest = &query[pos + prefix.len()..];
            if let Some(day_pos) = rest.find('天') {
                let num_slice = &rest[..day_pos];
                if let Some(days) = parse_cn_or_ascii_days(num_slice) {
                    let start = today - ChronoDuration::days(days - 1);
                    return Some(DateRange {
                        label: format!("last {days} days"),
                        start: format!("{} 00:00", fmt_day(start)),
                        end: format!("{} 23:59", fmt_day(today)),
                    });
                }
            }
        }
    }
    for prefix in ["last ", "past "] {
        if let Some(pos) = low.find(prefix) {
            let rest = &low[pos + prefix.len()..];
            let mut parts = rest.split_whitespace();
            if let (Some(n_str), Some(unit)) = (parts.next(), parts.next()) {
                if unit.starts_with("day") {
                    if let Some(days) = parse_cn_or_ascii_days(n_str) {
                        let start = today - ChronoDuration::days(days - 1);
                        return Some(DateRange {
                            label: format!("last {days} days"),
                            start: format!("{} 00:00", fmt_day(start)),
                            end: format!("{} 23:59", fmt_day(today)),
                        });
                    }
                }
            }
        }
    }
    for token in query.split(|c: char| !c.is_ascii_digit() && c != '-' && c != '/') {
        if token.len() == 10 {
            let norm = token.replace('/', "-");
            if NaiveDate::parse_from_str(&norm, "%Y-%m-%d").is_ok() {
                return Some(DateRange {
                    label: norm.clone(),
                    start: format!("{norm} 00:00"),
                    end: format!("{norm} 23:59"),
                });
            }
        }
    }
    None
}

pub(crate) fn should_inject_l3_verbatim(query: &str) -> bool {
    const HINTS: &[&str] = &[
        "原文",
        "原话",
        "堆栈",
        "报错",
        "sql",
        "怎么写",
        "verbatim",
        "exact quote",
        "stack trace",
        "stacktrace",
    ];
    let low = query.to_lowercase();
    HINTS.iter().any(|h| low.contains(h))
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
            "(({alias}.project_key != '' AND {alias}.project_key = ?{p_key}) OR ({alias}.project_key = '' AND LOWER({alias}.project) = LOWER(?{p_name})))"
        ));
        dyn_params.push(rusqlite::types::Value::from(sc.key.to_string()));
        dyn_params.push(rusqlite::types::Value::from(sc.name.to_string()));
    } else if !sc.key.is_empty() {
        let p_key = *param_idx;
        *param_idx += 1;
        conditions.push(format!("{alias}.project_key = ?{p_key}"));
        dyn_params.push(rusqlite::types::Value::from(sc.key.to_string()));
    } else if !sc.name.is_empty() {
        let p_name = *param_idx;
        *param_idx += 1;
        conditions.push(format!("LOWER({alias}.project) = LOWER(?{p_name})"));
        dyn_params.push(rusqlite::types::Value::from(sc.name.to_string()));
    }
}

pub(crate) fn query_nodes_fts_conn(
    conn: &Connection,
    agent_id: &str,
    fts_query: &str,
    scope: Option<ProjectScope<'_>>,
    limit: usize,
) -> Vec<NeuralNode> {
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
                n.windows_json, n.summary, n.events_json, n.updated_at
         FROM nodes_fts
         JOIN nodes n ON n.id = nodes_fts.node_id
         WHERE {}
         ORDER BY bm25(nodes_fts) ASC
         LIMIT ?{param_idx}",
        conditions.join(" AND ")
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return Vec::new();
    };
    stmt.query_map(rusqlite::params_from_iter(dyn_params), |row| {
        let win_str: String = row.get(8)?;
        let ev_str: String = row.get(10)?;
        Ok(NeuralNode {
            id: row.get(0)?,
            cat: row.get(1)?,
            name: row.get(2)?,
            project: row.get(3)?,
            project_key: row.get(4)?,
            status: row.get(5)?,
            mastery: row.get(6)?,
            tier: row.get(7)?,
            windows: serde_json::from_str(&win_str).unwrap_or_else(|_| default_windows()),
            summary: row.get(9)?,
            events: serde_json::from_str(&ev_str).unwrap_or_default(),
            updated_at: row.get(11)?,
        })
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

pub(crate) fn query_timeline_fts_conn(
    conn: &Connection,
    agent_id: &str,
    fts_query: &str,
    scope: Option<ProjectScope<'_>>,
    date_range: Option<&DateRange>,
    limit: usize,
) -> Vec<TimelineEvent> {
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
        let p_s = param_idx;
        let p_e = param_idx + 1;
        param_idx += 2;
        conditions.push(format!("t.ts >= ?{p_s}"));
        conditions.push(format!("t.ts <= ?{p_e}"));
        dyn_params.push(rusqlite::types::Value::from(dr.start.clone()));
        dyn_params.push(rusqlite::types::Value::from(dr.end.clone()));
    }
    dyn_params.push(rusqlite::types::Value::from(limit as i64));

    let sql = format!(
        "SELECT t.id, t.ts, t.agent_id, t.project, t.project_key, t.working_dir, t.session_id, t.turn_id, t.raw_file,
                t.user_goal, t.summary, t.ai_outcome, t.keywords_json, t.touched_files_json, t.distilled_by
         FROM timeline_fts
         JOIN timeline t ON t.id = timeline_fts.event_id
         WHERE {}
         ORDER BY bm25(timeline_fts) ASC
         LIMIT ?{param_idx}",
        conditions.join(" AND ")
    );
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return Vec::new();
    };
    stmt.query_map(rusqlite::params_from_iter(dyn_params), |row| {
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
    })
    .map(|rows| rows.flatten().collect())
    .unwrap_or_default()
}

pub(crate) fn query_raw_turns_fts_conn(
    conn: &Connection,
    agent_id: &str,
    fts_query: &str,
    scope: Option<ProjectScope<'_>>,
    date_range: Option<&DateRange>,
    limit: usize,
) -> Vec<RawTurn> {
    let mut conditions = vec![
        "r.agent_id = ?1".to_string(),
        "raw_turns_fts MATCH ?2".to_string(),
    ];
    let mut dyn_params: Vec<rusqlite::types::Value> = vec![
        rusqlite::types::Value::from(agent_id.to_string()),
        rusqlite::types::Value::from(fts_query.to_string()),
    ];
    let mut param_idx = 3usize;
    append_project_scope_sql(&mut conditions, &mut dyn_params, &mut param_idx, "r", scope);
    if let Some(dr) = date_range {
        let p_s = param_idx;
        let p_e = param_idx + 1;
        param_idx += 2;
        conditions.push(format!("r.ts >= ?{p_s}"));
        conditions.push(format!("r.ts <= ?{p_e}"));
        dyn_params.push(rusqlite::types::Value::from(dr.start.clone()));
        dyn_params.push(rusqlite::types::Value::from(dr.end.clone()));
    }
    dyn_params.push(rusqlite::types::Value::from(limit as i64));

    let sql = format!(
        "SELECT r.turn_id, r.ts, r.agent_id, r.project, r.project_key, r.working_dir, r.session_id,
                r.user_text, r.ai_text, r.touched_files_json
         FROM raw_turns_fts
         JOIN raw_turns r ON r.turn_id = raw_turns_fts.turn_id
         WHERE {}
         ORDER BY bm25(raw_turns_fts) ASC
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
