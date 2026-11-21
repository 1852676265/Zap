//! 文本切词（含驼峰/下划线拆分 + 东亚文字 + 虚词过滤）、双语词林扩展、精确路牌提取、查询意图分析与时间范围解析。

use std::collections::HashSet;
use std::sync::OnceLock;

use chrono::{Datelike, Duration as ChronoDuration, Local, NaiveDate};
use regex::Regex;
use zap_capabilities::codeintel::bilingual_nlp::{split_identifier, DynamicThesaurus};
use zap_capabilities::reminder::strip_injected_reminders_for_display;

use super::store::KnownProject;
use super::types::MemoryAnchor;

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

/// 解析后的结构化查询计划：把用户的自然语言拆成意图、精确路牌、核心实词、双语扩展词与辅助上下文词。
pub(crate) struct QueryPlan {
    pub(crate) date_range: Option<DateRange>,
    pub(crate) wants_progress: bool,
    pub(crate) wants_superseded: bool,
    pub(crate) wants_verbatim: bool,
    pub(crate) anchors: Vec<String>,
    pub(crate) core_terms: Vec<String>,
    pub(crate) expanded_terms: Vec<String>,
    pub(crate) context_terms: Vec<String>,
    pub(crate) fts_strict: Option<String>,
    pub(crate) fts_broad: Option<String>,
}

/// 进程级双语同义词林（加载内置默认词典 + `~/.zap/thesaurus/*.txt`）。
pub(crate) fn shared_thesaurus() -> &'static DynamicThesaurus {
    static THESAURUS: OnceLock<DynamicThesaurus> = OnceLock::new();
    THESAURUS.get_or_init(|| {
        let mut dt = DynamicThesaurus::new();
        if !cfg!(test) {
            let dir = zap_config::Config::config_dir().join("thesaurus");
            dt.load_from_dir(&dir);
        }
        dt
    })
}

/// 判断字符是否属于 CJK 汉字（含扩展区/兼容区）、日文假名或韩文音节。
fn is_cjk_or_kana_hangul(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2EBEF
            | 0x3040..=0x309F
            | 0x30A0..=0x30FF
            | 0xAC00..=0xD7AF
    )
}

/// 中文高频虚词/助词单字：用于切分口语长句时滤除跨词边界的垃圾 2-gram（如“那怎”“么改”“改呢”）。
fn is_cjk_particle(ch: char) -> bool {
    matches!(
        ch,
        '的' | '了'
            | '在'
            | '是'
            | '我'
            | '你'
            | '他'
            | '她'
            | '它'
            | '这'
            | '那'
            | '呢'
            | '吗'
            | '吧'
            | '啊'
            | '哦'
            | '嗯'
            | '就'
            | '都'
            | '很'
            | '也'
            | '还'
            | '把'
            | '被'
            | '给'
            | '让'
            | '向'
            | '往'
            | '跟'
            | '和'
            | '与'
            | '或'
            | '但'
            | '却'
            | '而'
            | '并'
            | '且'
            | '之'
            | '于'
            | '其'
            | '从'
            | '到'
            | '对'
            | '为'
            | '以'
            | '着'
            | '过'
            | '得'
            | '地'
            | '么'
            | '啥'
    )
}

/// 兼容旧接口：构建单列 FTS 文本（现委托给去重无重复膨胀的分列索引文本构造器）。
#[cfg(test)]
pub(crate) fn build_fts_search_text(parts: &[&str]) -> String {
    build_fts_column_text(parts)
}

/// 生成供分列 FTS5 检索的紧凑去重词元串：
/// - 保留英文原词 + 驼峰/下划线/点号拆分子词（`resolveAgent` <-> `resolve_agent` 互通）；
/// - 东亚文字片段补充 2-gram / 3-gram 及短整词，不再重复拼接原始全文，避免扭曲 BM25 长度惩罚。
pub(crate) fn build_fts_column_text(parts: &[&str]) -> String {
    let joined = parts
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if joined.is_empty() {
        return String::new();
    }

    let mut tokens: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for word in joined
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.' && c != '/')
        .filter(|w| w.chars().count() >= 2)
    {
        let clean = word.trim_matches(|c: char| c == '.' || c == '/');
        if clean.chars().count() < 2 {
            continue;
        }
        let low = clean.to_lowercase();
        if seen.insert(low.clone()) {
            tokens.push(low.clone());
        }
        for sub in split_identifier(clean) {
            let sl = sub.to_lowercase();
            if sl.chars().count() >= 2 && seen.insert(sl.clone()) {
                tokens.push(sl);
            }
        }
    }

    let mut cjk_run = Vec::new();
    let flush_cjk = |run: &mut Vec<char>, tokens: &mut Vec<String>, seen: &mut HashSet<String>| {
        if run.len() >= 2 {
            if run.len() <= 4 {
                let whole: String = run.iter().collect();
                if seen.insert(whole.clone()) {
                    tokens.push(whole);
                }
            }
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

    tokens.join(" ")
}

pub(crate) fn is_fts_stop_word(term: &str) -> bool {
    const STOPS: &[&str] = &[
        "这个",
        "那个",
        "这些",
        "那些",
        "怎么",
        "如何",
        "怎样",
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
        "改下",
        "处理",
        "解决",
        "办法",
        "方案",
        "建议",
        "优化",
        "功能",
        "效果",
        "情况",
        "感觉",
        "可能",
        "应该",
        "需要",
        "希望",
        "最好",
        "然后",
        "不过",
        "但是",
        "如果",
        "因为",
        "所以",
        "继续",
        "排查",
        "分析",
        "确认",
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
        "告诉",
        "觉得",
        "其实",
        "是不是",
        "有没有",
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
        "fix",
        "change",
        "update",
        "issue",
        "problem",
        "plan",
    ];
    STOPS.contains(&term)
}

fn anchor_regexes() -> (
    &'static Regex,
    &'static Regex,
    &'static Regex,
    &'static Regex,
) {
    static FILE_RE: OnceLock<Regex> = OnceLock::new();
    static ERR_RE: OnceLock<Regex> = OnceLock::new();
    static ENDPOINT_RE: OnceLock<Regex> = OnceLock::new();
    static TEAM_RE: OnceLock<Regex> = OnceLock::new();
    (
        FILE_RE.get_or_init(|| {
            Regex::new(
                r"(?i)([A-Za-z0-9_.\-/\\]+\.(?:rs|ts|tsx|js|jsx|vue|java|py|go|sql|toml|yaml|yml|json|md|css|scss|html|xml|sh|ps1|db))\b",
            )
            .unwrap()
        }),
        ERR_RE.get_or_init(|| {
            Regex::new(
                r"\b(?:D\d{4}|E\d{4}|C\d{4}|LNK\d{4}|ORA-\d{4,5}|ERR_[A-Z0-9_]{2,}|SQLSTATE[_\s]?\d{4,5})\b",
            )
            .unwrap()
        }),
        ENDPOINT_RE.get_or_init(|| {
            Regex::new(r#"(?:^|[\s'"(`])(/(?:api|v[1-9])/[A-Za-z0-9_\-/:.]{2,64})"#).unwrap()
        }),
        TEAM_RE.get_or_init(|| Regex::new(r"\b(trun_[a-zA-Z0-9_\-]{4,40})\b").unwrap()),
    )
}

fn is_camel_or_pascal_symbol(word: &str) -> bool {
    if word.len() < 6 || word.len() > 64 {
        return false;
    }
    if !word.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return false;
    }
    let has_upper = word.bytes().any(|b| b.is_ascii_uppercase());
    let has_lower = word.bytes().any(|b| b.is_ascii_lowercase());
    if !(has_upper && has_lower) {
        return false;
    }
    word.as_bytes()
        .windows(2)
        .any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase())
}

fn is_snake_or_table_symbol(word: &str) -> bool {
    if word.len() < 5 || word.len() > 64 || !word.contains('_') {
        return false;
    }
    word.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && word.bytes().any(|b| b.is_ascii_alphabetic())
}

fn extract_raw_anchor_pairs(
    texts: &[&str],
    explicit_files: &[String],
    team_run_id: &str,
) -> Vec<(String, String)> {
    let (file_re, err_re, ep_re, team_re) = anchor_regexes();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();

    let mut push = |atype: &str, val: &str| {
        let trimmed = val.trim();
        let norm = trimmed.to_lowercase();
        if norm.len() < 3 || norm.len() > 96 {
            return;
        }
        if out.len() < 28 && seen.insert((atype.to_string(), norm)) {
            out.push((atype.to_string(), trimmed.to_string()));
        }
    };

    if !team_run_id.trim().is_empty() {
        push("team_run", team_run_id.trim());
    }

    for f in explicit_files {
        let norm_path = f.trim().replace('\\', "/");
        if norm_path.is_empty() {
            continue;
        }
        if let Some(base) = norm_path.rsplit('/').next() {
            if base.contains('.') {
                push("file", base);
            }
        }
        let segs: Vec<&str> = norm_path.split('/').filter(|s| !s.is_empty()).collect();
        if segs.len() >= 2 {
            let tail2 = format!("{}/{}", segs[segs.len() - 2], segs[segs.len() - 1]);
            push("file", &tail2);
        }
    }

    for text in texts {
        if text.trim().is_empty() {
            continue;
        }
        for cap in file_re.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let norm_path = m.as_str().replace('\\', "/");
                if let Some(base) = norm_path.rsplit('/').next() {
                    push("file", base);
                }
                let segs: Vec<&str> = norm_path.split('/').filter(|s| !s.is_empty()).collect();
                if segs.len() >= 2 {
                    let tail2 = format!("{}/{}", segs[segs.len() - 2], segs[segs.len() - 1]);
                    push("file", &tail2);
                }
            }
        }
        for m in err_re.find_iter(text) {
            push("error_code", m.as_str());
        }
        for cap in ep_re.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                push("endpoint", m.as_str());
            }
        }
        for cap in team_re.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                push("team_run", m.as_str());
            }
        }
        for tok in text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            if is_camel_or_pascal_symbol(tok) {
                push("symbol", tok);
            } else if is_snake_or_table_symbol(tok) {
                let kind = if tok.chars().all(|c| !c.is_ascii_lowercase()) {
                    "table"
                } else {
                    "symbol"
                };
                push(kind, tok);
            }
        }
    }

    out
}

/// 写入端路牌提取：为单条记忆抽取精确路牌（文件、报错码、接口、表名、符号、Team 单号）。
pub(crate) fn extract_storage_anchors(
    agent_id: &str,
    project_key: &str,
    target_kind: &str,
    target_id: &str,
    texts: &[&str],
    explicit_files: &[String],
    team_run_id: &str,
) -> Vec<MemoryAnchor> {
    extract_raw_anchor_pairs(texts, explicit_files, team_run_id)
        .into_iter()
        .map(|(anchor_type, anchor_value)| MemoryAnchor {
            agent_id: agent_id.to_string(),
            project_key: project_key.to_string(),
            target_kind: target_kind.to_string(),
            target_id: target_id.to_string(),
            anchor_type,
            anchor_norm: anchor_value.to_lowercase(),
            anchor_value,
        })
        .collect()
}

/// 查询端路牌提取：提取当前提问中出现的规范化路牌，供 `memory_anchors` B-Tree 秒查。
pub(crate) fn extract_query_anchors(query: &str) -> Vec<String> {
    let mut norms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (_, val) in extract_raw_anchor_pairs(&[query], &[], "") {
        let n = val.to_lowercase();
        if seen.insert(n.clone()) {
            norms.push(n);
        }
    }
    norms
}

/// 从单段文本提取按「区分度」排序的核心实词：
/// 优先级：技术标识符/文件名/报错码 > 3-4 字中文实词 > 纯净 2 字中文实词（剔除虚词污染的跨界 2-gram）。
fn extract_salient_terms(text: &str) -> Vec<(String, u8)> {
    let mut scored: Vec<(String, u8)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for word in text
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.' && c != '/')
        .filter(|w| w.chars().count() >= 2)
    {
        let clean = word.trim_matches(|c: char| c == '.' || c == '/');
        let low = clean.to_lowercase();
        if low.len() < 2 || low.len() > 48 || is_fts_stop_word(&low) {
            continue;
        }
        let has_ascii = low.chars().any(|c| c.is_ascii_alphanumeric());
        if has_ascii {
            let prio = if low.contains('.')
                || low.contains('_')
                || low.contains('-')
                || is_camel_or_pascal_symbol(clean)
                || low.chars().any(|c| c.is_ascii_digit())
            {
                4
            } else {
                3
            };
            if seen.insert(low.clone()) {
                scored.push((low.clone(), prio));
            }
            for sub in split_identifier(clean) {
                let sl = sub.to_lowercase();
                if sl.len() >= 3 && !is_fts_stop_word(&sl) && seen.insert(sl.clone()) {
                    scored.push((sl, 2));
                }
            }
        }
    }

    let mut cjk_seg: Vec<char> = Vec::new();
    let flush_seg =
        |seg: &mut Vec<char>, scored: &mut Vec<(String, u8)>, seen: &mut HashSet<String>| {
            if seg.len() >= 2 {
                if seg.len() <= 4 {
                    let whole: String = seg.iter().collect();
                    if !is_fts_stop_word(&whole) && seen.insert(whole.clone()) {
                        let prio = if seg.len() >= 3 { 3 } else { 2 };
                        scored.push((whole, prio));
                    }
                }
                if seg.len() >= 3 {
                    for win in seg.windows(3) {
                        let g: String = win.iter().collect();
                        if !is_fts_stop_word(&g) && seen.insert(g.clone()) {
                            scored.push((g, 2));
                        }
                    }
                }
                for win in seg.windows(2) {
                    let g: String = win.iter().collect();
                    if !is_fts_stop_word(&g) && seen.insert(g.clone()) {
                        scored.push((g, 1));
                    }
                }
            }
            seg.clear();
        };

    for ch in text.chars() {
        if is_cjk_or_kana_hangul(ch) && !is_cjk_particle(ch) {
            cjk_seg.push(ch);
        } else {
            flush_seg(&mut cjk_seg, &mut scored, &mut seen);
        }
    }
    flush_seg(&mut cjk_seg, &mut scored, &mut seen);

    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())));
    scored
}

fn expand_via_thesaurus(core_terms: &[String], raw_query: &str) -> Vec<String> {
    let dt = shared_thesaurus();
    let parsed = zap_capabilities::codeintel::bilingual_nlp::parse_bilingual_query_with_thesaurus(
        raw_query, dt,
    );
    let core_set: HashSet<&str> = core_terms.iter().map(String::as_str).collect();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for term in parsed.expanded_terms {
        let low = term.trim().to_lowercase();
        if low.len() >= 2
            && !is_fts_stop_word(&low)
            && !core_set.contains(low.as_str())
            && seen.insert(low.clone())
        {
            out.push(low);
            if out.len() >= 8 {
                break;
            }
        }
    }
    out
}

fn wants_progress_intent(query: &str) -> bool {
    let low = query.to_lowercase();
    const HINTS: &[&str] = &[
        "进度",
        "改好没",
        "做完没",
        "完成了吗",
        "做好没",
        "还剩",
        "剩下",
        "没做完",
        "未完成",
        "学到哪",
        "明天学",
        "明天该学",
        "下一步",
        "学习计划",
        "待办",
        "哪些需求",
        "这个需求",
        "progress",
        "remaining",
        "unfinished",
        "todo",
        "next step",
        "study plan",
    ];
    HINTS.iter().any(|h| low.contains(h))
}

fn wants_superseded_intent(query: &str) -> bool {
    let low = query.to_lowercase();
    const HINTS: &[&str] = &[
        "为什么放弃",
        "之前放弃",
        "之前弃用",
        "以前用的",
        "旧方案",
        "作废",
        "历史版本",
        "原来为什么",
        "superseded",
        "deprecated",
        "old decision",
    ];
    HINTS.iter().any(|h| low.contains(h))
}

fn quote_fts_term(term: &str) -> Option<String> {
    let escaped = term.replace('"', "").trim().to_string();
    if escaped.len() < 2 {
        None
    } else {
        Some(format!("\"{escaped}\""))
    }
}

/// 构造完整结构化查询计划（支持短句追问借用上 1~2 轮用户对话主题词）。
pub(crate) fn analyze_query(query: &str, prior_user_turns: &[String]) -> QueryPlan {
    let date_range = parse_temporal_range(query);
    let wants_progress = wants_progress_intent(query);
    let wants_superseded = wants_superseded_intent(query);
    let wants_verbatim = should_inject_l3_verbatim(query);
    let anchors = extract_query_anchors(query);

    let salient = extract_salient_terms(query);
    let core_terms: Vec<String> = salient.iter().take(12).map(|(t, _)| t.clone()).collect();
    let expanded_terms = expand_via_thesaurus(&core_terms, query);

    let mut context_terms = Vec::new();
    if core_terms.len() < 2 && anchors.is_empty() && !prior_user_turns.is_empty() {
        let core_set: HashSet<&str> = core_terms.iter().map(String::as_str).collect();
        let mut seen_ctx = HashSet::new();
        for prev in prior_user_turns.iter().rev().take(2) {
            for (term, prio) in extract_salient_terms(prev) {
                if prio >= 2 && !core_set.contains(term.as_str()) && seen_ctx.insert(term.clone()) {
                    context_terms.push(term);
                    if context_terms.len() >= 4 {
                        break;
                    }
                }
            }
            if context_terms.len() >= 4 {
                break;
            }
        }
    }

    let high_prio: Vec<String> = salient
        .iter()
        .filter(|(_, p)| *p >= 3)
        .filter_map(|(t, _)| quote_fts_term(t))
        .take(4)
        .collect();
    let fts_strict = if high_prio.len() >= 2 {
        Some(high_prio.join(" AND "))
    } else {
        let top2: Vec<String> = core_terms
            .iter()
            .filter_map(|t| quote_fts_term(t))
            .take(2)
            .collect();
        if top2.len() >= 2 {
            Some(top2.join(" AND "))
        } else {
            None
        }
    };

    let mut broad_vec: Vec<String> = Vec::new();
    let mut seen_b = HashSet::new();
    for term in core_terms
        .iter()
        .chain(expanded_terms.iter())
        .chain(context_terms.iter())
    {
        if let Some(q) = quote_fts_term(term) {
            if seen_b.insert(q.clone()) {
                broad_vec.push(q);
                if broad_vec.len() >= 22 {
                    break;
                }
            }
        }
    }
    let fts_broad = (!broad_vec.is_empty()).then(|| broad_vec.join(" OR "));

    QueryPlan {
        date_range,
        wants_progress,
        wants_superseded,
        wants_verbatim,
        anchors,
        core_terms,
        expanded_terms,
        context_terms,
        fts_strict,
        fts_broad,
    }
}

/// 兼容旧接口：生成 `(strict_and_query, broad_or_query)`。
#[cfg(test)]
pub(crate) fn build_fts_queries(query: &str) -> (Option<String>, Option<String>) {
    let plan = analyze_query(query, &[]);
    (plan.fts_strict, plan.fts_broad)
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

/// 词边界感知的跨项目点名检测，避免 `api` / `web` 等短词或普通单词子串误触发跨项目注入。
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
