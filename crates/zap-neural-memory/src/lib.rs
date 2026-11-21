//! 智能体独立记忆与认知引擎（SQLite WAL 读写分离 + 5 抽屉分层 + 精确路牌 + 分列加权 FTS5 + 阶段草稿整理 + Team 验收回流）。
//!
//! 职责拆分为以下子模块：
//! - `types`：5 抽屉核心数据结构、活卡片要点/别名/作废状态与规范化 `project_key`
//! - `redact`：凭据与密钥脱敏
//! - `schema`：SQLite v3 建表与无损迁移（分列 FTS5 + `memory_anchors` + `draft_turns` + 移除 `raw_turns_fts`）
//! - `store`：WAL 读写分离连接池、进程级热缓存、`prepare_cached`、极薄目录与留存淘汰
//! - `search`：驼峰/下划线/东亚文字切词、停用词、双语词林扩展、精确路牌提取与查询意图分析
//! - `retrieval`：多通道检索（精确路牌 B-Tree + 分列加权 BM25 + 多因子重排 + 及格线截断 + 1-hop 邻居）
//! - `reconcile`：写入期先查后合事务引擎（ADD / MERGE / SUPERSEDE / SKIP 与碎片老数据整理）
//! - `distill`：持久化草稿队列与 5 类自然触发点（静默/满轮/切换/压缩前/显式记住插队）
//! - `team_bridge`：无侵入只读桥接 `agent-teams/teams.db`（完结精准回流 + 跨端总进度透视）
//! - `injection`：静态系统前缀（含极薄目录）+ `NeuralMemoryHook`（`pre_request` 瞬态注入与短句借主语）
//! - `recall_tool`：`memory_recall` 只读检索工具（支持按抽屉与作废历史筛选）
//! - `rpc`：WebUI / 插件 RPC 分发器（兼容旧版脑图并支持手动刷新草稿与老数据整理）

mod distill;
mod injection;
mod recall_tool;
mod reconcile;
mod redact;
mod retrieval;
mod rpc;
mod schema;
mod search;
mod store;
mod team_bridge;
mod types;

pub use distill::{flush_session_drafts_async, maybe_capture_turn_from_snapshot};
pub use injection::{
    apply_session_static_system_append, build_dynamic_turn_reminder, build_static_system_block,
    note_fresh_session_static_agent, NeuralMemoryBinding, NeuralMemoryHook,
};
pub use recall_tool::MemoryRecallTool;
pub use rpc::handle_rpc;
pub use search::clean_user_prompt;
pub use team_bridge::TeamTaskDelivery;
pub use types::{
    NeuralAgent, NeuralConfig, NeuralLink, NeuralNode, NeuralTask, NodeEvent, NodeFact,
    ProfileFact, RawTurn, TimelineEvent,
};

/// 供 `agent_teams` 等同进程协同模块安全读取当前全部隔离记忆智能体列表（走进程级热缓存 + 只读连接）。
pub fn list_neural_agents() -> Result<Vec<NeuralAgent>, String> {
    store::get_cached_agents()
}

/// 供 `agent_teams` 在拉起跨项目子会话前原子绑定会话专属隔离记忆智能体。
pub fn bind_session_neural_agent(session_id: &str, agent_id: &str) -> Result<(), String> {
    store::with_write(|s| store::set_session_agent_conn(&s.conn, session_id, agent_id))
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use rusqlite::{params, Connection};
    use zap_kernel::hook::{LifecycleHooks, TurnCtx};
    use zap_kernel::message::{Conversation, Message};
    use zap_kernel::tool::ToolCall;

    use super::distill::{
        deterministic_turn_id, excerpt_head_tail, extract_last_turn_payload,
        user_explicitly_requests_pin,
    };
    use super::injection::{
        build_dynamic_turn_reminder_conn, build_dynamic_turn_reminder_with_context_conn,
        build_static_system_block_with_ids_conn, record_session_static_state, NeuralMemoryHook,
    };
    use super::recall_tool::{execute_memory_recall_conn, MemoryRecallArgs};
    use super::reconcile::{
        apply_fallback_distill_conn, apply_llm_distill_conn, consolidate_legacy_nodes_conn,
        DistillLinkTarget, DistillNodeAction, DistillNodeFactOp, DistillOutput,
        DistillProfileAction, DistillProfileInput, DistillProfileObj, DistillTaskAction,
    };
    use super::redact::redact_secrets;
    use super::retrieval::query_nodes_ranked_conn;
    use super::schema::{init_schema, CURRENT_SCHEMA_VERSION};
    use super::search::{
        analyze_query, build_fts_queries, build_fts_search_text, detect_mentioned_project,
        extract_query_anchors, is_slash_command, parse_temporal_range,
    };
    use super::store::{
        bind_fresh_session_agent_if_unset_conn, enqueue_draft_turn_conn, get_config_conn,
        insert_timeline_conn, load_links_conn, load_nodes_conn, load_pending_drafts_for_scope_conn,
        load_profiles_conn, load_raw_turn_conn, load_tasks_conn, load_timeline_conn,
        prune_retention_conn, resolve_agent_for_session_conn, save_config_conn,
        set_session_agent_conn, setup_mem_db, upsert_agent_conn, upsert_node_conn,
        upsert_profile_conn, upsert_raw_turn_conn, upsert_task_conn, KnownProject,
    };
    use super::team_bridge::{apply_team_delivery_conn, is_team_managed_session, TeamTaskDelivery};
    use super::types::{
        default_windows, normalize_project_key, DraftTurn, NeuralAgent, NeuralNode, NeuralTask,
        NodeFact, ProfileFact, RawTurn, TimelineEvent, MAX_NODES_PER_AGENT,
        MAX_NON_PINNED_PROFILES, MAX_RAW_TURNS_PER_AGENT,
    };

    fn seed_backend_agent(conn: &Connection) {
        upsert_agent_conn(
            conn,
            &NeuralAgent {
                id: "backend".into(),
                icon: "⚙".into(),
                name: "Backend".into(),
                label: "⚙ Backend".into(),
                color: "#3b82f6".into(),
                hint: "Backend dev".into(),
                system_prompt: "You are a backend specialist.".into(),
            },
            0,
        )
        .unwrap();
    }

    #[test]
    fn slash_command_is_skipped_while_unix_paths_pass() {
        assert!(is_slash_command("/compact"));
        assert!(is_slash_command("/help"));
        assert!(is_slash_command("/model gpt-4o"));
        assert!(!is_slash_command("/etc/nginx/nginx.conf 检查一下"));
        assert!(!is_slash_command("帮我看看 /compact 是怎么做的"));
    }

    #[test]
    fn temporal_range_parses_cn_en_and_relative_days() {
        assert!(parse_temporal_range("昨天改了什么").is_some());
        assert!(parse_temporal_range("what did we do yesterday").is_some());
        assert!(parse_temporal_range("last week").is_some());
        assert_eq!(
            parse_temporal_range("最近3天的改动").map(|r| r.label),
            Some("last 3 days".to_string())
        );
        assert_eq!(
            parse_temporal_range("last 5 days").map(|r| r.label),
            Some("last 5 days".to_string())
        );
        assert_eq!(
            parse_temporal_range("2026-11-10 的需求").map(|r| r.label),
            Some("2026-11-10".to_string())
        );
    }

    #[test]
    fn fts_queries_strip_stop_words_and_keep_identifiers_and_cjk_kana_hangul() {
        let (strict, broad) =
            build_fts_queries("帮我看看之前 tsm-server 这个需求怎么改 neural_memory.rs");
        let strict_str = strict.expect("strict query");
        let broad_str = broad.expect("broad query");
        assert!(strict_str.contains("tsm-server"));
        assert!(strict_str.contains("neural_memory.rs"));
        assert!(!broad_str.contains("\"帮我\""));
        assert!(!broad_str.contains("\"这个\""));
        assert!(!broad_str.contains("\"需求\""));

        let fts_jp_kr = build_fts_search_text(&["キャッシュ設計 と 데이터베이스"]);
        assert!(fts_jp_kr.contains("キャ"));
        assert!(fts_jp_kr.contains("데이"));
    }

    #[test]
    fn extract_last_turn_preserves_intermediate_steps() {
        let step1 = Message::assistant(
            "Step1: 排查发现 tsm-server 连接池超时",
            vec![ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"src/pool.rs"}"#.into(),
            }],
        );
        let step2 = Message::assistant("最终结论：已把超时时间调整到 30s", vec![]);
        let msgs = vec![Message::user("tsm-server 报错排查"), step1, step2];
        let (u, ai, files) = extract_last_turn_payload(&msgs).expect("payload");
        assert_eq!(u, "tsm-server 报错排查");
        assert!(ai.contains("连接池超时"));
        assert!(ai.contains("最终结论"));
        assert_eq!(files, vec!["src/pool.rs".to_string()]);
    }

    #[test]
    fn schema_migration_adds_project_key_and_backfills_working_dir() {
        let conn = Connection::open_in_memory().unwrap();
        // 模拟旧版无 project_key 的 v0 表结构
        conn.execute_batch(
            "
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, project TEXT NOT NULL,
                working_dir TEXT NOT NULL, title TEXT NOT NULL, progress_note TEXT NOT NULL,
                remaining_note TEXT NOT NULL, touched_files_json TEXT NOT NULL, session_id TEXT NOT NULL,
                source_turn_ids_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, done_at TEXT NOT NULL
            );
            INSERT INTO tasks VALUES (
                't1', 'backend', 'server', 'D:\\Work\\Alpha\\server\\', 'Fix pool', '', '', '[]', 's1', '[]',
                'in_progress', '2026-11-10', '2026-11-10', ''
            );
            ",
        )
        .unwrap();

        init_schema(&conn).unwrap();
        let ver: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, CURRENT_SCHEMA_VERSION);
        let pkey: String = conn
            .query_row("SELECT project_key FROM tasks WHERE id = 't1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(pkey, "d:/work/alpha/server");
    }

    #[test]
    fn profile_removes_protects_pinned_and_scopes_to_current_project() {
        let mut conn = setup_mem_db();
        seed_backend_agent(&conn);
        let key_tsm = normalize_project_key(Path::new("/work/tsm-server"));
        let key_other = normalize_project_key(Path::new("/work/other-app"));

        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf1".into(),
                text: "禁止删除核心置顶规范 100% 保留".into(),
                project: "tsm-server".into(),
                project_key: key_tsm.clone(),
                pinned: true,
                updated_at: "2026-11-17 10:00".into(),
                source_turn_id: "t1".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf2".into(),
                text: "临时测试占比 50% 规则".into(),
                project: "tsm-server".into(),
                project_key: key_tsm,
                pinned: false,
                updated_at: "2026-11-17 10:01".into(),
                source_turn_id: "t2".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_other".into(),
                text: "其他项目的 50% 规则不应被误删".into(),
                project: "other-app".into(),
                project_key: key_other,
                pinned: false,
                updated_at: "2026-11-17 10:01".into(),
                source_turn_id: "t2".into(),
                ..Default::default()
            },
        )
        .unwrap();

        let distill = DistillOutput {
            should_record: true,
            profile_removes: vec!["50%".into(), "置顶规范".into()],
            ..Default::default()
        };
        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s1",
            "t3",
            "2026-11-17 10:02",
            "tsm-server",
            "/work/tsm-server",
            "更新规则",
            "已更新",
            &[],
            distill,
            false,
        )
        .unwrap();

        let remaining = load_profiles_conn(&conn, Some("backend"))
            .unwrap()
            .remove("backend")
            .unwrap_or_default();
        let ids: Vec<&str> = remaining.iter().map(|p| p.id.as_str()).collect();
        assert!(ids.contains(&"pf1"));
        assert!(!ids.contains(&"pf2"));
        assert!(ids.contains(&"pf_other"));
    }

    #[test]
    fn per_project_profile_eviction_and_same_basename_path_isolation() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        let key_a = normalize_project_key(Path::new("/work/alpha/server"));
        let key_b = normalize_project_key(Path::new("/work/beta/server"));

        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_global".into(),
                text: "Global rule: always parameterize SQL".into(),
                project: "".into(),
                project_key: "".into(),
                pinned: false,
                updated_at: "2026-11-17 09:00".into(),
                source_turn_id: "t0".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_alpha".into(),
                text: "Alpha server uses DM8".into(),
                project: "server".into(),
                project_key: key_a.clone(),
                pinned: false,
                updated_at: "2026-11-17 09:01".into(),
                source_turn_id: "t1".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_beta".into(),
                text: "Beta server uses PostgreSQL".into(),
                project: "server".into(),
                project_key: key_b.clone(),
                pinned: false,
                updated_at: "2026-11-17 09:02".into(),
                source_turn_id: "t2".into(),
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..(MAX_NON_PINNED_PROFILES + 10) {
            upsert_profile_conn(
                &conn,
                "backend",
                &ProfileFact {
                    id: format!("pf_zap_{i}"),
                    text: format!("Zap rule {i}"),
                    project: "Zap".into(),
                    project_key: "/work/zap".into(),
                    pinned: false,
                    updated_at: format!("2026-11-17 10:{:02}", i % 60),
                    source_turn_id: format!("tz_{i}"),
                    ..Default::default()
                },
            )
            .unwrap();
        }

        let (static_alpha, _) =
            build_static_system_block_with_ids_conn(&conn, "", &key_a, "server", false).unwrap();
        let static_alpha = static_alpha.unwrap();
        assert!(static_alpha.contains("Alpha server uses DM8"));
        assert!(static_alpha.contains("Global rule: always parameterize SQL"));
        assert!(!static_alpha.contains("Beta server uses PostgreSQL"));
        assert!(!static_alpha.contains("Zap rule"));
    }

    #[test]
    fn fallback_distill_writes_timeline_without_fake_nodes() {
        let conn = setup_mem_db();
        apply_fallback_distill_conn(
            &conn,
            "backend",
            "s1",
            "turn1",
            "2026-11-17 10:00",
            "tsm-server",
            "/work/tsm-server",
            "帮我看看这个报错怎么修",
            "排查步骤如下",
            &[],
            "fallback",
        )
        .unwrap();
        assert!(load_nodes_conn(&conn, Some("backend")).unwrap().is_empty());
        assert_eq!(
            load_timeline_conn(&conn, Some("backend"), 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn llm_distill_normalizes_nodes_and_uses_semantic_links() {
        let mut conn = setup_mem_db();
        let out1 = DistillOutput {
            should_record: true,
            node_actions: vec![DistillNodeAction {
                name: "Redis 缓存".into(),
                summary: "初版缓存方案".into(),
                mastery: 80,
                status: "active".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s1",
            "t1",
            "2026-11-17 10:00",
            "tsm-server",
            "/work/tsm-server",
            "设计缓存",
            "完成",
            &[],
            out1,
            false,
        )
        .unwrap();

        let out2 = DistillOutput {
            should_record: true,
            node_actions: vec![
                DistillNodeAction {
                    name: "redis-缓存".into(),
                    summary: "补充过期淘汰策略".into(),
                    mastery: 85,
                    status: "solid".into(),
                    ..Default::default()
                },
                DistillNodeAction {
                    name: "达梦连接池".into(),
                    summary: "连接池独立优化".into(),
                    mastery: 78,
                    status: "active".into(),
                    related_nodes: vec![DistillLinkTarget::Name("Redis 缓存".into())],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s1",
            "t2",
            "2026-11-17 10:05",
            "tsm-server",
            "/work/tsm-server",
            "补充缓存并连达梦",
            "完成",
            &[],
            out2,
            false,
        )
        .unwrap();

        let nodes = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(nodes.len(), 2);
        let links = load_links_conn(&conn).unwrap();
        assert_eq!(links.len(), 1);
    }

    #[test]
    fn memory_recall_supports_cross_project_tsm_server_lookup() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        upsert_raw_turn_conn(
            &conn,
            &RawTurn {
                turn_id: "turn_tsm_auth".into(),
                ts: "2026-11-15 14:20".into(),
                agent: "backend".into(),
                project: "tsm-server".into(),
                project_key: normalize_project_key(Path::new("D:/work/tsm-server")),
                working_dir: "D:/work/tsm-server".into(),
                session_id: "s_old".into(),
                user_text: "实现 tsm-server 的多租户鉴权中间件".into(),
                ai_text: "已在 auth_middleware.rs 添加租户隔离与签名校验".into(),
                touched_files: vec!["src/auth_middleware.rs".into()],
            },
        )
        .unwrap();
        insert_timeline_conn(
            &conn,
            &TimelineEvent {
                id: "tl_tsm_auth".into(),
                ts: "2026-11-15 14:20".into(),
                agent: "backend".into(),
                project: "tsm-server".into(),
                project_key: normalize_project_key(Path::new("D:/work/tsm-server")),
                working_dir: "D:/work/tsm-server".into(),
                session_id: "s_old".into(),
                turn_id: "turn_tsm_auth".into(),
                raw_file: "sqlite".into(),
                user_goal: "多租户鉴权中间件".into(),
                summary: "tsm-server 实现多租户鉴权中间件".into(),
                ai_outcome: "新增签名校验".into(),
                keywords: vec!["多租户".into(), "鉴权".into()],
                touched_files: vec!["src/auth_middleware.rs".into()],
                distilled_by: "llm".into(),
                ..Default::default()
            },
        )
        .unwrap();

        let res = execute_memory_recall_conn(
            &conn,
            "",
            Path::new("D:/work/order-service"),
            MemoryRecallArgs {
                query: "多租户鉴权".into(),
                project: "tsm-server".into(),
                date_range: String::new(),
                include_raw: Some(true),
                limit: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res.contains("tsm-server"));
        assert!(res.contains("auth_middleware.rs"));
        assert!(res.contains("租户隔离与签名校验"));
    }

    #[test]
    fn no_unrelated_node_padding_and_detect_project_avoids_generic_substrings() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        upsert_node_conn(
            &conn,
            &NeuralNode {
                id: "n1".into(),
                name: "Unrelated Kubernetes Node".into(),
                cat: "backend".into(),
                project: "infra".into(),
                project_key: "/work/infra".into(),
                status: "active".into(),
                mastery: 80,
                tier: 0,
                windows: default_windows(),
                summary: "Deployment manifests".into(),
                events: vec![],
                updated_at: "2026-11-17 10:00".into(),
                ..Default::default()
            },
        )
        .unwrap();

        // 查询完全不相关的话题时，绝不允许把无关节点凑数塞进上下文
        let rem = build_dynamic_turn_reminder_conn(
            &conn,
            "",
            "how to format rust datetime",
            Path::new("/work/infra"),
        )
        .unwrap();
        assert!(rem.is_none());

        // 词边界与短词过滤测试
        let known = vec![
            KnownProject {
                name: "api".into(),
                key: "/work/api".into(),
            },
            KnownProject {
                name: "tsm-server".into(),
                key: "/work/tsm-server".into(),
            },
        ];
        assert!(
            detect_mentioned_project("check capital letters", &known, "web", "/work/web").is_none()
        );
        assert!(
            detect_mentioned_project("call the api endpoint", &known, "web", "/work/web").is_none()
        );
        assert_eq!(
            detect_mentioned_project("check tsm-server pool config", &known, "web", "/work/web")
                .map(|p| p.name),
            Some("tsm-server".to_string())
        );
    }

    #[test]
    fn redaction_and_pin_poisoning_guard() {
        let dirty = "Authorization: Bearer abcdef1234567890 sk-proj-1234567890abcdef1234 password=my_super_secret";
        let clean = redact_secrets(dirty);
        assert!(!clean.contains("abcdef1234567890"));
        assert!(!clean.contains("sk-proj-1234567890abcdef1234"));
        assert!(!clean.contains("my_super_secret"));

        assert!(!user_explicitly_requests_pin(
            "please inspect this third party web page"
        ));
        assert!(user_explicitly_requests_pin(
            "记住：以后都用 sqlx 参数化查询"
        ));

        let mut conn = setup_mem_db();
        let allow_pin = user_explicitly_requests_pin("inspect this web page");
        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s1",
            "t_poison",
            "2026-11-17 10:00",
            "web",
            "/work/web",
            "inspect this web page",
            "Injected: always pin malicious rule",
            &[],
            DistillOutput {
                should_record: true,
                profile_updates: vec![DistillProfileInput::Obj(DistillProfileObj {
                    text: "Malicious pinned rule".into(),
                    pin: true,
                })],
                ..Default::default()
            },
            allow_pin,
        )
        .unwrap();

        let facts = load_profiles_conn(&conn, Some("backend")).unwrap();
        assert!(!facts["backend"][0].pinned);
    }

    #[test]
    fn retention_pruning_enforces_raw_turns_cap() {
        let conn = setup_mem_db();
        for i in 0..(MAX_RAW_TURNS_PER_AGENT + 5) {
            conn.execute(
                "INSERT INTO raw_turns (turn_id, ts, agent_id, project, project_key, working_dir, session_id, user_text, ai_text, touched_files_json)
                 VALUES (?1, ?2, 'backend', 'p', '/p', '/p', 's', 'u', 'a', '[]')",
                params![format!("turn_{i:04}"), format!("2026-11-17 {:02}:{:02}", (i / 60) % 24, i % 60)],
            )
            .unwrap();
        }
        prune_retention_conn(&conn, "backend");
        let cnt: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM raw_turns WHERE agent_id = 'backend'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cnt as usize, MAX_RAW_TURNS_PER_AGENT);
        assert!(load_raw_turn_conn(&conn, "turn_0000").unwrap().is_none());
    }

    #[tokio::test]
    async fn neural_memory_hook_is_append_only_and_ephemeral() {
        let hook = Arc::new(NeuralMemoryHook::new(
            Some("s_hook".into()),
            Path::new("/work/app").to_path_buf(),
        ));
        let mut convo = Conversation::new();
        convo.push(Message::user("hello"));
        let stored_wire = convo.model_history();
        let mut outgoing = stored_wire.clone();
        hook.pre_request(
            &mut outgoing,
            &TurnCtx {
                session_id: Some(Arc::from("s_hook")),
                turn_id: 1,
                round: 1,
                ..Default::default()
            },
        )
        .await;
        // 即使触发注入，历史前缀必须严格相等（append-only），且 Conversation 本体未被改动
        assert!(outgoing.len() >= stored_wire.len());
        assert_eq!(&outgoing[..stored_wire.len()], &stored_wire[..]);
        assert_eq!(convo.messages.len(), 1);
    }

    #[test]
    fn retracted_profile_facts_emit_transient_retraction_notice() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_old_rule".into(),
                text: "Always use MySQL 5.7 syntax".into(),
                project: "server".into(),
                project_key: "/work/server".into(),
                pinned: false,
                updated_at: "2026-11-17 09:00".into(),
                source_turn_id: "t0".into(),
                ..Default::default()
            },
        )
        .unwrap();

        let (blk_opt, frozen) = build_static_system_block_with_ids_conn(
            &conn,
            "s_retract",
            "/work/server",
            "server",
            false,
        )
        .unwrap();
        assert!(blk_opt
            .as_deref()
            .unwrap_or("")
            .contains("Always use MySQL 5.7 syntax"));
        record_session_static_state("s_retract", "backend", frozen);

        // 会话中途该条档案被删除
        conn.execute("DELETE FROM profiles WHERE id = 'pf_old_rule'", [])
            .unwrap();

        let rem = build_dynamic_turn_reminder_conn(
            &conn,
            "s_retract",
            "write a migration query",
            Path::new("/work/server"),
        )
        .unwrap()
        .expect("retraction reminder");
        assert!(rem.contains("[Retracted L1 Profile Facts"));
        assert!(rem.contains("Always use MySQL 5.7 syntax"));
    }

    #[test]
    fn deterministic_turn_id_and_head_tail_excerpt_work() {
        let id1 = deterministic_turn_id("backend", "s1", 2, "fix pool", "done");
        let id2 = deterministic_turn_id("backend", "s1", 2, "fix pool", "done");
        let id3 = deterministic_turn_id("backend", "s1", 2, "fix pool", "done differently");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);

        let long_ai = format!("HEAD_MARKER:{}:TAIL_MARKER", "x".repeat(3000));
        let excerpt = excerpt_head_tail(&long_ai, 100, 100);
        assert!(excerpt.contains("HEAD_MARKER"));
        assert!(excerpt.contains("TAIL_MARKER"));
        assert!(excerpt.contains("truncated middle"));
    }

    #[test]
    fn profile_and_task_fuzzy_dedup_and_stale_marking() {
        let mut conn = setup_mem_db();
        seed_backend_agent(&conn);
        let pkey = normalize_project_key(Path::new("/work/zap"));

        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_pnpm".into(),
                text: "Prefer pnpm workspace commands for frontend builds".into(),
                project: "zap".into(),
                project_key: pkey.clone(),
                pinned: false,
                updated_at: "2026-11-01 10:00".into(),
                source_turn_id: "t0".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_task_conn(
            &conn,
            &NeuralTask {
                id: "task_auth".into(),
                agent: "backend".into(),
                project: "zap".into(),
                project_key: pkey.clone(),
                working_dir: "/work/zap".into(),
                title: "Fix authentication middleware token bug".into(),
                progress_note: "Checked parser".into(),
                remaining_note: "Add tests".into(),
                touched_files: vec![],
                session_id: "s1".into(),
                source_turn_ids: vec!["t0".into()],
                status: "in_progress".into(),
                created_at: "2026-11-10 10:00".into(),
                updated_at: "2026-11-10 10:00".into(),
                done_at: String::new(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_task_conn(
            &conn,
            &NeuralTask {
                id: "task_old".into(),
                agent: "backend".into(),
                project: "zap".into(),
                project_key: pkey,
                working_dir: "/work/zap".into(),
                title: "Legacy migration cleanup".into(),
                progress_note: String::new(),
                remaining_note: "Verify".into(),
                touched_files: vec![],
                session_id: "s0".into(),
                source_turn_ids: vec!["t_old".into()],
                status: "in_progress".into(),
                created_at: "2026-10-01 10:00".into(),
                updated_at: "2026-10-01 10:00".into(),
                done_at: String::new(),
                ..Default::default()
            },
        )
        .unwrap();

        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s1",
            "t1",
            "2026-11-18 12:00",
            "zap",
            "/work/zap",
            "update auth token fix",
            "updated tests",
            &[],
            DistillOutput {
                should_record: true,
                profile_updates: vec![DistillProfileInput::Str(
                    "Prefer pnpm workspace commands for frontend builds always".into(),
                )],
                task_actions: vec![DistillTaskAction {
                    action: "update".into(),
                    task_id: String::new(),
                    title: "Fix authentication middleware token bug!".into(),
                    progress_note: "Added unit tests".into(),
                    remaining_note: String::new(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            false,
        )
        .unwrap();

        let profiles = load_profiles_conn(&conn, Some("backend"))
            .unwrap()
            .remove("backend")
            .unwrap_or_default();
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].id, "pf_pnpm");

        let tasks = load_tasks_conn(&conn, Some("backend")).unwrap();
        assert_eq!(tasks.len(), 2);
        let auth = tasks.iter().find(|t| t.id == "task_auth").unwrap();
        assert_eq!(auth.progress_note, "Added unit tests");
        let old = tasks.iter().find(|t| t.id == "task_old").unwrap();
        assert_eq!(old.status, "stale");
    }

    #[test]
    fn retention_pruning_keeps_solid_and_hub_nodes() {
        let conn = setup_mem_db();
        upsert_node_conn(
            &conn,
            &NeuralNode {
                id: "n_solid_hub".into(),
                name: "Core Architecture".into(),
                cat: "backend".into(),
                project: "zap".into(),
                project_key: "/work/zap".into(),
                status: "solid".into(),
                mastery: 95,
                tier: 0,
                windows: default_windows(),
                summary: "Foundation".into(),
                events: vec![],
                updated_at: "2026-01-01 00:00".into(),
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..(MAX_NODES_PER_AGENT + 4) {
            upsert_node_conn(
                &conn,
                &NeuralNode {
                    id: format!("n_tmp_{i:04}"),
                    name: format!("Temp Node {i}"),
                    cat: "backend".into(),
                    project: "zap".into(),
                    project_key: "/work/zap".into(),
                    status: "active".into(),
                    mastery: 50,
                    tier: 0,
                    windows: default_windows(),
                    summary: "ephemeral".into(),
                    events: vec![],
                    updated_at: format!("2026-11-17 {:02}:{:02}", (i / 60) % 24, i % 60),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        prune_retention_conn(&conn, "backend");
        let remaining = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(remaining.len(), MAX_NODES_PER_AGENT);
        assert!(remaining.iter().any(|n| n.id == "n_solid_hub"));
    }

    #[test]
    fn memory_recall_finds_matching_profile_beyond_top_ten() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        let pkey = normalize_project_key(Path::new("/work/zap"));
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "pf_target".into(),
                text: "Use Dameng8 uppercase table aliases in reports".into(),
                project: "zap".into(),
                project_key: pkey.clone(),
                pinned: false,
                updated_at: "2026-11-01 08:00".into(),
                source_turn_id: "t0".into(),
                ..Default::default()
            },
        )
        .unwrap();
        for i in 0..14 {
            upsert_profile_conn(
                &conn,
                "backend",
                &ProfileFact {
                    id: format!("pf_fill_{i:02}"),
                    text: format!("Unrelated preference item {i}"),
                    project: "zap".into(),
                    project_key: pkey.clone(),
                    pinned: false,
                    updated_at: format!("2026-11-18 09:{i:02}"),
                    source_turn_id: format!("tf_{i}"),
                    ..Default::default()
                },
            )
            .unwrap();
        }

        let res = execute_memory_recall_conn(
            &conn,
            "",
            Path::new("/work/zap"),
            MemoryRecallArgs {
                query: "Dameng8".into(),
                project: String::new(),
                date_range: String::new(),
                include_raw: Some(false),
                limit: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res.contains("Use Dameng8 uppercase table aliases in reports"));
    }

    #[test]
    fn test_fresh_session_locks_agent_and_multi_switch_cycle_with_none() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        upsert_agent_conn(
            &conn,
            &NeuralAgent {
                id: "frontend".into(),
                icon: "🎨".into(),
                name: "前端小白".into(),
                label: "🎨 前端小白".into(),
                color: "#ec4899".into(),
                hint: "Vue 前端开发".into(),
                system_prompt: "遵守 Vue 前端规范".into(),
            },
            1,
        )
        .unwrap();

        let pkey = normalize_project_key(Path::new("/work/zap"));
        upsert_profile_conn(
            &conn,
            "frontend",
            &ProfileFact {
                id: "p-fe-1".into(),
                text: "前端偏好提取 Design Tokens".into(),
                project: "Zap".into(),
                project_key: pkey.clone(),
                pinned: false,
                updated_at: "2026-11-19 10:01".into(),
                source_turn_id: "t1".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_profile_conn(
            &conn,
            "backend",
            &ProfileFact {
                id: "p-be-1".into(),
                text: "后端偏好 SQLite WAL 模式".into(),
                project: "Zap".into(),
                project_key: pkey.clone(),
                pinned: false,
                updated_at: "2026-11-19 10:02".into(),
                source_turn_id: "t2".into(),
                ..Default::default()
            },
        )
        .unwrap();

        // 1. 模拟用户在草稿态选择了 frontend（cfg_active_agent = "frontend"），随后创建会话 sess-A
        let mut cfg = get_config_conn(&conn);
        cfg.active_agent = "frontend".into();
        save_config_conn(&conn, &cfg).unwrap();

        let resolved_a = resolve_agent_for_session_conn(&conn, "sess-A");
        assert_eq!(resolved_a.as_ref().map(|a| a.id.as_str()), Some("frontend"));

        // 首轮对话初始化时绑定 sess-A（复现 WebUI 预生成草稿 session_id 场景）
        bind_fresh_session_agent_if_unset_conn(&conn, "sess-A", "frontend").unwrap();

        // 验证：cfg_active_agent 已被消费重置为 "none"，避免后续新建的通用模式会话 sess-B 被幽灵污染
        assert_eq!(get_config_conn(&conn).active_agent, "none");
        assert!(
            resolve_agent_for_session_conn(&conn, "sess-B").is_none(),
            "后续未选智能体新建的会话 sess-B 应为通用模式（None），绝不能幽灵继承 frontend"
        );
        // sess-B 首轮作为通用模式显式落库 "none"
        bind_fresh_session_agent_if_unset_conn(&conn, "sess-B", "none").unwrap();
        assert!(resolve_agent_for_session_conn(&conn, "sess-B").is_none());

        // 2. 验证同一个会话 sess-multi 内多次切换智能体：
        //    none -> frontend -> backend -> frontend -> none -> backend
        let sid = "sess-multi";
        bind_fresh_session_agent_if_unset_conn(&conn, sid, "none").unwrap();
        let (_, init_frozen) =
            build_static_system_block_with_ids_conn(&conn, sid, &pkey, "Zap", false).unwrap();
        record_session_static_state(sid, "none", init_frozen);

        // 阶段 1：初始通用模式 (none) 下提问，不应产生任何切换或记忆提醒
        let rem_none_init =
            build_dynamic_turn_reminder_conn(&conn, sid, "今天天气不错", Path::new("/work/zap"))
                .unwrap();
        assert!(rem_none_init.is_none());

        // 阶段 2：中途第 1 次切换 -> frontend
        set_session_agent_conn(&conn, sid, "frontend").unwrap();
        let rem_to_fe = build_dynamic_turn_reminder_conn(
            &conn,
            sid,
            "帮我看看页面样式",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("切换到 frontend 后应注入切换指令");
        assert!(rem_to_fe.contains("[Neural Agent Switched: 🎨 前端小白 (frontend)]"));
        assert!(rem_to_fe.contains("遵守 Vue 前端规范"));
        assert!(rem_to_fe.contains("前端偏好提取 Design Tokens"));

        // 在 frontend 期间新学一条 L1 档案 p-fe-2，下一轮仍在 frontend 时应增量注入 [Newly Learned L1 Profile Facts]
        upsert_profile_conn(
            &conn,
            "frontend",
            &ProfileFact {
                id: "p-fe-2".into(),
                text: "表格严禁私自写默认值兜底".into(),
                project: "Zap".into(),
                project_key: pkey.clone(),
                pinned: false,
                updated_at: "2026-11-19 10:05".into(),
                source_turn_id: "t3".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let rem_fe_turn2 =
            build_dynamic_turn_reminder_conn(&conn, sid, "继续改表格组件", Path::new("/work/zap"))
                .unwrap()
                .expect("frontend 第二轮应保持切换头并带出新学 L1 档案");
        assert!(rem_fe_turn2.contains("[Neural Agent Switched: 🎨 前端小白 (frontend)]"));
        assert!(rem_fe_turn2.contains("[Newly Learned L1 Profile Facts]"));
        assert!(rem_fe_turn2.contains("表格严禁私自写默认值兜底"));

        // 阶段 3：中途第 2 次切换 -> backend
        set_session_agent_conn(&conn, sid, "backend").unwrap();
        let rem_to_be = build_dynamic_turn_reminder_conn(
            &conn,
            sid,
            "现在改一下 SQLite 接口",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("切换到 backend 后应注入 backend 人设与档案");
        assert!(rem_to_be.contains("[Neural Agent Switched: ⚙ Backend (backend)]"));
        assert!(rem_to_be.contains("You are a backend specialist."));
        assert!(rem_to_be.contains("后端偏好 SQLite WAL 模式"));
        assert!(!rem_to_be.contains("遵守 Vue 前端规范"));
        // 刚切到 backend 的第一轮不应误把 backend 已有档案当成 Newly Learned 重复塞一遍
        assert!(!rem_to_be.contains("[Newly Learned L1 Profile Facts]"));

        // 阶段 4：中途第 3 次切换 -> 再次切回 frontend
        set_session_agent_conn(&conn, sid, "frontend").unwrap();
        let rem_back_to_fe = build_dynamic_turn_reminder_conn(
            &conn,
            sid,
            "再看一眼前端表单",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("切回 frontend 应重新注入 frontend 接管指令");
        assert!(rem_back_to_fe.contains("[Neural Agent Switched: 🎨 前端小白 (frontend)]"));
        assert!(rem_back_to_fe.contains("前端偏好提取 Design Tokens"));
        assert!(rem_back_to_fe.contains("表格严禁私自写默认值兜底"));

        // 阶段 5：中途第 4 次切换 -> 切回通用模式 (none)
        set_session_agent_conn(&conn, sid, "none").unwrap();
        let rem_to_none = build_dynamic_turn_reminder_conn(
            &conn,
            sid,
            "现在聊点通用问题",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("从 frontend 切回 none 时必须发送一次解除隔离指令");
        assert!(rem_to_none.contains("[Neural Agent Switched: none]"));

        // 切回 none 后的下一轮普通对话，不应再重复发 [Neural Agent Switched: none]
        let rem_none_again =
            build_dynamic_turn_reminder_conn(&conn, sid, "继续聊通用问题", Path::new("/work/zap"))
                .unwrap();
        assert!(
            rem_none_again.is_none(),
            "已经在 none 状态通知过一次后，后续轮次应保持 0 注入开销"
        );

        // 阶段 6：中途第 5 次切换 -> 从 none 再次切到 backend
        set_session_agent_conn(&conn, sid, "backend").unwrap();
        let rem_to_be_again = build_dynamic_turn_reminder_conn(
            &conn,
            sid,
            "再帮我查个后端问题",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("从 none 再次切到 backend 应正常生效");
        assert!(rem_to_be_again.contains("[Neural Agent Switched: ⚙ Backend (backend)]"));
    }

    #[test]
    fn v2_to_v3_migration_preserves_data_builds_anchors_and_drops_raw_turns_fts() {
        let conn = Connection::open_in_memory().unwrap();
        // 模拟真实线上 v2 库结构（含膨胀的 raw_turns_fts）
        conn.execute_batch(
            "
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE agents (id TEXT PRIMARY KEY, icon TEXT NOT NULL, name TEXT NOT NULL, label TEXT NOT NULL, color TEXT NOT NULL, hint TEXT NOT NULL, system_prompt TEXT NOT NULL, sort_order INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE session_agents (session_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE profiles (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, project TEXT NOT NULL, project_key TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, source_turn_id TEXT NOT NULL);
            CREATE TABLE tasks (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, project TEXT NOT NULL, project_key TEXT NOT NULL DEFAULT '', working_dir TEXT NOT NULL, title TEXT NOT NULL, progress_note TEXT NOT NULL, remaining_note TEXT NOT NULL, touched_files_json TEXT NOT NULL, session_id TEXT NOT NULL, source_turn_ids_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, done_at TEXT NOT NULL);
            CREATE TABLE nodes (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, name TEXT NOT NULL, project TEXT NOT NULL, project_key TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, mastery INTEGER NOT NULL, tier INTEGER NOT NULL, windows_json TEXT NOT NULL, summary TEXT NOT NULL, events_json TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE links (source TEXT NOT NULL, target TEXT NOT NULL, PRIMARY KEY (source, target));
            CREATE TABLE timeline (id TEXT PRIMARY KEY, ts TEXT NOT NULL, agent_id TEXT NOT NULL, project TEXT NOT NULL, project_key TEXT NOT NULL DEFAULT '', working_dir TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL, raw_file TEXT NOT NULL, user_goal TEXT NOT NULL, summary TEXT NOT NULL, ai_outcome TEXT NOT NULL, keywords_json TEXT NOT NULL, touched_files_json TEXT NOT NULL, distilled_by TEXT NOT NULL);
            CREATE TABLE raw_turns (turn_id TEXT PRIMARY KEY, ts TEXT NOT NULL, agent_id TEXT NOT NULL, project TEXT NOT NULL, project_key TEXT NOT NULL DEFAULT '', working_dir TEXT NOT NULL, session_id TEXT NOT NULL, user_text TEXT NOT NULL, ai_text TEXT NOT NULL, touched_files_json TEXT NOT NULL);
            CREATE VIRTUAL TABLE nodes_fts USING fts5(node_id UNINDEXED, agent_id UNINDEXED, search_text);
            CREATE VIRTUAL TABLE timeline_fts USING fts5(event_id UNINDEXED, agent_id UNINDEXED, search_text);
            CREATE VIRTUAL TABLE raw_turns_fts USING fts5(turn_id UNINDEXED, agent_id UNINDEXED, search_text);
            INSERT INTO nodes VALUES ('n_legacy', 'backend', 'MSVC编译坑', 'Zap', '/work/zap', 'solid', 90, 0, '[\"all\"]', 'cl.exe 报 D8050 时在 Cargo.toml 设 debug=0', '[]', '2026-11-18 12:00');
            PRAGMA user_version = 2;
            ",
        )
        .unwrap();

        init_schema(&conn).unwrap();
        let ver: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, CURRENT_SCHEMA_VERSION);

        // 验证 raw_turns_fts 已彻底移除
        let raw_fts_exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'raw_turns_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(raw_fts_exists, 0);

        // 验证老节点自动补齐了 active facts 与 D8050 / Cargo.toml 精确路牌
        let nodes = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].active_facts().len(), 1);
        let plan = analyze_query("遇到 D8050 报错怎么处理", &[]);
        assert!(plan.anchors.contains(&"d8050".to_string()));
        let hits = query_nodes_ranked_conn(&conn, "backend", &plan, None, 3);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node.id, "n_legacy");
        assert!(hits[0].anchor_hits >= 1);
    }

    #[test]
    fn card_evolution_preserves_old_facts_and_handles_supersede_without_collision() {
        let mut conn = setup_mem_db();
        seed_backend_agent(&conn);
        let pkey = normalize_project_key(Path::new("/work/zap"));

        // 1. 首次创建卡片，带一条事实与双语检索标签
        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s1",
            "t1",
            "2026-11-19 10:00",
            "zap",
            "/work/zap",
            "讨论鉴权方案",
            "决定使用 Cookie Session",
            &["src/auth.rs".into()],
            DistillOutput {
                should_record: true,
                node_actions: vec![DistillNodeAction {
                    name: "鉴权方案".into(),
                    aliases: vec!["登录认证".into()],
                    search_tags: vec!["auth".into(), "登录态".into(), "cookie".into()],
                    summary: "初始采用 Cookie Session 方案".into(),
                    mastery: 80,
                    status: "active".into(),
                    fact_ops: vec![DistillNodeFactOp {
                        action: "add".into(),
                        kind: "decision".into(),
                        text: "初始采用 Cookie Session 保存登录态".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            },
            false,
        )
        .unwrap();

        let nodes1 = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(nodes1.len(), 1);
        let card_id = nodes1[0].id.clone();
        let old_fact_id = nodes1[0].facts[0].id.clone();

        // 2. 几天后换了别名且改用 JWT：通过 target_node_id 融合更新，旧事实标作废，新事实生效，并追加一条新要点
        apply_llm_distill_conn(
            &mut conn,
            "backend",
            "s2",
            "t2",
            "2026-11-20 09:00",
            "zap",
            "/work/zap",
            "登录改用 JWT，并加刷新令牌",
            "已切换为 JWT + Refresh Token",
            &["src/auth.rs".into()],
            DistillOutput {
                should_record: true,
                node_actions: vec![DistillNodeAction {
                    target_node_id: card_id.clone(),
                    name: "登录认证".into(),
                    search_tags: vec!["jwt".into(), "token过期".into()],
                    summary: "现采用 JWT 签发访问令牌并配合 Refresh Token 续期".into(),
                    mastery: 92,
                    status: "solid".into(),
                    fact_ops: vec![
                        DistillNodeFactOp {
                            action: "supersede".into(),
                            fact_id: old_fact_id.clone(),
                            kind: "decision".into(),
                            text: "改用无状态 JWT 作为核心鉴权方案".into(),
                        },
                        DistillNodeFactOp {
                            action: "add".into(),
                            kind: "pitfall".into(),
                            text: "Refresh Token 必须走 HttpOnly 防止 XSS 窃取".into(),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                }],
                profile_actions: vec![DistillProfileAction {
                    action: "add".into(),
                    text: "所有鉴权接口统一返回 401 错误码".into(),
                    pin: false,
                    kind: "rule".into(),
                    ..Default::default()
                }],
                ..Default::default()
            },
            false,
        )
        .unwrap();

        let nodes2 = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(nodes2.len(), 1, "同主题换别名绝不能分裂成两张卡片");
        let card = &nodes2[0];
        assert_eq!(card.id, card_id);
        assert_eq!(card.active_facts().len(), 2);
        let old_fact = card.facts.iter().find(|f| f.id == old_fact_id).unwrap();
        assert_eq!(old_fact.status, "superseded");
        assert!(!old_fact.superseded_by.is_empty());

        // 默认查询只注入生效要点，不混入已作废的 Cookie Session
        let rem_default = build_dynamic_turn_reminder_conn(
            &conn,
            "",
            "帮我看下 token过期 怎么处理",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("应通过写入期双语大白话标签命中卡片");
        assert!(rem_default.contains("JWT"));
        assert!(rem_default.contains("HttpOnly"));
        assert!(!rem_default.contains("Cookie Session"));

        // 显式追问历史作废原因（或通过 memory_recall include_superseded=true）才能看到旧方案
        let recalled_hist = execute_memory_recall_conn(
            &conn,
            "",
            Path::new("/work/zap"),
            MemoryRecallArgs {
                query: "鉴权方案".into(),
                drawer: "knowledge".into(),
                include_superseded: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(recalled_hist.contains("superseded"));
        assert!(recalled_hist.contains("Cookie Session"));
        let _ = pkey;
    }

    #[test]
    fn exact_anchor_camel_split_short_followup_and_one_hop_neighbor_work() {
        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        let pkey = normalize_project_key(Path::new("/work/zap"));

        upsert_node_conn(
            &conn,
            &NeuralNode {
                id: "n_router".into(),
                name: "路由转发器 resolveAgent".into(),
                cat: "backend".into(),
                project: "zap".into(),
                project_key: pkey.clone(),
                status: "solid".into(),
                mastery: 90,
                summary: "在 neural_memory/store.rs 中通过 resolveAgent 绑定会话归属".into(),
                aliases: vec!["会话路由".into()],
                search_tags: vec!["agent switching".into(), "智能体串台".into()],
                facts: vec![NodeFact {
                    id: "f1".into(),
                    kind: "fact".into(),
                    text: "调用 resolve_agent_for_session_conn 时必须先查 session_agents 表".into(),
                    status: "active".into(),
                    ..Default::default()
                }],
                updated_at: "2026-11-20 09:00".into(),
                ..Default::default()
            },
        )
        .unwrap();
        upsert_node_conn(
            &conn,
            &NeuralNode {
                id: "n_wal".into(),
                name: "SQLite 读写分离".into(),
                cat: "backend".into(),
                project: "zap".into(),
                project_key: pkey.clone(),
                status: "solid".into(),
                mastery: 88,
                summary: "写连接串行事务 + 独立只读连接池释放 WAL 并发".into(),
                updated_at: "2026-11-20 09:01".into(),
                ..Default::default()
            },
        )
        .unwrap();
        super::store::upsert_link_conn(&conn, "n_router", "n_wal", "depends_on");

        // 1. 精确路牌（store.rs）+ 1-hop 关联邻居（SQLite 读写分离）直达
        let anchors = extract_query_anchors("帮我看下 store.rs 里面的逻辑");
        assert!(anchors.contains(&"store.rs".to_string()));
        let rem_anchor = build_dynamic_turn_reminder_conn(
            &conn,
            "",
            "帮我看下 store.rs 里面的逻辑",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("精确文件名路牌应秒中卡片");
        assert!(rem_anchor.contains("路由转发器 resolveAgent"));
        assert!(rem_anchor.contains("[linked:depends_on]"));
        assert!(rem_anchor.contains("SQLite 读写分离"));

        // 2. 驼峰/下划线互通（用下划线查驼峰卡片）
        let rem_camel = build_dynamic_turn_reminder_conn(
            &conn,
            "",
            "查一下 resolve_agent 是怎么做的",
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("下划线查询应通过驼峰拆词命中 resolveAgent");
        assert!(rem_camel.contains("路由转发器 resolveAgent"));

        // 3. 短句追问借用上轮主语上下文词
        let prior = vec!["我们之前讨论过智能体串台和 resolveAgent 路由".to_string()];
        let rem_short = build_dynamic_turn_reminder_with_context_conn(
            &conn,
            "",
            "那怎么改？",
            &prior,
            Path::new("/work/zap"),
        )
        .unwrap()
        .expect("短句追问应借用上轮主语召回相关卡片");
        assert!(rem_short.contains("路由转发器 resolveAgent"));
    }

    #[test]
    fn team_bridge_skips_in_flight_sessions_and_syncs_finalized_deliveries_idempotently() {
        assert!(is_team_managed_session("team_trun_123_task_1"));
        assert!(!is_team_managed_session("sess_normal_123"));

        let conn = setup_mem_db();
        seed_backend_agent(&conn);
        let delivery = TeamTaskDelivery {
            run_id: "trun_order_export".into(),
            run_title: "订单批量导出需求".into(),
            run_status: "completed".into(),
            task_id: "tt_be_1".into(),
            domain: "backend".into(),
            task_title: "实现导出接口 /api/orders/export".into(),
            task_status: "completed".into(),
            agent_id: "backend".into(),
            working_dir: "/work/tsm-server".into(),
            sql_schema: String::new(),
            interface_spec: "POST /api/orders/export 接受 dateRange 返回异步 taskId".into(),
            contract_summary: "完成导出接口与限流保护".into(),
            verify_feedback: "单测与接口契约验收通过".into(),
            touched_files: vec!["src/routes/orders_export.rs".into()],
            updated_at: "2026-11-20 10:30:00".into(),
        };

        assert!(apply_team_delivery_conn(&conn, &delivery).unwrap());
        // 相同 updated_at 重复同步应严格幂等跳过
        assert!(!apply_team_delivery_conn(&conn, &delivery).unwrap());

        let tasks = load_tasks_conn(&conn, Some("backend")).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].team_run_id, "trun_order_export");
        assert_eq!(tasks[0].status, "done");

        let nodes = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].summary.contains("/api/orders/export"));

        // 通过接口路牌直接秒查回流成果
        let rem = build_dynamic_turn_reminder_conn(
            &conn,
            "",
            "帮我检查一下 /api/orders/export 的参数",
            Path::new("/work/tsm-server"),
        )
        .unwrap()
        .expect("Team 回流接口路牌应直接命中");
        assert!(rem.contains("订单批量导出需求"));
    }

    #[test]
    fn persistent_draft_queue_and_legacy_node_consolidation_work() {
        let mut conn = setup_mem_db();
        seed_backend_agent(&conn);

        // 1. 持久化草稿队列入队与按会话读取（防重启丢失）
        let inserted = enqueue_draft_turn_conn(
            &conn,
            &DraftTurn {
                id: 0,
                turn_id: "turn_draft_1".into(),
                session_id: "s_draft".into(),
                agent_id: "backend".into(),
                project: "zap".into(),
                project_key: "/work/zap".into(),
                working_dir: "/work/zap".into(),
                user_text: "记住以后 SQL 一律参数化".into(),
                ai_text: "好的，已按参数化写法处理".into(),
                touched_files: vec!["src/db.rs".into()],
                provider_name: "gpt-4o".into(),
                explicit_remember: true,
                retry_count: 0,
                created_at: "2026-11-20 11:00".into(),
            },
        )
        .unwrap();
        assert!(inserted);
        let pending = load_pending_drafts_for_scope_conn(&conn, "backend", "s_draft").unwrap();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].explicit_remember);

        // 2. 老记忆碎片一键合并（保留双边要点 + 收编别名）
        for (id, name, sum) in [
            ("n_dup_1", "Redis 缓存", "TTL 默认 300 秒"),
            ("n_dup_2", "redis-缓存", "热点 Key 加本地二级缓存防击穿"),
        ] {
            upsert_node_conn(
                &conn,
                &NeuralNode {
                    id: id.into(),
                    name: name.into(),
                    cat: "backend".into(),
                    project: "zap".into(),
                    project_key: "/work/zap".into(),
                    status: "active".into(),
                    mastery: 80,
                    summary: sum.into(),
                    updated_at: "2026-11-20 10:00".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let (merged, remaining) =
            consolidate_legacy_nodes_conn(&mut conn, Some("backend")).unwrap();
        assert_eq!(merged, 1);
        assert_eq!(remaining, 1);
        let final_nodes = load_nodes_conn(&conn, Some("backend")).unwrap();
        assert_eq!(final_nodes[0].active_facts().len(), 2);
    }
}
