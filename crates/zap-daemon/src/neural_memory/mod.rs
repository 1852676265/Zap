//! 智能体独立记忆与认知引擎（SQLite WAL + FTS5 + `pre_request` 瞬态注入 + 后台 LLM 提炼）。
//!
//! 职责拆分为以下子模块：
//! - `types`：核心数据结构与规范化 `project_key`
//! - `redact`：凭据与密钥脱敏
//! - `schema`：SQLite 建表与 `PRAGMA user_version` 迁移
//! - `store`：可恢复连接池、SQL 下推查询、留存淘汰与旧版迁移
//! - `search`：CJK/日韩切词、时间解析、词边界项目点名检测与 FTS5 检索
//! - `injection`：静态系统前缀、`NeuralMemoryHook`（`pre_request` 瞬态尾注）
//! - `recall_tool`：`memory_recall` 只读检索工具
//! - `distill`：按智能体串行排队的后台异步提炼与事务写回
//! - `rpc`：WebUI / 插件 RPC 分发器

mod distill;
mod injection;
mod recall_tool;
mod redact;
mod rpc;
mod schema;
mod search;
mod store;
mod types;

pub use distill::maybe_capture_turn_from_snapshot;
pub use injection::{
    apply_session_static_system_append, build_dynamic_turn_reminder, build_static_system_block,
    note_fresh_session_static_agent, NeuralMemoryBinding, NeuralMemoryHook,
};
pub use recall_tool::MemoryRecallTool;
pub use rpc::handle_rpc;
pub use search::clean_user_prompt;
pub use types::{
    NeuralAgent, NeuralConfig, NeuralLink, NeuralNode, NeuralTask, NodeEvent, ProfileFact, RawTurn,
    TimelineEvent,
};

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use rusqlite::{params, Connection};
    use zap_kernel::hook::{LifecycleHooks, TurnCtx};
    use zap_kernel::message::{Conversation, Message};
    use zap_kernel::tool::ToolCall;

    use super::distill::{
        apply_fallback_distill_conn, apply_llm_distill_conn, deterministic_turn_id,
        excerpt_head_tail, extract_last_turn_payload, user_explicitly_requests_pin,
        DistillNodeAction, DistillOutput, DistillProfileInput, DistillProfileObj,
        DistillTaskAction,
    };
    use super::injection::{
        build_dynamic_turn_reminder_conn, build_static_system_block_with_ids_conn,
        record_session_static_state, NeuralMemoryHook,
    };
    use super::recall_tool::{execute_memory_recall_conn, MemoryRecallArgs};
    use super::redact::redact_secrets;
    use super::schema::{init_schema, CURRENT_SCHEMA_VERSION};
    use super::search::{
        build_fts_queries, build_fts_search_text, detect_mentioned_project, is_slash_command,
        parse_temporal_range,
    };
    use super::store::{
        insert_timeline_conn, load_links_conn, load_nodes_conn, load_profiles_conn,
        load_raw_turn_conn, load_tasks_conn, load_timeline_conn, prune_retention_conn,
        setup_mem_db, upsert_agent_conn, upsert_node_conn, upsert_profile_conn,
        upsert_raw_turn_conn, upsert_task_conn, KnownProject,
    };
    use super::types::{
        default_windows, normalize_project_key, NeuralAgent, NeuralNode, NeuralTask, ProfileFact,
        RawTurn, TimelineEvent, MAX_NODES_PER_AGENT, MAX_NON_PINNED_PROFILES,
        MAX_RAW_TURNS_PER_AGENT,
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
            },
        )
        .unwrap();

        let distill = DistillOutput {
            should_record: true,
            profile_updates: vec![],
            profile_removes: vec!["50%".into(), "置顶规范".into()],
            task_actions: vec![],
            node_actions: vec![],
            timeline: None,
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
            profile_updates: vec![],
            profile_removes: vec![],
            task_actions: vec![],
            node_actions: vec![DistillNodeAction {
                name: "Redis 缓存".into(),
                summary: "初版缓存方案".into(),
                mastery: 80,
                status: "active".into(),
                related_nodes: vec![],
            }],
            timeline: None,
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
            profile_updates: vec![],
            profile_removes: vec![],
            task_actions: vec![],
            node_actions: vec![
                DistillNodeAction {
                    name: "redis-缓存".into(),
                    summary: "补充过期淘汰策略".into(),
                    mastery: 85,
                    status: "solid".into(),
                    related_nodes: vec![],
                },
                DistillNodeAction {
                    name: "达梦连接池".into(),
                    summary: "连接池独立优化".into(),
                    mastery: 78,
                    status: "active".into(),
                    related_nodes: vec!["Redis 缓存".into()],
                },
            ],
            timeline: None,
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
                profile_removes: vec![],
                task_actions: vec![],
                node_actions: vec![],
                timeline: None,
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
                profile_removes: vec![],
                task_actions: vec![DistillTaskAction {
                    action: "update".into(),
                    task_id: String::new(),
                    title: "Fix authentication middleware token bug!".into(),
                    progress_note: "Added unit tests".into(),
                    remaining_note: String::new(),
                }],
                node_actions: vec![],
                timeline: None,
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
            },
        )
        .unwrap();
        assert!(res.contains("Use Dameng8 uppercase table aliases in reports"));
    }
}
