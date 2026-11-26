//! `zap-agent-teams` 跨项目多智能体协作常驻引擎（方案 B：外部插件 UI + Rust 守护进程常驻编排）。
//!
//! 职责模块拆分：
//! - `types`：领域模型、乐观锁版本号、attempt_id 栅栏与 Kahn DAG 波次拓扑切分
//! - `schema`：SQLite WAL 账本建表与索引迁移
//! - `store`：持久化连接池、CAS 草稿保存、attempt 栅栏提交与重启孤儿自愈
//! - `tools`：团队子会话按需挂载工具（`team_publish_contract` 契约发布 + `team_ask_peer` 跨仓库只读问询）
//! - `planner`：AI 主控拆单规划、默认项目与专属记忆智能体智能匹配、联合会诊转修复流水线
//! - `prompts`：全角色提示词模板注册中心、目录特征感知器与多层级覆写安全渲染
//! - `runner`：波次并发调度、独立验收闸门、最多 `max_repairs` 次自动打回返工与跨库会诊报告汇总
//! - `rpc`：面向外部插件 `@zap/agent-teams` 的 `teams:*` 统一 RPC 路由

pub mod host;
mod planner;
mod prompts;
mod rpc;
mod runner;
mod schema;
mod sessions;
mod store;
mod tools;
mod types;

pub use host::{HostChildTurnRequest, HostModelInfo, HostProjectInfo, TeamHost};
pub use rpc::handle_rpc;
pub use tools::apply_session_team_tools;
pub use types::{
    compute_and_assign_waves, RoleBindingPreset, TeamContract, TeamEvent, TeamGlobalConfig,
    TeamRound, TeamRoundTaskSnapshot, TeamRun, TeamRunDetail, TeamTask,
};

#[cfg(test)]
mod tests {
    use super::store::{
        append_task_note_conn, append_tasks_to_run_conn, build_round_task_snapshot,
        claim_task_attempt_conn, create_run_with_tasks_conn, finalize_round_delivery_conn,
        finish_task_fenced_conn, load_run_conn, load_run_detail_conn,
        load_task_statuses_by_run_conn, schedule_revision_conn, setup_mem_team_db,
        update_run_delivery_conn, update_run_draft_cas_conn, upsert_contract_conn,
        RevisionRoundOptions, RevisionTaskTarget, RunDraftExtras,
    };
    use super::types::{
        compute_and_assign_waves, TeamContract, TeamImage, TeamRound, TeamRun, TeamTask,
    };

    fn dummy_task(id: &str, domain: &str, deps: Vec<&str>) -> TeamTask {
        TeamTask {
            id: id.to_string(),
            run_id: "r1".to_string(),
            domain: domain.to_string(),
            title: format!("Task {id}"),
            description: String::new(),
            acceptance_criteria: String::new(),
            depends_on: deps.into_iter().map(str::to_string).collect(),
            wave: 1,
            sort_order: 0,
            project_slot_id: String::new(),
            project_name: String::new(),
            avatar_id: String::new(),
            working_dir: "D:/Projects/Zap".to_string(),
            project_hash: "abc".to_string(),
            agent_id: "backend".to_string(),
            agent_label: "⚙ Backend".to_string(),
            model: String::new(),
            session_id: String::new(),
            custom_prompt: String::new(),
            images: Vec::new(),
            revision_note: String::new(),
            status: "pending".to_string(),
            attempt: 0,
            attempt_id: String::new(),
            repair_count: 0,
            contract_summary: String::new(),
            last_output_excerpt: String::new(),
            verify_verdict: String::new(),
            verify_feedback: String::new(),
            touched_files: Vec::new(),
            round_touched_files: Vec::new(),
            started_at: String::new(),
            updated_at: "2026-11-18 10:00:00".to_string(),
            finished_at: String::new(),
        }
    }

    fn persistence_run() -> TeamRun {
        serde_json::from_value(serde_json::json!({
            "id": "r1", "title": "roundtrip", "userGoal": "goal",
            "sqlWorkingDir": "/team/old", "sqlAgentId": "sql-agent",
            "sqlModel": "sql-model", "sqlSessionId": "sql-session",
            "promptOverrides": {"sql_design": "run-template"},
            "executionPlan": "plan", "sqlSchemaDraft": "schema",
            "interfaceSpec": "api", "deliverySql": "delivery"
        }))
        .unwrap()
    }

    #[test]
    fn templates_bindings_and_custom_prompt_roundtrip_and_cas() {
        use super::store::*;
        let mut conn = setup_mem_team_db();
        let mut cfg = super::types::TeamGlobalConfig::default();
        cfg.prompt_templates
            .insert("sql_design".into(), "global-template".into());
        cfg.prompt_templates.insert("empty".into(), " ".into());
        save_config_conn(&conn, &cfg).unwrap();
        let loaded_cfg = load_config_conn(&conn);
        assert_eq!(
            loaded_cfg.prompt_templates.get("sql_design").unwrap(),
            "global-template"
        );
        assert!(!loaded_cfg.prompt_templates.contains_key("empty"));
        let run = persistence_run();
        let mut task = dummy_task("t1", "backend", vec![]);
        task.custom_prompt = "task-specific prompt".into();
        task.session_id = "db-session".into();
        task.attempt = 3;
        task.attempt_id = "db-attempt".into();
        task.status = "completed".into();
        task.contract_summary = "old-contract".into();
        task.verify_verdict = "passed".into();
        task.touched_files = vec!["old.rs".into()];
        create_run_with_tasks_conn(&mut conn, &run, &[task.clone()]).unwrap();
        let loaded = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(loaded.run.sql_session_id, "sql-session");
        assert_eq!(loaded.run.sql_agent_id, "sql-agent");
        assert_eq!(loaded.run.sql_model, "sql-model");
        assert_eq!(loaded.run.execution_plan, "plan");
        assert_eq!(loaded.run.sql_schema_draft, "schema");
        assert_eq!(loaded.run.interface_spec, "api");
        assert_eq!(loaded.run.delivery_sql, "delivery");
        assert_eq!(loaded.run.prompt_overrides, run.prompt_overrides);
        assert_eq!(
            list_runs_conn(&conn, 10).unwrap()[0].prompt_overrides,
            run.prompt_overrides
        );
        assert_eq!(loaded.tasks[0].custom_prompt, task.custom_prompt);
        assert_eq!(
            load_task_conn(&conn, "t1").unwrap().unwrap().attempt_id,
            "db-attempt"
        );
        let mut client = task.clone();
        client.session_id = "injected".into();
        client.attempt_id = "injected".into();
        client.verify_verdict = "injected".into();
        update_run_draft_cas_conn(
            &mut conn,
            "r1",
            Some(1),
            "",
            true,
            2,
            "",
            None,
            &[client.clone()],
            &RunDraftExtras::default(),
        )
        .unwrap();
        let kept = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(kept.tasks[0].session_id, "db-session");
        assert_eq!(kept.tasks[0].attempt_id, "db-attempt");
        assert_eq!(kept.tasks[0].verify_verdict, "passed");
        assert_eq!(kept.run.sql_session_id, "sql-session");
        assert_eq!(kept.run.prompt_overrides, run.prompt_overrides);
        upsert_contract_conn(&conn, "r1", "t1", "backend", "old", "old", "").unwrap();
        let rev = load_run_conn(&conn, "r1").unwrap().unwrap().revision;
        client.working_dir = "/team/new".into();
        let empty = std::collections::BTreeMap::new();
        update_run_draft_cas_conn(
            &mut conn,
            "r1",
            Some(rev),
            "",
            true,
            2,
            "",
            None,
            &[client],
            &RunDraftExtras {
                sql_working_dir: Some("/team/new"),
                sql_agent_id: Some(""),
                sql_model: Some(""),
                interface_spec: None,
                prompt_overrides: Some(&empty),
                lead_agent_id: None,
            },
        )
        .unwrap();
        let reset = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert!(reset.run.sql_session_id.is_empty());
        assert!(reset.run.sql_model.is_empty());
        assert_eq!(reset.run.sql_agent_id, "none");
        assert!(reset.run.prompt_overrides.is_empty());
        let t = &reset.tasks[0];
        assert!(t.session_id.is_empty() && t.attempt_id.is_empty());
        assert!(t.touched_files.is_empty() && t.contract_summary.is_empty());
        assert!(t.verify_verdict.is_empty() && t.last_output_excerpt.is_empty());
        assert!(t.project_hash.is_empty() && t.started_at.is_empty() && t.finished_at.is_empty());
        assert_eq!(t.status, "pending");
        assert_eq!(t.attempt, 0);
        assert!(reset.contracts.is_empty());
        assert!(!finish_task_fenced_conn(
            &conn,
            "t1",
            "db-attempt",
            "completed",
            "stale",
            None,
            "passed",
            "",
            0,
            &[],
            &[]
        )
        .unwrap());
        update_run_sql_binding_conn(&conn, "r1", None, None, None, Some("new-session")).unwrap();
        update_run_sql_binding_conn(&conn, "r1", Some("/team/next"), None, None, None).unwrap();
        assert!(load_run_conn(&conn, "r1")
            .unwrap()
            .unwrap()
            .sql_session_id
            .is_empty());
    }

    #[test]
    fn schema_migrates_old_prompt_columns_without_losing_rows() {
        let mut conn = setup_mem_team_db();
        create_run_with_tasks_conn(
            &mut conn,
            &persistence_run(),
            &[dummy_task("t1", "backend", vec![])],
        )
        .unwrap();
        conn.execute_batch(
            "ALTER TABLE team_runs DROP COLUMN prompt_overrides_json;
            ALTER TABLE team_tasks DROP COLUMN custom_prompt;
            PRAGMA user_version = 3;",
        )
        .unwrap();
        super::schema::init_schema(&conn).unwrap();
        super::schema::init_schema(&conn).unwrap();
        let detail = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert!(detail.run.prompt_overrides.is_empty());
        assert!(detail.tasks[0].custom_prompt.is_empty());
        assert_eq!(detail.run.sql_session_id, "sql-session");
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, super::schema::CURRENT_SCHEMA_VERSION);
    }

    #[test]
    fn dag_wave_assignment_orders_dependencies_and_detects_cycles() {
        let mut tasks = vec![
            dummy_task("sql1", "sql", vec![]),
            dummy_task("back1", "backend", vec!["sql1"]),
            dummy_task("front1", "frontend", vec!["back1"]),
            dummy_task("doc1", "general", vec![]),
        ];
        let waves = compute_and_assign_waves(&mut tasks).expect("valid DAG");
        assert_eq!(
            waves.get(&1).unwrap(),
            &vec!["sql1".to_string(), "doc1".to_string()]
        );
        assert_eq!(waves.get(&2).unwrap(), &vec!["back1".to_string()]);
        assert_eq!(waves.get(&3).unwrap(), &vec!["front1".to_string()]);

        let mut cyclic = vec![
            dummy_task("a", "backend", vec!["b"]),
            dummy_task("b", "frontend", vec!["a"]),
        ];
        assert!(compute_and_assign_waves(&mut cyclic).is_err());
    }

    #[test]
    fn cas_revision_and_attempt_id_fencing_prevent_stale_overwrites() {
        let mut conn = setup_mem_team_db();
        let run = TeamRun {
            id: "r1".into(),
            title: "RBAC Upgrade".into(),
            user_goal: "Add role table, API, and Vue page".into(),
            mode: "pipeline".into(),
            status: "draft".into(),
            stage: "awaiting_approval".into(),
            requires_db: true,
            execution_plan: String::new(),
            sql_schema_draft: String::new(),
            interface_spec: String::new(),
            delivery_sql: String::new(),
            sql_working_dir: String::new(),
            sql_agent_id: "none".to_string(),
            sql_model: String::new(),
            sql_session_id: String::new(),
            prompt_overrides: std::collections::BTreeMap::new(),
            goal_images: Vec::new(),
            revision: 1,
            auto_verify: true,
            max_repairs: 2,
            lead_model: String::new(),
            lead_agent_id: "none".to_string(),
            current_round_no: 1,
            source_triage_id: String::new(),
            synthesis_report: String::new(),
            created_at: "2026-11-18".into(),
            updated_at: "2026-11-18".into(),
            finished_at: String::new(),
        };
        let tasks = vec![dummy_task("t1", "sql", vec![])];
        create_run_with_tasks_conn(&mut conn, &run, &tasks).unwrap();

        let rev2 = update_run_draft_cas_conn(
            &mut conn,
            "r1",
            Some(1),
            "RBAC Upgrade v2",
            true,
            2,
            "",
            Some(true),
            &tasks,
            &RunDraftExtras::default(),
        )
        .unwrap();
        assert_eq!(rev2, 2);
        assert!(update_run_draft_cas_conn(
            &mut conn,
            "r1",
            Some(1),
            "Stale write",
            true,
            2,
            "",
            Some(true),
            &tasks,
            &RunDraftExtras::default(),
        )
        .is_err());

        let (_, att1) = claim_task_attempt_conn(&conn, "t1", "sess1", "hash1", "running").unwrap();
        let (_, att2) = claim_task_attempt_conn(&conn, "t1", "sess1", "hash1", "running").unwrap();
        assert_ne!(att1, att2);

        // 旧 attempt_id 回写必须被拒绝（返回 false）
        let stale_ok = finish_task_fenced_conn(
            &conn,
            "t1",
            &att1,
            "completed",
            "old output",
            None,
            "passed",
            "",
            0,
            &[],
            &[],
        )
        .unwrap();
        assert!(!stale_ok);

        // 最新 attempt_id 回写成功
        let fresh_ok = finish_task_fenced_conn(
            &conn,
            "t1",
            &att2,
            "completed",
            "new output",
            None,
            "passed",
            "ok",
            1,
            &["schema.sql".to_string()],
            &["schema.sql".to_string()],
        )
        .unwrap();
        assert!(fresh_ok);

        let _ = upsert_contract_conn(
            &conn,
            "r1",
            "t1",
            "sql",
            "sys_role DDL",
            "CREATE TABLE sys_role (...);",
            "",
        )
        .unwrap();
        let detail = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(detail.contracts.len(), 1);
        assert_eq!(detail.tasks[0].status, "completed");
        assert_eq!(detail.tasks[0].repair_count, 1);

        // 验证批量进度状态查询与补充指令写入
        let status_map = load_task_statuses_by_run_conn(&conn, &["r1".to_string()]).unwrap();
        assert_eq!(
            status_map.get("r1").unwrap(),
            &vec!["completed".to_string()]
        );

        append_task_note_conn(&conn, "r1", "t1", "请确保补充索引注释").unwrap();
        let detail_after_note = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert!(detail_after_note.tasks[0]
            .description
            .contains("请确保补充索引注释"));
    }

    #[test]
    fn append_tasks_mid_run_preserves_existing_progress_and_updates_spec() {
        let mut conn = setup_mem_team_db();
        let mut run = persistence_run();
        run.status = "running".into();
        run.stage = "executing".into();
        let mut t1 = dummy_task("t1", "backend", vec![]);
        t1.status = "completed".into();
        t1.session_id = "sess-be".into();
        t1.attempt = 1;
        t1.attempt_id = "att-be".into();
        t1.last_output_excerpt = "done api".into();
        let mut t2 = dummy_task("t2", "frontend", vec![]);
        t2.status = "running".into();
        t2.session_id = "sess-fe".into();
        t2.attempt = 1;
        t2.attempt_id = "att-fe".into();

        create_run_with_tasks_conn(&mut conn, &run, &[t1, t2]).unwrap();

        let mut t3 = dummy_task("t3", "mobile", vec!["t1"]);
        t3.project_name = "App".into();
        append_tasks_to_run_conn(
            &mut conn,
            "r1",
            &[t3],
            Some("## App Sync\nGET /api/app/badge"),
            None,
        )
        .unwrap();

        let detail = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(detail.run.status, "running");
        assert!(detail.run.interface_spec.contains("GET /api/app/badge"));
        assert_eq!(detail.tasks.len(), 3);
        let kept1 = detail.tasks.iter().find(|t| t.id == "t1").unwrap();
        let kept2 = detail.tasks.iter().find(|t| t.id == "t2").unwrap();
        let added = detail.tasks.iter().find(|t| t.id == "t3").unwrap();
        assert_eq!(kept1.status, "completed");
        assert_eq!(kept1.session_id, "sess-be");
        assert_eq!(kept2.status, "running");
        assert_eq!(kept2.attempt_id, "att-fe");
        assert_eq!(added.status, "pending");
        assert_eq!(added.wave, 2);
    }

    #[test]
    fn schedule_revision_supports_single_and_cascading_updates_with_images_and_archives_on_complete(
    ) {
        let mut conn = setup_mem_team_db();
        let mut run = persistence_run();
        run.status = "completed".into();
        run.stage = "completed".into();

        let mut be = dummy_task("t-be", "backend", vec![]);
        be.status = "completed".into();
        be.session_id = "sess-be".into();
        be.attempt = 1;

        let mut fe_a = dummy_task("t-fe-a", "frontend", vec![]);
        fe_a.status = "completed".into();
        fe_a.session_id = "sess-fea".into();
        fe_a.attempt = 1;

        let mut mob = dummy_task("t-mob", "mobile", vec![]);
        mob.status = "completed".into();
        mob.session_id = "sess-mob".into();
        mob.attempt = 1;

        create_run_with_tasks_conn(&mut conn, &run, &[be, fe_a, mob]).unwrap();

        // Cascading revision: backend changes API route -> frontend A syncs in Wave 2, mobile stays untouched
        let rev_img = TeamImage {
            media_type: "image/png".into(),
            data: "BASE64IMG".into(),
        };
        schedule_revision_conn(
            &mut conn,
            "r1",
            &[
                RevisionTaskTarget {
                    task_id: "t-be".into(),
                    note: "把路径改为 /api/system/announcements".into(),
                    images: vec![rev_img.clone()],
                    replace_images: true,
                    extra_depends_on: vec![],
                },
                RevisionTaskTarget {
                    task_id: "t-fe-a".into(),
                    note: "同步修改前端请求路径".into(),
                    images: vec![],
                    replace_images: false,
                    extra_depends_on: vec!["t-be".into()],
                },
            ],
            Some("Updated route: /api/system/announcements"),
            None,
            true,
            &RevisionRoundOptions::default(),
        )
        .unwrap();

        let detail = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(detail.run.status, "running");
        assert!(detail
            .run
            .interface_spec
            .contains("/api/system/announcements"));
        let be2 = detail.tasks.iter().find(|t| t.id == "t-be").unwrap();
        let fea2 = detail.tasks.iter().find(|t| t.id == "t-fe-a").unwrap();
        let mob2 = detail.tasks.iter().find(|t| t.id == "t-mob").unwrap();
        assert_eq!(be2.status, "pending");
        assert_eq!(be2.wave, 1);
        assert_eq!(be2.images.len(), 1);
        assert_eq!(be2.revision_note, "把路径改为 /api/system/announcements");
        assert_eq!(fea2.status, "pending");
        assert_eq!(fea2.wave, 2);
        assert!(fea2.depends_on.contains(&"t-be".to_string()));
        assert_eq!(mob2.status, "completed");

        // Completing the revised backend task clears revision_note and archives it into description
        let (_, att) = claim_task_attempt_conn(&conn, "t-be", "sess-be", "abc", "running").unwrap();
        assert!(finish_task_fenced_conn(
            &conn,
            "t-be",
            &att,
            "completed",
            "Updated routes",
            None,
            "passed",
            "ok",
            0,
            &["AnnouncementController.java".into()],
            &["AnnouncementController.java".into()],
        )
        .unwrap());
        let after = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        let be_done = after.tasks.iter().find(|t| t.id == "t-be").unwrap();
        assert!(be_done.revision_note.is_empty());
        assert!(be_done.description.contains("已完成改造"));
        assert!(be_done.description.contains("/api/system/announcements"));
    }

    #[test]
    fn update_run_delivery_appends_incremental_sql_without_overwriting_phase1() {
        let mut conn = setup_mem_team_db();
        let mut run = persistence_run();
        run.delivery_sql.clear();
        create_run_with_tasks_conn(&mut conn, &run, &[dummy_task("t1", "backend", vec![])])
            .unwrap();

        let phase1_sql = "CREATE TABLE sys_order (id BIGINT PRIMARY KEY, status VARCHAR(32));";
        update_run_delivery_conn(
            &conn,
            "r1",
            "completed",
            "completed",
            phase1_sql,
            "一期交付",
        )
        .unwrap();
        let r1 = load_run_conn(&conn, "r1").unwrap().unwrap();
        assert_eq!(r1.delivery_sql, phase1_sql);

        // 相同 SQL 或子串不重复追加
        update_run_delivery_conn(
            &conn,
            "r1",
            "completed",
            "completed",
            phase1_sql,
            "重复汇总",
        )
        .unwrap();
        let r1_dup = load_run_conn(&conn, "r1").unwrap().unwrap();
        assert_eq!(r1_dup.delivery_sql, phase1_sql);

        // 二期未提供新 SQL 时保留一期建表脚本
        update_run_delivery_conn(&conn, "r1", "completed", "completed", "", "仅改前端").unwrap();
        let r1_empty = load_run_conn(&conn, "r1").unwrap().unwrap();
        assert_eq!(r1_empty.delivery_sql, phase1_sql);

        // 二期增量改造 SQL 以分段注释追加在末尾，保留一期建表脚本
        let phase2_sql = "ALTER TABLE sys_order ADD COLUMN export_batch_id VARCHAR(64);";
        update_run_delivery_conn(
            &conn,
            "r1",
            "completed",
            "completed",
            phase2_sql,
            "二期交付",
        )
        .unwrap();
        let r2 = load_run_conn(&conn, "r1").unwrap().unwrap();
        assert!(r2.delivery_sql.starts_with(phase1_sql));
        assert!(r2.delivery_sql.contains("-- === 增量改造 SQL（"));
        assert!(r2.delivery_sql.ends_with(phase2_sql));
    }

    #[test]
    fn multi_round_snapshot_round_touched_files_and_segmented_delivery_sql() {
        let mut conn = setup_mem_team_db();
        let mut run = persistence_run();
        run.status = "completed".into();
        run.stage = "completed".into();
        run.delivery_sql = "CREATE TABLE sys_notice (id BIGINT PRIMARY KEY);".into();
        run.synthesis_report = "第 1 期交付总结".into();

        let mut be = dummy_task("t-be", "backend", vec![]);
        be.project_name = "tsm-server".into();
        be.status = "completed".into();
        be.touched_files = vec!["NoticeController.java".into()];
        be.round_touched_files = vec!["NoticeController.java".into()];

        let mut fe = dummy_task("t-fe", "frontend", vec![]);
        fe.project_name = "tsm-ui".into();
        fe.status = "completed".into();
        fe.touched_files = vec!["NoticeList.vue".into()];
        fe.round_touched_files = vec!["NoticeList.vue".into()];

        create_run_with_tasks_conn(&mut conn, &run, &[be, fe]).unwrap();
        upsert_contract_conn(
            &conn,
            "r1",
            "t-be",
            "backend",
            "Notice API v1",
            "GET /api/notice",
            "",
        )
        .unwrap();

        // 历史已完成单未写 team_rounds 时保持兼容（rounds 为空，currentRoundNo = 1）
        let legacy_detail = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert!(legacy_detail.rounds.is_empty());
        assert_eq!(legacy_detail.run.current_round_no, 1);

        // 发起第 2 期改造（仅选中后端，前端不参与），应先快照第 1 期再递增至第 2 期并清零后端 round_touched_files
        schedule_revision_conn(
            &mut conn,
            "r1",
            &[RevisionTaskTarget {
                task_id: "t-be".into(),
                note: "给公告表增加置顶字段并支持排序".into(),
                images: vec![],
                replace_images: false,
                extra_depends_on: vec![],
            }],
            Some("GET /api/notice 新增 isTop 字段"),
            None,
            true,
            &RevisionRoundOptions {
                round_requirement: Some("二期：公告支持置顶与按置顶排序".into()),
                round_title: Some("公告置顶迭代".into()),
                requires_sql: Some(true),
                sql_note: Some(
                    "```sql\nALTER TABLE sys_notice ADD COLUMN is_top INT DEFAULT 0;\n```".into(),
                ),
                ..Default::default()
            },
        )
        .unwrap();

        let mid = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(mid.run.current_round_no, 2);
        assert_eq!(mid.rounds.len(), 2);
        assert_eq!(mid.rounds[0].round_no, 1);
        assert_eq!(mid.rounds[0].status, "completed");
        assert!(mid.rounds[0].tasks.iter().all(|t| t.participated));
        assert_eq!(
            mid.rounds[0].delivery_sql,
            "CREATE TABLE sys_notice (id BIGINT PRIMARY KEY);"
        );
        assert_eq!(mid.rounds[1].round_no, 2);
        assert_eq!(mid.rounds[1].status, "running");
        assert_eq!(mid.rounds[1].title, "公告置顶迭代");
        assert!(mid.rounds[1].requires_db);

        let be_mid = mid.tasks.iter().find(|t| t.id == "t-be").unwrap();
        assert_eq!(
            be_mid.touched_files,
            vec!["NoticeController.java".to_string()]
        );
        assert!(be_mid.round_touched_files.is_empty());

        // 模拟第 2 期后端执行并完成
        let (_, att) =
            claim_task_attempt_conn(&conn, "t-be", "sess-be", "hash", "running").unwrap();
        assert!(finish_task_fenced_conn(
            &conn,
            "t-be",
            &att,
            "completed",
            "已完成置顶字段开发",
            None,
            "passed",
            "ok",
            0,
            &["NoticeController.java".into(), "NoticeMapper.xml".into()],
            &["NoticeMapper.xml".into()],
        )
        .unwrap());

        let after_tasks = super::store::load_tasks_for_run_conn(&conn, "r1").unwrap();
        let r2_snaps = vec![
            build_round_task_snapshot(
                &after_tasks[0],
                true,
                "给公告表增加置顶字段并支持排序",
                false,
            ),
            build_round_task_snapshot(&after_tasks[1], false, "", false),
        ];
        let r2 = TeamRound {
            round_no: 2,
            title: "公告置顶迭代".into(),
            requirement: "二期：公告支持置顶与按置顶排序".into(),
            status: "completed".into(),
            stage: "completed".into(),
            requires_db: true,
            execution_plan: String::new(),
            sql_schema_draft: "ALTER TABLE sys_notice ADD COLUMN is_top INT DEFAULT 0;".into(),
            interface_spec: "GET /api/notice 新增 isTop 字段".into(),
            delivery_sql: "ALTER TABLE sys_notice ADD COLUMN is_top INT DEFAULT 0;".into(),
            synthesis_report: "第 2 期交付完成".into(),
            tasks: r2_snaps,
            contracts: vec![],
            created_at: "2026-11-25 10:00:00".into(),
            finished_at: "2026-11-25 10:05:00".into(),
        };
        finalize_round_delivery_conn(&conn, "r1", "completed", "completed", &r2).unwrap();

        let done = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(done.rounds.len(), 2);
        assert_eq!(
            done.rounds[1].delivery_sql,
            "ALTER TABLE sys_notice ADD COLUMN is_top INT DEFAULT 0;"
        );
        let r2_be = done.rounds[1]
            .tasks
            .iter()
            .find(|t| t.id == "t-be")
            .unwrap();
        let r2_fe = done.rounds[1]
            .tasks
            .iter()
            .find(|t| t.id == "t-fe")
            .unwrap();
        assert!(r2_be.participated);
        assert_eq!(r2_be.touched_files, vec!["NoticeMapper.xml".to_string()]);
        assert!(!r2_fe.participated);
        assert!(r2_fe.touched_files.is_empty());

        // run.delivery_sql 按期号分段汇总
        assert!(done.run.delivery_sql.contains("-- === 第 1 期"));
        assert!(done.run.delivery_sql.contains("CREATE TABLE sys_notice"));
        assert!(done
            .run
            .delivery_sql
            .contains("-- === 第 2 期 · 公告置顶迭代 SQL ==="));
        assert!(done
            .run
            .delivery_sql
            .contains("ALTER TABLE sys_notice ADD COLUMN is_top"));

        // 校验 JSON camelCase 协议字段严格对齐
        let v = serde_json::to_value(&done).unwrap();
        assert_eq!(v["run"]["currentRoundNo"], 2);
        assert_eq!(v["run"]["leadAgentId"], "none");
        assert_eq!(v["rounds"][1]["roundNo"], 2);
        assert_eq!(v["rounds"][1]["requiresDb"], true);
        assert_eq!(v["rounds"][1]["tasks"][0]["projectName"], "tsm-server");
        assert_eq!(v["rounds"][1]["tasks"][0]["participated"], true);
        assert_eq!(v["rounds"][1]["tasks"][1]["participated"], false);
    }

    #[test]
    fn extract_delivery_sql_keeps_contract_sql_even_when_draft_exists() {
        let mut run = persistence_run();
        run.sql_schema_draft = "```sql\nCREATE TABLE sys_base (id BIGINT PRIMARY KEY);\n```".into();
        let contracts = vec![
            TeamContract {
                id: "c1".into(),
                run_id: "r1".into(),
                from_task_id: "t-sql".into(),
                domain: "sql".into(),
                title: "Extra Index".into(),
                schema_or_api: "CREATE INDEX idx_base_id ON sys_base(id);".into(),
                notes: String::new(),
                created_at: "2026-11-25 10:00:00".into(),
            },
            TeamContract {
                id: "c2".into(),
                run_id: "r1".into(),
                from_task_id: "t-be".into(),
                domain: "backend".into(),
                title: "Backend DDL".into(),
                schema_or_api:
                    "API /api/base\n```sql\nALTER TABLE sys_base ADD COLUMN code VARCHAR(32);\n```"
                        .into(),
                notes: String::new(),
                created_at: "2026-11-25 10:01:00".into(),
            },
        ];
        let merged = super::runner::extract_delivery_sql(&run, &contracts);
        assert!(merged.contains("CREATE TABLE sys_base"));
        assert!(merged.contains("CREATE INDEX idx_base_id"));
        assert!(merged.contains("ALTER TABLE sys_base ADD COLUMN code"));
    }

    #[test]
    fn lead_agent_config_run_persistence_and_v6_migration() {
        use super::store::*;
        let mut conn = setup_mem_team_db();
        let mut cfg = super::types::TeamGlobalConfig::default();
        assert_eq!(cfg.default_lead_agent_id, "none");
        cfg.default_lead_agent_id = "whale-lead".into();
        save_config_conn(&conn, &cfg).unwrap();
        let loaded_cfg = load_config_conn(&conn);
        assert_eq!(loaded_cfg.default_lead_agent_id, "whale-lead");
        assert_eq!(
            serde_json::to_value(&loaded_cfg).unwrap()["defaultLeadAgentId"],
            "whale-lead"
        );

        let mut run = persistence_run();
        run.lead_agent_id = "whale-lead".into();
        create_run_with_tasks_conn(&mut conn, &run, &[dummy_task("t1", "backend", vec![])])
            .unwrap();
        assert_eq!(
            load_run_conn(&conn, "r1").unwrap().unwrap().lead_agent_id,
            "whale-lead"
        );

        update_run_lead_agent_conn(&conn, "r1", "arch-lead").unwrap();
        assert_eq!(
            load_run_conn(&conn, "r1").unwrap().unwrap().lead_agent_id,
            "arch-lead"
        );

        // 校验 v5 -> v6 迁移幂等且不丢数据
        conn.execute_batch(
            "DROP TABLE team_rounds;
             ALTER TABLE team_runs DROP COLUMN current_round_no;
             ALTER TABLE team_runs DROP COLUMN lead_agent_id;
             ALTER TABLE team_tasks DROP COLUMN round_touched_files_json;
             PRAGMA user_version = 5;",
        )
        .unwrap();
        super::schema::init_schema(&conn).unwrap();
        super::schema::init_schema(&conn).unwrap();
        let migrated = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        assert_eq!(migrated.run.current_round_no, 1);
        assert_eq!(migrated.run.lead_agent_id, "none");
        assert!(migrated.rounds.is_empty());
        assert!(migrated.tasks[0].round_touched_files.is_empty());
    }

    #[test]
    fn revision_plan_requires_sql_heuristic_and_camel_case() {
        assert!(super::planner::revision_implies_sql(
            "给公告表新增置顶字段并加索引"
        ));
        assert!(!super::planner::revision_implies_sql(
            "调整弹窗按钮颜色和间距"
        ));
        let res = super::planner::RevisionPlanResult {
            cascading: true,
            summary: "需要改表与接口联动".into(),
            interface_spec_addition: String::new(),
            requires_sql: true,
            members: vec![],
        };
        assert_eq!(serde_json::to_value(&res).unwrap()["requiresSql"], true);
    }

    #[test]
    fn ask_peer_resolves_project_name_teammates_line_and_scans_cjk_and_extra_exts() {
        let mut be = dummy_task("tsk-be", "backend", vec![]);
        be.project_name = "tsm-server".into();
        let mut fe1 = dummy_task("tsk-fe1", "frontend", vec![]);
        fe1.project_name = "tsm-ui".into();
        let mut fe2 = dummy_task("tsk-fe2", "frontend", vec![]);
        fe2.project_name = "观民宗前端".into();
        let tasks = vec![be, fe1, fe2];

        // 多个前端时按 project_name 精准/模糊定位
        assert_eq!(
            super::tools::resolve_peer_task(&tasks, "tsk-be", "TSM-UI").map(|t| t.id.as_str()),
            Some("tsk-fe1")
        );
        assert_eq!(
            super::tools::resolve_peer_task(&tasks, "tsk-be", "观民宗").map(|t| t.id.as_str()),
            Some("tsk-fe2")
        );
        assert_eq!(
            super::tools::resolve_peer_task(&tasks, "tsk-fe1", "tsk-fe2").map(|t| t.id.as_str()),
            Some("tsk-fe2")
        );
        assert_eq!(
            super::tools::resolve_peer_task(&tasks, "tsk-fe1", "backend").map(|t| t.id.as_str()),
            Some("tsk-be")
        );

        let roster = super::prompts::format_teammates_line(&tasks, "tsk-be");
        assert!(roster.contains("tsk-fe1 | tsm-ui | frontend"));
        assert!(roster.contains("tsk-fe2 | 观民宗前端 | frontend"));
        assert!(!roster.contains("tsk-be"));

        // 中文关键词提取 + 扩展后缀扫描（MyBatis .xml、.wxml、.properties）
        let kws = super::tools::extract_peer_search_keywords(
            "观民宗",
            "sys_notice表的公告状态字段在Mapper里叫什么",
        );
        assert!(kws.iter().any(|k| k == "观民宗"));
        assert!(kws.iter().any(|k| k == "sys_notice"));
        assert!(kws.iter().any(|k| k.contains("公告状态")));

        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("NoticeMapper.xml"),
            "<mapper><select id=\"list\">SELECT 公告状态 FROM sys_notice</select></mapper>",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("notice.wxml"),
            "<view class=\"title\">观民宗公告</view>",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("application.properties"),
            "app.notice.title=观民宗系统\n",
        )
        .unwrap();
        let evidence = super::tools::scan_peer_repo_evidence(
            tmp.path(),
            "观民宗",
            "公告状态字段在Mapper里怎么查",
        );
        assert!(evidence.contains("NoticeMapper.xml"));
        assert!(evidence.contains("notice.wxml"));
        assert!(evidence.contains("application.properties"));
    }

    #[test]
    fn qa_tool_access_blocks_file_writes_but_allows_verification_commands() {
        use super::tools::team_qa_verify_tool;
        for allowed in [
            "read",
            "grep",
            "glob",
            "code_explore",
            "team_ask_peer",
            "memory_recall",
            "run_command",
            "todo_write",
        ] {
            assert!(
                team_qa_verify_tool(allowed),
                "expected QA tool `{allowed}` to be allowed"
            );
        }
        for denied in [
            "edit",
            "write",
            "write_file",
            "edit_file",
            "create_file",
            "team_publish_contract",
            "task",
        ] {
            assert!(
                !team_qa_verify_tool(denied),
                "expected QA tool `{denied}` to be denied"
            );
        }
    }

    #[test]
    fn qa_upstream_defect_routing_resets_only_responsible_specialists_and_preserves_sessions() {
        use super::prompts::{
            collect_upstream_ancestor_tasks, format_qa_upstream_roster, is_pipeline_qa_gate_task,
        };
        use super::runner::{resolve_qa_rework_targets, VerifierReworkTarget, VerifierVerdict};
        use super::store::{
            load_task_conn, promote_retest_ancestors_passed_conn, route_qa_rework_fenced_conn,
        };

        let mut conn = setup_mem_team_db();
        let run = persistence_run();

        let mut be = dummy_task("tsk_100_61d0c", "backend", vec![]);
        be.project_name = "tsm-server".into();
        be.working_dir = "D:/MZ/tsm-server".into();
        be.status = "completed".into();
        be.session_id = "sess_be_native".into();
        be.verify_verdict = "passed".into();
        be.touched_files = vec!["ReligionGroupVo.java".into()];

        let mut fe = dummy_task("tsk_100_97391", "frontend", vec![]);
        fe.project_name = "tsm-ui".into();
        fe.working_dir = "D:/MZ/tsm-ui".into();
        fe.status = "completed".into();
        fe.session_id = "sess_fe_native".into();
        fe.verify_verdict = "passed".into();
        fe.touched_files = vec!["src/utils/tsm.js".into()];

        let mut mob = dummy_task("tsk_100_mob01", "mobile", vec![]);
        mob.project_name = "tsm-mini".into();
        mob.working_dir = "D:/MZ/tsm-mini".into();
        mob.status = "completed".into();
        mob.session_id = "sess_mob_native".into();
        mob.verify_verdict = "passed".into();

        let mut qa = dummy_task(
            "tsk_200_bc69f",
            "qa",
            vec!["tsk_100_61d0c", "tsk_100_97391", "tsk_100_mob01"],
        );
        qa.title = "宗教团体负责人法名道名展示与导出回归验收".into();

        let tasks = vec![be.clone(), fe.clone(), mob.clone(), qa.clone()];
        create_run_with_tasks_conn(&mut conn, &run, &tasks).unwrap();
        assert!(is_pipeline_qa_gate_task("pipeline", &qa));
        assert!(!is_pipeline_qa_gate_task("triage", &qa));
        assert!(!is_pipeline_qa_gate_task("pipeline", &be));

        let ancestors = collect_upstream_ancestor_tasks(&tasks, &qa);
        assert_eq!(ancestors.len(), 3);
        let roster = format_qa_upstream_roster(&ancestors);
        assert!(roster.contains("tsk_100_61d0c"));
        assert!(roster.contains("D:/MZ/tsm-ui"));

        // 解析 Verifier 返回的 rework_targets：支持精确 ID、短后缀 #97391、自动忽略 QA 自身与非法目标并去重合并
        let verdict = VerifierVerdict {
            passed: false,
            summary: "发现前端 P1 阻断缺陷与后端 P2 脱敏一致性缺陷".into(),
            repair_instructions: String::new(),
            rework_targets: vec![
                VerifierReworkTarget {
                    task_id: "#97391".into(),
                    instructions: "修复 tsm.js formatLeaderWithTitle 姓名与法名一致去重逻辑".into(),
                },
                VerifierReworkTarget {
                    task_id: "tsk_100_61d0c".into(),
                    instructions: "补齐 ReligionGroupVo leaderReligiousTitle 脱敏注解".into(),
                },
                VerifierReworkTarget {
                    task_id: "tsk_200_bc69f".into(),
                    instructions: "不应该打回 QA 自己".into(),
                },
            ],
        };
        let resolved = resolve_qa_rework_targets(&verdict, &ancestors, "QA 验收结论：不通过");
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].task_id, "tsk_100_97391");
        assert_eq!(resolved[1].task_id, "tsk_100_61d0c");

        let (_, qa_attempt_id) =
            claim_task_attempt_conn(&conn, &qa.id, "sess_qa", "hqa", "running").unwrap();

        // 过期 attempt_id 栅栏必须拒绝写回
        let stale = route_qa_rework_fenced_conn(
            &mut conn,
            &run.id,
            &qa.id,
            "wrong_attempt",
            1,
            2,
            "QA 报告",
            &verdict.summary,
            &[],
            &[],
            &resolved,
        )
        .unwrap();
        assert!(!stale);

        // 有效 attempt_id 成功回流：仅命中的前后端进入 repairing/qa_rework 且保留原会话；未命中的 mobile 保持 completed；QA 回到 pending/routed_rework
        let ok = route_qa_rework_fenced_conn(
            &mut conn,
            &run.id,
            &qa.id,
            &qa_attempt_id,
            1,
            2,
            "QA 报告正文",
            &verdict.summary,
            &[],
            &[],
            &resolved,
        )
        .unwrap();
        assert!(ok);

        let be_after = load_task_conn(&conn, &be.id).unwrap().unwrap();
        assert_eq!(be_after.status, "repairing");
        assert_eq!(be_after.verify_verdict, "qa_rework");
        assert_eq!(be_after.repair_count, 1);
        assert_eq!(be_after.session_id, "sess_be_native");
        assert!(be_after.revision_note.contains("ReligionGroupVo"));

        let fe_after = load_task_conn(&conn, &fe.id).unwrap().unwrap();
        assert_eq!(fe_after.status, "repairing");
        assert_eq!(fe_after.verify_verdict, "qa_rework");
        assert_eq!(fe_after.repair_count, 1);
        assert_eq!(fe_after.session_id, "sess_fe_native");
        assert!(fe_after.revision_note.contains("formatLeaderWithTitle"));

        let mob_after = load_task_conn(&conn, &mob.id).unwrap().unwrap();
        assert_eq!(mob_after.status, "completed");
        assert_eq!(mob_after.verify_verdict, "passed");
        assert_eq!(mob_after.repair_count, 0);

        let qa_after = load_task_conn(&conn, &qa.id).unwrap().unwrap();
        assert_eq!(qa_after.status, "pending");
        assert_eq!(qa_after.verify_verdict, "routed_rework");
        assert_eq!(qa_after.repair_count, 1);
        assert!(qa_after.verify_feedback.contains("#97391"));
        assert!(qa_after.verify_feedback.contains("#61d0c"));

        let detail = load_run_detail_conn(&conn, &run.id, 0).unwrap().unwrap();
        assert!(
            detail
                .events
                .iter()
                .filter(|e| e.kind == "task_qa_rework_routed")
                .count()
                >= 3
        );

        // 模拟前后端返工完成（进入 retest 态），随后 QA 复测通过并统一将上游 retest 升级为 passed
        let (_, be_att2) =
            claim_task_attempt_conn(&conn, &be.id, "sess_be_native", "hbe", "repairing").unwrap();
        finish_task_fenced_conn(
            &conn,
            &be.id,
            &be_att2,
            "completed",
            "已补齐脱敏注解",
            None,
            "retest",
            "等待复测",
            1,
            &["ReligionGroupVo.java".into()],
            &["ReligionGroupVo.java".into()],
        )
        .unwrap();
        let (_, fe_att2) =
            claim_task_attempt_conn(&conn, &fe.id, "sess_fe_native", "hfe", "repairing").unwrap();
        finish_task_fenced_conn(
            &conn,
            &fe.id,
            &fe_att2,
            "completed",
            "已修复去重逻辑",
            None,
            "retest",
            "等待复测",
            1,
            &["src/utils/tsm.js".into()],
            &["src/utils/tsm.js".into()],
        )
        .unwrap();

        let promoted = promote_retest_ancestors_passed_conn(
            &conn,
            &run.id,
            &[be.id.clone(), fe.id.clone(), mob.id.clone()],
            "二轮回归全部通过",
        )
        .unwrap();
        assert_eq!(promoted, 2);
        assert_eq!(
            load_task_conn(&conn, &be.id)
                .unwrap()
                .unwrap()
                .verify_verdict,
            "passed"
        );
        assert_eq!(
            load_task_conn(&conn, &fe.id)
                .unwrap()
                .unwrap()
                .verify_verdict,
            "passed"
        );
    }

    #[test]
    fn in_place_rework_keeps_current_round_no_and_preserves_round_touched_files() {
        use super::store::{load_round_conn, load_task_conn};

        let mut conn = setup_mem_team_db();
        let mut run = persistence_run();
        run.status = "completed".into();
        run.stage = "completed".into();
        run.current_round_no = 2;

        let mut be = dummy_task("t_be", "backend", vec![]);
        be.status = "completed".into();
        be.session_id = "sess_be".into();
        be.touched_files = vec!["A.java".into(), "B.java".into()];
        be.round_touched_files = vec!["B.java".into()];

        let mut fe = dummy_task("t_fe", "frontend", vec![]);
        fe.status = "completed".into();
        fe.session_id = "sess_fe".into();
        fe.touched_files = vec!["page.vue".into()];
        fe.round_touched_files = vec!["page.vue".into()];

        create_run_with_tasks_conn(&mut conn, &run, &[be.clone(), fe.clone()]).unwrap();
        // 先落盘第 2 期已完成快照
        let r2_snap = TeamRound {
            round_no: 2,
            title: "第 2 期 · 追加需求".into(),
            requirement: "增加维护结束时间与 5 秒倒计时".into(),
            status: "completed".into(),
            stage: "completed".into(),
            requires_db: false,
            execution_plan: String::new(),
            sql_schema_draft: String::new(),
            interface_spec: String::new(),
            delivery_sql: "ALTER TABLE sys_announcement ADD maintenance_end_time TIMESTAMP;".into(),
            synthesis_report: "第 2 期首版交付总结".into(),
            tasks: vec![
                build_round_task_snapshot(&be, true, "后端支持维护结束时间", false),
                build_round_task_snapshot(&fe, true, "前端增加 5 秒倒计时", false),
            ],
            contracts: vec![],
            created_at: "2026-11-25 10:00:00".into(),
            finished_at: "2026-11-25 10:30:00".into(),
        };
        finalize_round_delivery_conn(&conn, &run.id, "completed", "completed", &r2_snap).unwrap();

        // 在第 2 期已交付状态下触发原地返工（in_place_rework = true）：不应升为第 3 期，且应保留第 2 期已改文件与原标题/需求
        schedule_revision_conn(
            &mut conn,
            &run.id,
            &[RevisionTaskTarget {
                task_id: fe.id.clone(),
                note: "修复 5 秒倒计时未结束时按钮仍可点击的 bug".into(),
                images: vec![],
                replace_images: false,
                extra_depends_on: vec![],
            }],
            None,
            None,
            true,
            &RevisionRoundOptions {
                round_requirement: Some("修复第 2 期倒计时按钮禁用逻辑".into()),
                round_title: None,
                requires_sql: Some(false),
                sql_note: None,
                in_place_rework: true,
            },
        )
        .unwrap();

        let run_after = load_run_conn(&conn, &run.id).unwrap().unwrap();
        assert_eq!(run_after.current_round_no, 2);
        assert_eq!(run_after.status, "running");

        let fe_after = load_task_conn(&conn, &fe.id).unwrap().unwrap();
        assert_eq!(fe_after.status, "pending");
        assert_eq!(fe_after.session_id, "sess_fe");
        assert_eq!(fe_after.round_touched_files, vec!["page.vue".to_string()]);
        assert!(fe_after.revision_note.contains("倒计时"));

        let be_after = load_task_conn(&conn, &be.id).unwrap().unwrap();
        assert_eq!(be_after.status, "completed");

        let round2 = load_round_conn(&conn, &run.id, 2).unwrap().unwrap();
        assert_eq!(round2.title, "第 2 期 · 追加需求");
        assert!(round2.requirement.contains("增加维护结束时间与 5 秒倒计时"));
        assert!(round2.requirement.contains("修复第 2 期倒计时按钮禁用逻辑"));
        assert!(load_round_conn(&conn, &run.id, 3).unwrap().is_none());
    }
}
