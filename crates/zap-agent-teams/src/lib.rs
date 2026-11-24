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
    TeamRun, TeamRunDetail, TeamTask,
};

#[cfg(test)]
mod tests {
    use super::store::{
        append_task_note_conn, append_tasks_to_run_conn, claim_task_attempt_conn,
        create_run_with_tasks_conn, finish_task_fenced_conn, load_run_detail_conn,
        load_task_statuses_by_run_conn, schedule_revision_conn, setup_mem_team_db,
        update_run_draft_cas_conn, upsert_contract_conn, RevisionTaskTarget, RunDraftExtras,
    };
    use super::types::{compute_and_assign_waves, TeamImage, TeamRun, TeamTask};

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
        )
        .unwrap());
        let after = load_run_detail_conn(&conn, "r1", 0).unwrap().unwrap();
        let be_done = after.tasks.iter().find(|t| t.id == "t-be").unwrap();
        assert!(be_done.revision_note.is_empty());
        assert!(be_done.description.contains("已完成改造"));
        assert!(be_done.description.contains("/api/system/announcements"));
    }
}
