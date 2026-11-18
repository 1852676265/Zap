//! `zap-agent-teams` 跨项目多智能体协作常驻引擎（方案 B：外部插件 UI + Rust 守护进程常驻编排）。
//!
//! 职责模块拆分：
//! - `types`：领域模型、乐观锁版本号、attempt_id 栅栏与 Kahn DAG 波次拓扑切分
//! - `schema`：SQLite WAL 账本建表与索引迁移
//! - `store`：持久化连接池、CAS 草稿保存、attempt 栅栏提交与重启孤儿自愈
//! - `tools`：团队子会话按需挂载工具（`team_publish_contract` 契约发布 + `team_ask_peer` 跨仓库只读问询）
//! - `planner`：AI 主控拆单规划、默认项目与专属记忆智能体智能匹配、联合会诊转修复流水线
//! - `runner`：波次并发调度、独立验收闸门、最多 `max_repairs` 次自动打回返工与跨库会诊报告汇总
//! - `rpc`：面向外部插件 `@zap/agent-teams` 的 `teams:*` 统一 RPC 路由

mod planner;
mod rpc;
mod runner;
mod schema;
mod store;
mod tools;
mod types;

pub use rpc::handle_rpc;
pub use tools::apply_session_team_tools;
pub use types::{
    compute_and_assign_waves, RoleBindingPreset, TeamContract, TeamEvent, TeamGlobalConfig,
    TeamRun, TeamRunDetail, TeamTask,
};

#[cfg(test)]
mod tests {
    use super::store::{
        claim_task_attempt_conn, create_run_with_tasks_conn, finish_task_fenced_conn,
        load_run_detail_conn, setup_mem_team_db, update_run_draft_cas_conn, upsert_contract_conn,
    };
    use super::types::{compute_and_assign_waves, TeamRun, TeamTask};

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
            working_dir: "D:/Projects/Zap".to_string(),
            project_hash: "abc".to_string(),
            agent_id: "backend".to_string(),
            agent_label: "⚙ Backend".to_string(),
            model: String::new(),
            session_id: String::new(),
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
            &tasks,
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
            &tasks
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
    }
}
