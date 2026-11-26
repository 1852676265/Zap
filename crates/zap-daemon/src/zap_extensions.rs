//! Zap 扩展模块薄适配层：将独立 crate `zap-neural-memory` 与 `zap-agent-teams`
//! 接入 `zap-daemon` 主生命周期，避免主干代码与扩展实现发生耦合冲突。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use zap_agent_teams::{HostChildTurnRequest, HostModelInfo, HostProjectInfo, TeamHost};
use zap_coding::CodingRuntimeConfig;
use zap_kernel::message::Message;
use zap_neural_memory::NeuralMemoryBinding;
use zap_telemetry::{CurrentContext, SessionMode};

use crate::{AppState, ChatAdmissionMode, ChatEvent, ChatRequest};

struct DaemonTeamHost(AppState);

#[async_trait::async_trait]
impl TeamHost for DaemonTeamHost {
    async fn current_working_dir(&self) -> PathBuf {
        self.0.project.read().await.working_dir.clone()
    }

    fn list_projects(&self) -> Vec<HostProjectInfo> {
        crate::list_projects()
            .unwrap_or_default()
            .into_iter()
            .map(|p| HostProjectInfo {
                hash: p.hash,
                name: p.name,
                working_dir: p.working_dir,
                session_count: p.session_count,
            })
            .collect()
    }

    fn hash_path(&self, path: &Path) -> String {
        crate::hash_path(path)
    }

    fn list_models(&self) -> (String, Vec<HostModelInfo>) {
        match crate::api_config::load_config() {
            Ok(c) => {
                let resp = crate::api_config::config_response(&c);
                let list = resp
                    .providers
                    .into_iter()
                    .map(|p| HostModelInfo {
                        name: p.name,
                        model: p.model,
                        is_default: p.is_default,
                    })
                    .collect();
                (resp.default_provider, list)
            }
            Err(_) => (String::new(), Vec::new()),
        }
    }

    async fn run_child_turn(
        &self,
        req: HostChildTurnRequest,
        cancel: CancellationToken,
    ) -> Result<(String, Vec<String>), String> {
        let req_images: Vec<crate::ImageInput> = req
            .images
            .iter()
            .map(|(media_type, data)| crate::ImageInput {
                media_type: media_type.clone(),
                data: data.clone(),
            })
            .collect();

        let mut admitted_opt = None;
        for attempt_idx in 0..8 {
            let chat_req = ChatRequest {
                message: req.message.clone(),
                working_dir: Some(req.working_dir.clone()),
                provider: req.provider.clone(),
                session_id: Some(req.session_id.clone()),
                request_id: Some(format!(
                    "req_team_{}_{}",
                    req.session_id,
                    uuid::Uuid::new_v4().simple()
                )),
                images: req_images.clone(),
                approval_mode: Some(crate::approval_mode::ApprovalMode::Auto),
                extra_system_append: req.extra_system_append.clone(),
                session_title: req.session_title.clone(),
                is_scheduled: false,
                is_team: true,
            };
            match crate::admit_chat_turn(&self.0, chat_req, ChatAdmissionMode::RejectIfBusy, None)
                .await
            {
                Ok(admitted) => {
                    admitted_opt = Some(admitted);
                    break;
                }
                Err(crate::ActiveChatAdmissionError::SessionBusy) if attempt_idx < 7 => {
                    if cancel.is_cancelled() {
                        return Err("已取消或暂停".to_string());
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                Err(e) => {
                    return Err(format!("无法启动子智能体会话 `{}`: {e:?}", req.session_id));
                }
            }
        }
        let admitted = admitted_opt
            .ok_or_else(|| format!("无法启动子智能体会话 `{}`: 会话持续占用中", req.session_id))?;

        let crate::AdmittedChatTurn {
            request: admitted_req,
            operation_id,
            cancellation: cancel_token,
            mut rx,
            fan_tx,
            terminal_sent,
            ..
        } = admitted;

        let cancel_watch = {
            let ext_cancel = cancel.clone();
            let inner_cancel = cancel_token.clone();
            tokio::spawn(async move {
                ext_cancel.cancelled().await;
                inner_cancel.cancel();
            })
        };

        let cleanup_chats = self.0.active_chats.clone();
        let cleanup_op = operation_id.clone();
        let mcp_pool = self.0.mcp_pool.clone();
        let telemetry = self.0.telemetry.clone();
        let pending_permissions = self.0.pending_permissions.clone();
        let pending_user_inputs = self.0.pending_user_inputs.clone();
        let op_for_proc = operation_id.clone();
        let cancel_for_proc = cancel_token.clone();
        let active_chats_for_proc = self.0.active_chats.clone();

        let ctx = CurrentContext {
            mode: Some(SessionMode::Channel),
            session_id: None,
            ..CurrentContext::current()
        };

        let (proc_err_tx, mut proc_err_rx) = tokio::sync::oneshot::channel::<String>();

        tokio::spawn(async move {
            let res = CurrentContext::scope(ctx, || async move {
                crate::process_chat_request(
                    admitted_req,
                    fan_tx,
                    cancel_for_proc,
                    op_for_proc,
                    active_chats_for_proc,
                    mcp_pool,
                    telemetry,
                    pending_permissions,
                    pending_user_inputs,
                    false,
                    false,
                    terminal_sent,
                    true,
                    Some(crate::approval_mode::ApprovalMode::Auto),
                )
                .await
            })
            .await;
            if let Err(e) = res {
                tracing::warn!("[zap-agent-teams] child session turn error: {e}");
                let _ = proc_err_tx.send(e.to_string());
            }
            cleanup_chats.complete(&cleanup_op).await;
        });

        let mut collected_text = String::new();
        let mut touched_files = Vec::new();
        let mut turn_err: Option<String> = None;

        while let Some(evt) = rx.recv().await {
            match evt {
                ChatEvent::TextDelta { content, .. } => {
                    collected_text.push_str(&content);
                }
                ChatEvent::ToolCallStarted {
                    name, arguments, ..
                } => {
                    if matches!(name.as_str(), "edit" | "write") {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&arguments) {
                            if let Some(p) = val.get("path").and_then(|v| v.as_str()) {
                                let clean = p.trim().replace('\\', "/");
                                if !clean.is_empty() && !touched_files.contains(&clean) {
                                    touched_files.push(clean);
                                }
                            }
                        }
                    }
                }
                ChatEvent::Done { message, .. } => {
                    if collected_text.trim().is_empty() {
                        if let Some(m) = message {
                            collected_text = m;
                        }
                    }
                    break;
                }
                ChatEvent::Error { message, .. } => {
                    turn_err = Some(message);
                    break;
                }
                ChatEvent::Stopped => {
                    if cancel.is_cancelled() {
                        cancel_watch.abort();
                        return Err("已取消或暂停".to_string());
                    }
                    break;
                }
                _ => {}
            }
        }

        cancel_watch.abort();

        if turn_err.is_none() {
            if let Ok(err) = proc_err_rx.try_recv() {
                turn_err = Some(err);
            }
        }

        if let Some(err) = turn_err {
            if collected_text.trim().is_empty() {
                return Err(err);
            }
        }
        Ok((collected_text, touched_files))
    }
}

pub(crate) async fn apply_chat_extensions(
    runtime_cfg: &mut CodingRuntimeConfig,
    session_id: Option<&str>,
) {
    let _ = zap_neural_memory::apply_session_static_system_append(runtime_cfg, session_id).await;
    zap_agent_teams::apply_session_team_tools(runtime_cfg, session_id);
}

pub(crate) async fn apply_live_extensions(
    runtime_cfg: &mut CodingRuntimeConfig,
    session_id: Option<&str>,
) -> NeuralMemoryBinding {
    zap_neural_memory::apply_session_static_system_append(runtime_cfg, session_id).await
}

pub(crate) async fn note_fresh_session(session_id: &str, working_dir: &Path) {
    zap_neural_memory::note_fresh_session_static_agent(session_id, working_dir).await;
}

pub(crate) fn on_turn_snapshot(
    session_id: &str,
    working_dir: &Path,
    messages: &[Message],
    provider_name: &str,
) {
    zap_neural_memory::maybe_capture_turn_from_snapshot(
        session_id,
        working_dir,
        messages,
        provider_name,
    );
}

pub(crate) fn on_pre_compact_flush(session_id: &str) {
    zap_neural_memory::flush_session_drafts_async(session_id, None);
}

pub(crate) async fn handle_teams_rpc(
    state: AppState,
    action: String,
    payload: Value,
) -> axum::response::Response {
    let host: Arc<dyn TeamHost> = Arc::new(DaemonTeamHost(state));
    match zap_agent_teams::handle_rpc(host, action.as_str(), payload).await {
        Ok(val) => Json(val).into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": err })),
        )
            .into_response(),
    }
}

pub(crate) fn handle_neural_rpc(action: &str, payload: &Value) -> axum::response::Response {
    match zap_neural_memory::handle_rpc(action, payload) {
        Ok(val) => Json(val).into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": err })),
        )
            .into_response(),
    }
}
