//! 外部扩展插件通用 RPC 路由服务。
//!
//! 为神经记忆引擎（`neural:*`）与外部插件沙箱存储（`storage:*`）提供统一 RPC 分发。

use crate::plugins_api::get_plugin_storage_file;
use crate::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
pub struct PluginRpcRequest {
    pub action: String,
    #[serde(default)]
    pub payload: Value,
}

/// POST /api/plugins-rpc/:plugin_id
///
/// 外部插件通用 RPC 路由器（统一调度至 `spawn_blocking`，杜绝磁盘读写与 SQLite 锁阻塞 Tokio 异步工作线程）
pub async fn handle_plugin_rpc(
    State(state): State<AppState>,
    Path(plugin_id): Path<String>,
    Json(req): Json<PluginRpcRequest>,
) -> axum::response::Response {
    if req.action.starts_with("teams:") {
        return crate::zap_extensions::handle_teams_rpc(state, req.action, req.payload).await;
    }
    match tokio::task::spawn_blocking(move || handle_plugin_rpc_sync(plugin_id, req)).await {
        Ok(resp) => resp,
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "success": false, "error": format!("RPC task error: {err}") })),
        )
            .into_response(),
    }
}

fn handle_plugin_rpc_sync(plugin_id: String, req: PluginRpcRequest) -> axum::response::Response {
    let action = req.action.as_str();
    let payload = req.payload;

    match action {
        a if a.starts_with("neural:") => crate::zap_extensions::handle_neural_rpc(a, &payload),

        "storage:get" => {
            let key = payload.get("key").and_then(|v| v.as_str()).unwrap_or("");
            let p = get_plugin_storage_file(&plugin_id);
            if !p.is_file() {
                return Json(json!({ "success": true, "value": Value::Null })).into_response();
            }
            if let Ok(s) = std::fs::read_to_string(&p) {
                if let Ok(map) = serde_json::from_str::<Value>(&s) {
                    let val = map.get(key).cloned().unwrap_or(Value::Null);
                    return Json(json!({ "success": true, "value": val })).into_response();
                }
            }
            Json(json!({ "success": true, "value": Value::Null })).into_response()
        }

        "storage:set" => {
            let key = payload.get("key").and_then(|v| v.as_str()).unwrap_or("");
            let val = payload.get("value").cloned().unwrap_or(Value::Null);
            let p = get_plugin_storage_file(&plugin_id);
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut map = if p.is_file() {
                std::fs::read_to_string(&p)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Value>(&s).ok())
                    .unwrap_or_else(|| json!({}))
            } else {
                json!({})
            };
            if !map.is_object() {
                map = json!({});
            }
            map[key] = val;
            if let Ok(s) = serde_json::to_string_pretty(&map) {
                let _ = std::fs::write(&p, s);
            }
            Json(json!({ "success": true })).into_response()
        }

        _ => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "success": false,
                "error": format!("Unsupported plugin RPC action: {action}"),
                "unhandledAction": action,
                "pluginId": plugin_id,
            })),
        )
            .into_response(),
    }
}
