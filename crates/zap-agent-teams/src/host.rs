//! 宿主运行时抽象契约（`TeamHost`），让 `zap-agent-teams` 与 `zap-daemon` 内部状态彻底解耦。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use zap_coding::{CodingProviderFactory, DefaultCodingProviderFactory};
use zap_config::Config;

#[derive(Clone, Debug)]
pub struct HostProjectInfo {
    pub hash: String,
    pub name: String,
    pub working_dir: PathBuf,
    pub session_count: usize,
}

#[derive(Clone, Debug)]
pub struct HostModelInfo {
    pub name: String,
    pub model: String,
    pub is_default: bool,
}

#[derive(Clone, Debug)]
pub struct HostChildTurnRequest {
    pub message: String,
    pub working_dir: PathBuf,
    pub provider: Option<String>,
    pub session_id: String,
    pub images: Vec<(String, String)>,
    pub extra_system_append: Option<String>,
    pub session_title: Option<String>,
}

#[async_trait::async_trait]
pub trait TeamHost: Send + Sync + 'static {
    async fn current_working_dir(&self) -> PathBuf;
    fn list_projects(&self) -> Vec<HostProjectInfo>;
    fn hash_path(&self, path: &Path) -> String;
    fn list_models(&self) -> (String, Vec<HostModelInfo>);
    async fn run_child_turn(
        &self,
        req: HostChildTurnRequest,
        cancel: CancellationToken,
    ) -> Result<(String, Vec<String>), String>;
}

pub(crate) fn load_config() -> Result<Config, String> {
    let path = Config::default_path();
    match Config::load(&path) {
        Ok(config) => Ok(config),
        Err(e) => {
            let is_missing = e.chain().any(|cause| {
                cause
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
            });
            if is_missing {
                Ok(Config::default())
            } else {
                Err(format!("Failed to load config: {:#}", e))
            }
        }
    }
}

pub(crate) fn coding_provider_factory() -> Arc<dyn CodingProviderFactory> {
    Arc::new(DefaultCodingProviderFactory::new(
        zap_auth::JEIKCODE_USER_AGENT,
    ))
}
