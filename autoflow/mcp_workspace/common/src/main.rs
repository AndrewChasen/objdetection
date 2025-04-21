use config::{Config, File, Environment};
use tracing::{info, instrument};
use tracing_subscriber;
use anyhow::Result;
use uuid::Uuid;

#[instrument(name = "init", level = "debug", skip_all)]
#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志系统
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .init();

    // 加载配置并支持环境变量多级覆盖
    let cfg = Config::builder()
        .add_source(File::with_name("config"))
        .add_source(Environment::with_prefix("MCP").separator("__"))
        .build()
        .map_err(|e| {
            tracing::error!("配置加载失败: {}", e);
            anyhow::anyhow!("配置加载失败: {}", e)
        })?;
    let trace_id = Uuid::new_v4();
    info!(%trace_id, "配置加载完成: {:#?}", cfg);

    // 示例错误分类处理
    if cfg.get::<String>("meta_agent.transport").is_err() {
        tracing::warn!(%trace_id, "transport 未配置，使用默认 stdio");
    }

    Ok(())
}
