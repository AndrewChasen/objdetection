use agents::create_vpn_agent;
use anyhow::Result;
use common::AgentMetadata;
use config::{Config, File, Environment};
use tracing::{info, instrument};
use tracing_subscriber;

#[instrument(name = "vpn_agent", level = "debug", skip_all)]
#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志系统
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .init();

    // 加载配置
    let cfg = Config::builder()
        .add_source(File::with_name("config"))
        .add_source(Environment::with_prefix("MCP").separator("__"))
        .build()
        .map_err(|e| {
            tracing::error!("配置加载失败: {}", e);
            anyhow::anyhow!("配置加载失败: {}", e)
        })?;
    
    // 检查配置项
    let transport = match cfg.get::<String>("agents.vpn.transport") {
        Ok(t) => t,
        Err(_) => {
            tracing::warn!("VPN Agent transport 未配置，使用默认 stdio");
            "stdio".to_string()
        }
    };
    info!("VPN Agent 使用传输方式: {}", transport);
    
    // 创建VPN Agent
    let mut vpn_agent = create_vpn_agent();
    
    // 设置元数据
    vpn_agent.set_metadata(AgentMetadata {
        name: "VPNAgent".to_string(),
        description: "处理VPN连接相关操作的专用Agent".to_string(),
        version: "1.0".to_string(),
    });
    
    // 输出可用工具信息
    for tool in vpn_agent.get_tools_info() {
        info!("工具: {} - {}", tool.name, tool.description);
    }
    
    // 启动服务
    vpn_agent.serve(&transport).await?;
    
    Ok(())
} 