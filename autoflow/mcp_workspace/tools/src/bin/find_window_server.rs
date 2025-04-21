use anyhow::Result;
use tools::FindWindowServer;
use tools::ToolServer;
use tracing::{info, instrument};
use tracing_subscriber;

#[instrument(name = "find_window_server", level = "debug", skip_all)]
#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志系统
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .init();

    info!("启动查找窗口工具服务器");
    
    // 创建工具服务器
    let server = FindWindowServer::new();
    
    // 输出工具信息
    info!("工具名称: {}", server.name());
    info!("工具描述: {}", server.description());
    info!("参数模式: {}", server.schema());
    
    // 启动服务
    server.serve().await?;
    
    // 在实际实现中，这里应该有阻塞逻辑
    info!("服务器启动完成，按Ctrl+C退出");
    tokio::signal::ctrl_c().await?;
    info!("服务器停止");
    
    Ok(())
} 