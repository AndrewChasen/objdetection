use anyhow::Result;
use async_trait::async_trait;
use common::{AgentMetadata, ToolError, ToolInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tracing::info;

// 工具接口定义
#[async_trait]
pub trait Tool {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> JsonValue;
    async fn call(&self, params: JsonValue) -> Result<JsonValue, ToolError>;
}

// Agent构建器
pub struct AgentBuilder {
    name: String,
    tools: Vec<Box<dyn Tool + Send + Sync>>,
}

// Agent表示
pub struct Agent {
    metadata: AgentMetadata,
    tools: Vec<Box<dyn Tool + Send + Sync>>,
}

impl AgentBuilder {
    pub fn new(name: &str) -> Self {
        AgentBuilder {
            name: name.to_string(),
            tools: Vec::new(),
        }
    }

    pub fn register_tool<T: Tool + Send + Sync + 'static>(mut self, tool: T) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            metadata: AgentMetadata {
                name: self.name.clone(),
                description: format!("{} Agent", self.name),
                version: "1.0".to_string(),
            },
            tools: self.tools,
        }
    }
}

impl Agent {
    pub fn set_metadata(&mut self, metadata: AgentMetadata) {
        self.metadata = metadata;
    }

    pub fn get_tools_info(&self) -> Vec<ToolInfo> {
        self.tools
            .iter()
            .map(|tool| ToolInfo {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                agent: self.metadata.name.clone(),
                schema: tool.schema(),
            })
            .collect()
    }

    pub async fn call_tool(&self, tool_name: &str, params: JsonValue) -> Result<JsonValue, ToolError> {
        for tool in &self.tools {
            if tool.name() == tool_name {
                return tool.call(params).await;
            }
        }
        
        Err(ToolError {
            code: "ToolNotFound".to_string(),
            message: format!("找不到工具: {}", tool_name),
            details: None,
        })
    }

    // 示例实现，实际会使用MCP SDK的serve方法
    pub async fn serve(&self, _transport: &str) -> Result<()> {
        info!("Agent启动: {}", self.metadata.name);
        // 在实际实现中，这里会启动服务并处理传输层
        Ok(())
    }
}

// VPN工具实现示例
#[derive(Debug)]
pub struct FindWindow;

impl FindWindow {
    pub fn new() -> Self {
        FindWindow
    }
}

#[derive(Debug, Deserialize)]
pub struct FindParams {
    pub title: String,
    pub trace_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WindowResult {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[async_trait]
impl Tool for FindWindow {
    fn name(&self) -> &str {
        "FindWindow"
    }
    
    fn description(&self) -> &str {
        "查找和定位屏幕上的VPN窗口"
    }
    
    fn schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "trace_id": {"type": "string"}
            },
            "required": ["title"]
        })
    }
    
    async fn call(&self, params: JsonValue) -> Result<JsonValue, ToolError> {
        let find_params: FindParams = serde_json::from_value(params)
            .map_err(|e| ToolError {
                code: "InvalidParams".to_string(),
                message: e.to_string(),
                details: None,
            })?;
            
        info!("查找窗口: {}", find_params.title);
        
        // 模拟找到窗口
        let result = WindowResult {
            x: 100,
            y: 200,
            w: 800,
            h: 600,
        };
        
        Ok(serde_json::to_value(result).unwrap())
    }
}

// 另一个VPN工具示例
#[derive(Debug)]
pub struct ActivateWindow;

impl ActivateWindow {
    pub fn new() -> Self {
        ActivateWindow
    }
}

#[derive(Debug, Deserialize)]
pub struct ActivateParams {
    pub x: i32,
    pub y: i32,
    pub trace_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ActivateResult {
    pub success: bool,
}

#[async_trait]
impl Tool for ActivateWindow {
    fn name(&self) -> &str {
        "ActivateWindow"
    }
    
    fn description(&self) -> &str {
        "激活指定的VPN窗口"
    }
    
    fn schema(&self) -> JsonValue {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "trace_id": {"type": "string"}
            },
            "required": ["x", "y"]
        })
    }
    
    async fn call(&self, params: JsonValue) -> Result<JsonValue, ToolError> {
        let activate_params: ActivateParams = serde_json::from_value(params)
            .map_err(|e| ToolError {
                code: "InvalidParams".to_string(),
                message: e.to_string(),
                details: None,
            })?;
            
        info!("激活窗口: ({}, {})", activate_params.x, activate_params.y);
        
        // 模拟激活窗口
        let result = ActivateResult {
            success: true,
        };
        
        Ok(serde_json::to_value(result).unwrap())
    }
}

// 示例：创建VPN Agent
pub fn create_vpn_agent() -> Agent {
    AgentBuilder::new("VPNAgent")
        .register_tool(FindWindow::new())
        .register_tool(ActivateWindow::new())
        .build()
}

// 如果需要，可以为其他类型的Agent创建类似的工厂函数

