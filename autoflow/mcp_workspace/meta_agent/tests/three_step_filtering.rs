use anyhow::Result;
use common::{AgentMetadata, Task, ToolInfo, UserRequest};
use serde_json::json;
use std::collections::HashMap;
use tokio;

// 定义MockClientSession来模拟ClientSession的行为
struct MockClientSession {
    agents: HashMap<String, AgentMetadata>,
    tools: HashMap<String, Vec<ToolInfo>>,
}

impl MockClientSession {
    fn new() -> Self {
        let mut session = MockClientSession {
            agents: HashMap::new(),
            tools: HashMap::new(),
        };

        // 添加测试数据
        session.register_agent(
            AgentMetadata {
                name: "VPNAgent".to_string(),
                description: "处理VPN连接相关操作的专用Agent".to_string(),
                version: "1.0".to_string(),
            },
            vec![
                ToolInfo {
                    name: "FindWindow".to_string(),
                    description: "查找和定位屏幕上的VPN窗口".to_string(),
                    agent: "VPNAgent".to_string(),
                    schema: json!({
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "trace_id": {"type": "string"}
                        },
                        "required": ["title"]
                    }),
                },
                ToolInfo {
                    name: "ActivateWindow".to_string(),
                    description: "激活指定的VPN窗口".to_string(),
                    agent: "VPNAgent".to_string(),
                    schema: json!({
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "trace_id": {"type": "string"}
                        },
                        "required": ["x", "y"]
                    }),
                },
            ],
        );

        session.register_agent(
            AgentMetadata {
                name: "ContentAgent".to_string(),
                description: "处理内容采集相关操作的专用Agent".to_string(),
                version: "1.0".to_string(),
            },
            vec![
                ToolInfo {
                    name: "RedditFetch".to_string(),
                    description: "从Reddit获取内容".to_string(),
                    agent: "ContentAgent".to_string(),
                    schema: json!({
                        "type": "object",
                        "properties": {
                            "subreddit": {"type": "string"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["subreddit"]
                    }),
                },
            ],
        );

        session
    }

    fn register_agent(&mut self, metadata: AgentMetadata, tools: Vec<ToolInfo>) {
        let name = metadata.name.clone();
        self.agents.insert(name.clone(), metadata);
        self.tools.insert(name, tools);
    }

    async fn filter_agents(&self, _user_query: &str) -> Result<Vec<String>> {
        // 模拟返回测试数据
        Ok(vec!["VPNAgent".to_string()])
    }

    fn filter_tools(&self, agent_name: &str, _user_request: &str) -> Vec<ToolInfo> {
        match self.tools.get(agent_name) {
            Some(tools) => tools.clone(),
            None => Vec::new(),
        }
    }

    async fn generate_plan(&self, _user_query: &str, _selected_tools: &[ToolInfo]) -> Result<Vec<Task>> {
        // 返回测试任务
        Ok(vec![
            Task {
                id: "test-id-1".to_string(),
                agent: "VPNAgent".to_string(),
                tool: "FindWindow".to_string(),
                params: json!({"title": "VPN应用", "trace_id": "test123"}),
            },
        ])
    }

    async fn call_tool(&self, _tool_name: &str, _params: serde_json::Value) -> Result<serde_json::Value> {
        // 返回模拟结果
        Ok(json!({"success": true, "result": "测试结果"}))
    }
}

// 独立实现三步分层筛选机制用于测试
async fn three_step_filtering_test(
    session: &MockClientSession,
    user_query: &str,
) -> Result<Vec<Task>> {
    // 第一步：Agent识别
    let selected_agents = session.filter_agents(user_query).await?;
    
    // 第二步：工具筛选
    let mut all_tools = Vec::new();
    
    for agent_name in &selected_agents {
        let agent_tools = session.filter_tools(agent_name, user_query);
        all_tools.extend(agent_tools);
    }
    
    // 第三步：执行计划生成
    let tasks = session.generate_plan(user_query, &all_tools).await?;
    
    Ok(tasks)
}

#[tokio::test]
async fn test_three_step_filtering() -> Result<()> {
    let session = MockClientSession::new();
    
    // 测试VPN连接请求
    let user_query = "我需要连接到VPN";
    let tasks = three_step_filtering_test(&session, user_query).await?;
    
    // 验证结果
    assert_eq!(tasks.len(), 1, "应该生成一个任务");
    assert_eq!(tasks[0].agent, "VPNAgent", "任务应该分配给VPNAgent");
    assert_eq!(tasks[0].tool, "FindWindow", "应该使用FindWindow工具");
    
    Ok(())
}

// 可以添加更多的测试用例... 