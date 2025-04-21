use anyhow::Result;
use common::{AgentMetadata, Task, ToolInfo, UserRequest};
use serde_json::json;
use std::collections::HashMap;
use tokio;

// 定义MockClientSession
struct MockClientSession {
    agents: HashMap<String, AgentMetadata>,
    tools: HashMap<String, Vec<ToolInfo>>,
    responses: HashMap<String, serde_json::Value>,
}

impl MockClientSession {
    fn new() -> Self {
        let mut session = MockClientSession {
            agents: HashMap::new(),
            tools: HashMap::new(),
            responses: HashMap::new(),
        };

        // 配置模拟响应
        session.responses.insert(
            "VPNAgent.FindWindow".to_string(),
            json!({"success": true, "result": "找到VPN窗口"}),
        );
        session.responses.insert(
            "VPNAgent.ActivateWindow".to_string(),
            json!({"success": true, "result": "激活VPN窗口"}),
        );
        session.responses.insert(
            "VPNAgent.ConnectVPN".to_string(),
            json!({"success": false, "error": {"code": "AlreadyConnected", "message": "VPN已连接"}}),
        );

        session
    }

    // 模拟filter_agents函数
    async fn filter_agents(&self, _user_query: &str) -> Result<Vec<String>> {
        Ok(vec!["VPNAgent".to_string()])
    }

    // 模拟filter_tools函数
    fn filter_tools(&self, _agent_name: &str, _user_request: &str) -> Vec<ToolInfo> {
        vec![
            ToolInfo {
                name: "FindWindow".to_string(),
                description: "查找VPN窗口".to_string(),
                agent: "VPNAgent".to_string(),
                schema: json!({}),
            },
            ToolInfo {
                name: "ConnectVPN".to_string(),
                description: "连接VPN".to_string(),
                agent: "VPNAgent".to_string(),
                schema: json!({}),
            },
        ]
    }

    // 模拟generate_plan函数
    async fn generate_plan(&self, _user_query: &str, _selected_tools: &[ToolInfo]) -> Result<Vec<Task>> {
        Ok(vec![
            Task {
                id: "task-1".to_string(),
                agent: "VPNAgent".to_string(),
                tool: "FindWindow".to_string(),
                params: json!({"title": "VPN应用"}),
            },
            Task {
                id: "task-2".to_string(),
                agent: "VPNAgent".to_string(),
                tool: "ConnectVPN".to_string(),
                params: json!({"country": "japan"}),
            },
        ])
    }

    // 模拟call_tool函数
    async fn call_tool(&self, tool_name: &str, _params: serde_json::Value) -> Result<serde_json::Value> {
        match self.responses.get(tool_name) {
            Some(response) => Ok(response.clone()),
            None => Ok(json!({"success": true, "result": "默认响应"})),
        }
    }
}

// 从main.rs复制的handle_user函数，为测试目的进行了简化
async fn handle_user_test(
    session: &MockClientSession,
    req: UserRequest
) -> Result<Vec<serde_json::Value>> {
    // 1. 三步分层筛选，获取任务列表
    let user_query = format!("{:?}", req);
    
    // 简化的三步筛选逻辑
    let selected_agents = session.filter_agents(&user_query).await?;
    let mut all_tools = Vec::new();
    for agent_name in &selected_agents {
        let agent_tools = session.filter_tools(agent_name, &user_query);
        all_tools.extend(agent_tools);
    }
    let tasks = session.generate_plan(&user_query, &all_tools).await?;
    
    // 2. 执行子任务
    let mut results = Vec::new();
    for task in &tasks {
        let result = session
            .call_tool(&format!("{}.{}", task.agent, task.tool), task.params.clone())
            .await?;
            
        results.push(result);
    }

    Ok(results)
}

#[tokio::test]
async fn test_handle_user_success() -> Result<()> {
    let session = MockClientSession::new();
    
    // 创建用户请求
    let req = UserRequest {
        r#type: "connect_vpn".to_string(),
        payload: json!({"profile": "japan"}),
    };
    
    // 执行处理函数
    let results = handle_user_test(&session, req).await?;
    
    // 验证结果
    assert_eq!(results.len(), 2, "应该返回两个任务结果");
    assert_eq!(results[0]["success"], json!(true), "第一个任务应该成功");
    assert_eq!(results[1]["success"], json!(false), "第二个任务应该返回已连接状态");
    assert_eq!(results[1]["error"]["code"], "AlreadyConnected", "应该返回已连接错误码");
    
    Ok(())
}

#[tokio::test]
async fn test_handle_user_with_empty_results() -> Result<()> {
    // 创建没有配置响应的会话
    let mut session = MockClientSession::new();
    session.responses.clear();
    
    // 创建用户请求
    let req = UserRequest {
        r#type: "unknown_action".to_string(),
        payload: json!({}),
    };
    
    // 执行处理函数
    let results = handle_user_test(&session, req).await?;
    
    // 验证结果
    assert_eq!(results.len(), 2, "即使是未知操作，也应该返回两个任务结果");
    for result in &results {
        assert_eq!(result["success"], json!(true), "应该返回默认成功响应");
    }
    
    Ok(())
} 