use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct UserRequest {
    pub r#type: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub name: String,
    pub description: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub agent: String,
    pub schema: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub agent: String,
    pub tool: String,
    pub params: serde_json::Value,
} 