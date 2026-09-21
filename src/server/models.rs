use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAI-compatible chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<crate::context::ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// OpenAI-compatible tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

/// OpenAI-compatible tool call in assistant message or streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// OpenAI-compatible streaming chat completion chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: DeltaMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}

impl ChatCompletionChunk {
    pub fn new(id: String, model: String, choices: Vec<ChunkChoice>) -> Self {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices,
        }
    }

    /// Creates an initial role delta chunk
    pub fn role_chunk(id: &str, model: &str, role: &str) -> Self {
        Self::new(
            id.to_string(),
            model.to_string(),
            vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some(role.to_string()),
                    content: Some(String::new()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        )
    }

    /// Creates a content delta chunk
    pub fn content_delta(id: &str, model: &str, content: &str) -> Self {
        Self::new(
            id.to_string(),
            model.to_string(),
            vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: None,
                    content: Some(content.to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        )
    }

    /// Creates a finish chunk (e.g. stop or tool_calls)
    pub fn finish_chunk(id: &str, model: &str, finish_reason: &str) -> Self {
        Self::new(
            id.to_string(),
            model.to_string(),
            vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage::default(),
                finish_reason: Some(finish_reason.to_string()),
            }],
        )
    }

    /// Formats chunk as standard SSE string `data: {...}\n\n`
    pub fn to_sse_event(&self) -> String {
        format!("data: {}\n\n", serde_json::to_string(self).unwrap_or_default())
    }
}

/// OpenAI-compatible non-streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}
