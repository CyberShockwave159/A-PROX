use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ExtractedToolCall {
    pub fn arguments_string(&self) -> String {
        serde_json::to_string(&self.arguments).unwrap_or_default()
    }
}

/// Result of parsing an assistant response, including tool calls and think-block analysis.
#[derive(Debug, Clone)]
pub struct ParseResult {
    /// Extracted tool calls (from structured `tool_calls` field or Qwen XML).
    pub tool_calls: Vec<ExtractedToolCall>,
    /// Whether the response contained `<think>...</think>` blocks.
    pub has_think_block: bool,
    /// Raw content of the response.
    pub raw_content: String,
}

impl ParseResult {
/// Returns true if the response consists ONLY of think blocks with no substantive text or tool call.
    pub fn is_think_only(&self) -> bool {
        if !self.tool_calls.is_empty() {
            return false;
        }
        let think_re = Regex::new(r"<think>[\s\S]*?</think>").unwrap();
        let stripped = think_re.replace_all(&self.raw_content, "").trim().to_string();
        self.has_think_block && stripped.is_empty()
    }
}

pub struct ToolParser;

impl ToolParser {
    /// Detects tool calls in assistant response, either from structured tool_calls field
    /// or from embedded Qwen XML tags `<tool_call>...</tool_call>`
    pub fn parse_tool_calls(
        message_content: &str,
        tool_calls_field: Option<&serde_json::Value>,
    ) -> Vec<ExtractedToolCall> {
        let mut calls = Vec::new();

        // 1. Check structured OpenAI tool_calls field
        if let Some(serde_json::Value::Array(arr)) = tool_calls_field {
            for (idx, item) in arr.iter().enumerate() {
                if let Some(func) = item.get("function") {
                    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let args_raw = func.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
                    let arguments: serde_json::Value = serde_json::from_str(args_raw).unwrap_or(serde_json::json!({}));
                    let id = item.get("id").and_then(|v| v.as_str()).unwrap_or(&format!("call_{}", idx)).to_string();

                    if !name.is_empty() {
                        calls.push(ExtractedToolCall { id, name, arguments });
                    }
                }
            }
            if !calls.is_empty() {
                return calls;
            }
        }

        // 2. Check Qwen XML format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        let xml_re = Regex::new(r"<tool_call>([\s\S]*?)</tool_call>").unwrap();
        for (idx, cap) in xml_re.captures_iter(message_content).enumerate() {
            let inner = cap[1].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(inner) {
                let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let arguments = val.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                let id = format!("call_xml_{}", idx);

                if !name.is_empty() {
                    calls.push(ExtractedToolCall { id, name, arguments });
                }
            }
        }

        calls
    }

    pub fn parse_response(
        message_content: &str,
        tool_calls_field: Option<&serde_json::Value>,
    ) -> ParseResult {
let tool_calls = Self::parse_tool_calls(message_content, tool_calls_field);
        let think_re = Regex::new(r"<think>[\s\S]*?</think>").unwrap();
        let has_think_block = think_re.is_match(message_content);

        ParseResult {
            tool_calls,
            has_think_block,
            raw_content: message_content.to_string(),
        }
    }
}
