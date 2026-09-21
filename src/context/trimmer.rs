use serde::{Deserialize, Serialize};
use super::tokenizer::FastTokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn content_as_str(&self) -> String {
        match &self.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => String::new(),
        }
    }
}

pub struct ContextManager {
    tokenizer: FastTokenizer,
    max_tokens: usize,
    reserve_tokens: usize,
}

impl ContextManager {
    pub fn new(
        tokenizer: FastTokenizer,
        max_tokens: usize,
        reserve_tokens: usize,
        _sliding_window_turns: usize,
    ) -> Self {
        Self {
            tokenizer,
            max_tokens,
            reserve_tokens,
        }
    }

    /// Estimates the token footprint of a message's content.
    ///
    /// Multimodal messages carry their content as an array of parts (text and
    /// `image_url` with base64-encoded image data). Tokenizing that raw JSON
    /// would treat megabytes of base64 as text tokens, blowing past the context
    /// budget and triggering destructive truncation. Instead we count only text
    /// parts and give each image a fixed allowance — llama.cpp sizes images
    /// internally and manages their tiles itself.
    fn content_token_estimate(&self, msg: &ChatMessage) -> usize {
        const TOKENS_PER_IMAGE: usize = 576;
        match &msg.content {
            None => 0,
            Some(serde_json::Value::String(s)) => self.tokenizer.count_tokens(s),
            Some(v) => {
                let mut total = 0;
                match v.as_array() {
                    Some(parts) => {
                        for part in parts {
                            match part.get("type").and_then(|t| t.as_str()) {
                                Some("text") => {
                                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                        total += self.tokenizer.count_tokens(text);
                                    }
                                }
                                Some("image_url") => total += TOKENS_PER_IMAGE,
                                _ => {}
                            }
                        }
                    }
                    None => {
                        // Non-array structured content (e.g. tool-shaped objects):
                        // fall back to the old heuristic so counts stay bounded.
                        total = self.tokenizer.count_tokens(&msg.content_as_str());
                    }
                }
                total
            }
        }
    }

    /// Computes total token count across all messages
    pub fn calculate_total_tokens(&self, messages: &[ChatMessage]) -> usize {
        messages
            .iter()
            .map(|m| {
                let content_tokens = self.content_token_estimate(m);
                let tool_str = m.tool_calls.as_ref().map(|t| t.to_string()).unwrap_or_default();
                content_tokens + self.tokenizer.count_tokens(&tool_str) + 4
            })
            .sum()
    }

    /// Prunes conversation history if it exceeds the token budget.
    /// Invariant 1: System prompt is NEVER pruned.
    /// Invariant 2: The latest user turn is ALWAYS preserved.
    /// Invariant 3: Recent sliding window turns are prioritized.
    pub fn prune_messages(&self, messages: Vec<ChatMessage>) -> (Vec<ChatMessage>, bool) {
        let budget = self.max_tokens.saturating_sub(self.reserve_tokens);
        let current_tokens = self.calculate_total_tokens(&messages);

        if current_tokens <= budget {
            return (messages, false);
        }

        tracing::info!(
            "Context exceeded budget: {} tokens > {} max allowed. Initiating pruning.",
            current_tokens,
            budget
        );

        if messages.len() <= 2 {
            // Only system + latest user message: truncate the content directly.
            // Multimodal content (arrays with image parts) must never be
            // replaced by truncated text — that would destroy the image payload.
            let mut pruned = messages;
            if let Some(last) = pruned.last_mut() {
                if matches!(last.content, Some(serde_json::Value::String(_))) {
                    let truncated = self.tokenizer.truncate_tokens(&last.content_as_str(), budget / 2);
                    last.content = Some(serde_json::Value::String(truncated));
                } else {
                    tracing::warn!("Context over budget but latest message has non-text content; forwarding intact (image parts preserved).");
                }
            }
            return (pruned, true);
        }

        // Separate system messages, recent window, and older history
        let mut system_msgs: Vec<ChatMessage> = Vec::new();
        let mut history: Vec<ChatMessage> = Vec::new();

        for msg in messages {
            if msg.role == "system" {
                system_msgs.push(msg);
            } else {
                history.push(msg);
            }
        }

        let system_tokens = self.calculate_total_tokens(&system_msgs);
        let remaining_budget = budget.saturating_sub(system_tokens);

        // Keep the latest turns within sliding window
        let mut final_history: Vec<ChatMessage> = Vec::new();
        let mut accumulated_tokens = 0;

        // Iterate backwards from newest to oldest
        for msg in history.into_iter().rev() {
            let msg_tokens = self.content_token_estimate(&msg) + 4;
            if accumulated_tokens + msg_tokens <= remaining_budget {
                accumulated_tokens += msg_tokens;
                final_history.push(msg);
            } else {
                // If it's the very last user message (first in reverse), we must keep at least a truncated version
                if final_history.is_empty() {
                    if matches!(msg.content, Some(serde_json::Value::String(_))) {
                        let truncated = self.tokenizer.truncate_tokens(&msg.content_as_str(), remaining_budget.max(256));
                        let mut truncated_msg = msg;
                        truncated_msg.content = Some(serde_json::Value::String(truncated));
                        final_history.push(truncated_msg);
                    } else {
                        // Multimodal content: keep intact rather than destroying image parts.
                        tracing::warn!("Context over budget but latest user message has non-text content; forwarding intact (image parts preserved).");
                        final_history.push(msg);
                    }
                }
                break;
            }
        }

        final_history.reverse();

        // Assemble final messages: system + [optional summary notice] + history
        let mut result = system_msgs;
        result.extend(final_history);

        (result, true)
    }

    pub fn tokenizer(&self) -> &FastTokenizer {
        &self.tokenizer
    }
}

pub const ANTI_HALLUCINATION_SYSTEM_PROMPT: &str = r#"CRITICAL OPERATIONAL RULES:
1. ANTI-HALLUCINATION & FACTUAL RIGOR:
   - Never invent, speculate, or guess facts, dates, news, URLs, numbers, technical specifications, or API parameters.
   - If you do not possess verified, certain facts to fulfill the request, you MUST invoke the relevant tool rather than guessing.
2. MANDATORY TOOL SELECTION:
   - Use `web_search` for real-time events, current news, recent technical documentation, external facts, weather, or pricing.
   - Use `rag_query` (or `rag_search`) to retrieve local notes, user documentation, and indexed knowledge chunks.
   - Use `get_current_time` (or `system_time`) whenever asked for the current date, time, timezone, or day of the week.
   - Use `web_fetch` to fetch and extract clean text from any URL provided in the conversation.
 3. GROUNDING:
    - Ground all answers strictly in facts returned by tool results.
 4. TOOL CALL DISCIPLINE:
    - When you decide inside a <think> block that you need to call a tool, you MUST immediately follow the </think> closing tag with a <tool_call> tag containing the tool invocation."#;

/// System message appended before the final synthesis turn to prevent further tool calls.
pub const SYNTHESIS_FORCING_PROMPT: &str = "You have completed all tool executions. Now synthesize a comprehensive, well-structured answer for the user based on the tool results above. Do NOT call any tools. Do NOT use <tool_call> tags. Respond directly with your final answer.";

/// Injects or appends anti-hallucination and tool guidance system instructions.
/// If a system message exists, it appends the directives (avoiding duplicates).
/// If no system message exists, it inserts a new system message at index 0.
pub fn inject_system_instructions(messages: &mut Vec<ChatMessage>, prompt: &str) {
    if let Some(sys_msg) = messages.iter_mut().find(|m| m.role == "system") {
        let existing = sys_msg.content_as_str();
        if !existing.contains("[CRITICAL OPERATIONAL RULES]") && !existing.contains("ANTI-HALLUCINATION") {
            let augmented = format!("{}\n\n[CRITICAL OPERATIONAL RULES]\n{}", existing.trim(), prompt);
            sys_msg.content = Some(serde_json::json!(augmented));
        }
    } else {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: Some(serde_json::json!(prompt)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }
}

