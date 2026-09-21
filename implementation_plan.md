# Implementation Plan: Rust Proxy Service - Think-Block Preserving Agentic Loop

## Problem Statement

When a user sends a query like `"Search the web for the latest news on the Rust programming language"` through A-PROX to llama.cpp (Qwen 35B), the agentic loop can terminate prematurely if the model produces only an internal reasoning block (`<think>...</think>`) without a tool call. Furthermore, previous implementation plans stripped these thinking blocks before returning responses to the client. 

This implementation plan fixes the premature termination and context-loss bugs in the agentic loop **while explicitly preserving thinking blocks**, allowing them to stream directly to the client application (CLAN-AI) so that the client's internal UI toggle can manage them.

---

## Root Causes Addressed

1. **Intermediate assistant response dropped:** Intermediate tool-turn responses were omitted from the `messages` history vector, breaking the model's reasoning continuity.
2. **Think-only premature termination:** Models occasionally output `<think>` blocks without tool calls or text. The proxy must detect these "think-only" responses via `ParseResult::is_think_only()` and loop back instead of treating them as final answers.
3. **Unfiltered output:** Previous plans stripped `<think>` tags. This plan retains them entirely, piping raw streaming chunks and response bodies straight through to the client.

---

## Execution Order

| Step | File | What to Do |
|------|------|------------|
| 1 | [`src/tools/parser.rs`](src/tools/parser.rs) | Add `ParseResult` struct with `is_think_only()` and update `ToolParser` |
| 2 | [`src/tools/mod.rs`](src/tools/mod.rs) | Re-export `ParseResult` |
| 3 | [`src/context/trimmer.rs`](src/context/trimmer.rs) | Add tool-call discipline rules to `ANTI_HALLUCINATION_SYSTEM_PROMPT` and define `SYNTHESIS_FORCING_PROMPT` |
| 4 | [`src/context/mod.rs`](src/context/mod.rs) | Re-export `SYNTHESIS_FORCING_PROMPT` |
| 5 | [`src/server/routes.rs`](src/server/routes.rs) | Update imports and rewrite agentic loop paths to preserve think blocks |
| 6 | [`tests/unit_tests.rs`](tests/unit_tests.rs) | Add unit tests for think-only detection and parser handling |
| 7 | Run `cargo test` | Verify all tests compile and pass |

---

## Step 1: Modify `src/tools/parser.rs`

Add `ParseResult` to track whether a response contains think blocks and whether it consists *only* of think blocks, while leaving raw content intact for client streaming.

```rust
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
        let think_re = Regex::new(r"<think>[\s\S]*?</think>").unwrap();
        let stripped = think_re.replace_all(&self.raw_content, "").trim().to_string();
        self.has_think_block && stripped.is_empty()
    }
}

pub struct ToolParser;

impl ToolParser {
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
```

---

## Step 2: Modify `src/tools/mod.rs`

Export `ParseResult`:

```rust
pub mod parser;
pub mod registry;

pub use parser::{ExtractedToolCall, ParseResult, ToolParser};
pub use registry::ToolRegistry;
```

---

## Step 3: Modify `src/context/trimmer.rs`

Update `ANTI_HALLUCINATION_SYSTEM_PROMPT` to encourage proper tool-call formatting alongside think blocks, and add `SYNTHESIS_FORCING_PROMPT`:

```rust
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
```

---

## Step 4: Modify `src/context/mod.rs`

Export `SYNTHESIS_FORCING_PROMPT`:

```rust
pub mod tokenizer;
pub mod trimmer;

pub use tokenizer::FastTokenizer;
pub use trimmer::{
    inject_system_instructions, ChatMessage, ContextManager, ANTI_HALLUCINATION_SYSTEM_PROMPT,
    SYNTHESIS_FORCING_PROMPT,
};
```

---

## Step 5: Modify `src/server/routes.rs`

Update imports to include `ParseResult` and `SYNTHESIS_FORCING_PROMPT`, and ensure both non-streaming and streaming agentic loop handlers preserve think blocks while handling retries on think-only responses and stripping tool definitions during final synthesis.

---

## Step 6: Add Unit Tests to `tests/unit_tests.rs`

```rust
#[test]
fn test_parse_result_think_only_detection() {
    let think_only = "<think>Let me think about what to do next without invoking any tools.</think>";
    let result = ToolParser::parse_response(think_only, None);
    assert!(result.has_think_block);
    assert!(result.tool_calls.is_empty());
    assert!(result.is_think_only());

    let think_plus_text = "<think>Let me summarize.</think>Here are the findings.";
    let result2 = ToolParser::parse_response(think_plus_text, None);
    assert!(result2.has_think_block);
    assert!(!result2.is_think_only());
}
```

---

## Step 7: Build and Test

Run:
```bash
cargo test
```
