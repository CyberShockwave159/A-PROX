use serde_json::Value;
use crate::context::ChatMessage;
use crate::config::{IntentConfig, ToolCommandsConfig};

mod intent;
pub use intent::{IntentCategory, IntentClassifier};

#[derive(Debug, PartialEq, Eq)]
pub enum RouteDecision {
    FastPassThrough,
    RAGAugmented { query: String },
    /// User asked the proxy to STORE content into the RAG database.
    /// The `content` field is the cleaned text to be ingested.
    RAGIngestion { content: String },
    AgenticToolLoop,
    /// Forced by a `/flag` tool command (see `ToolCommandsConfig`). `tools` lists
    /// the tool names to arm (empty = all internal tools); `query` is the message
    /// text with the flag (and any tool-name args) stripped.
    AgenticToolForced { tools: Vec<String>, query: String },
}

/// Internal intent classification used by classify_request.
#[derive(Debug, PartialEq, Eq)]
enum RouteIntent {
    AgenticTool,
    RAG,
    RAGIngestion,
    None,
}

pub struct RequestRouter;

impl RequestRouter {
    /// Evaluates incoming payload to classify request execution strategy.
    ///
    /// Priority order:
    /// 1. Explicit model targeting (a-prox-direct, a-prox-rag, a-prox-agent)
    /// 2. Header bypass (X-Proxy-Bypass: true)
    /// 3. Slash command overrides (/bypass, /direct, /pass)
    /// 4. Client-provided tools → AgenticToolLoop
    /// 5. Intent-based classification (web search, web fetch, system time, RAG search, RAG ingest)
    ///    - Hybrid: keyword patterns first, then embedding similarity if intent_classifier provided
    /// 6. Default → FastPassThrough
    pub fn classify_request(
        messages: &[ChatMessage],
        tools_payload: Option<&Value>,
        headers_bypass: bool,
        _agentic_tools_enabled: bool,
        model_name: Option<&str>,
        intent_classifier: Option<&IntentClassifier>,
        intent_config: Option<&IntentConfig>,
        tool_commands: Option<&ToolCommandsConfig>,
    ) -> RouteDecision {
        // 1. Explicit model targeting takes highest priority
        if let Some(name) = model_name {
            let name = name.trim().to_lowercase();
            if name == "a-prox-direct" || name == "a-prox-pass" || name == "a-prox-fast" {
                return RouteDecision::FastPassThrough;
            }
            if name == "a-prox-rag" || name == "a-prox-knowledge" || name == "a-prox-docs" {
                if let Some(last_user) = messages.iter().rev().find(|m| m.role == "user") {
                    let content = last_user.content_as_str();
                    return RouteDecision::RAGAugmented { query: content.to_string() };
                }
                return RouteDecision::RAGAugmented { query: "implicit rag query".to_string() };
            }
            if name == "a-prox-agent" || name == "a-prox-tools" {
                return RouteDecision::AgenticToolLoop;
            }
        }

        // 2. Header-based bypass
        if headers_bypass {
            return RouteDecision::FastPassThrough;
        }

        // 3. Inspect latest user message for slash commands and intent
        if let Some(last_user_msg) = messages.iter().rev().find(|m| m.role == "user") {
            let content = last_user_msg.content_as_str();
            let trimmed = content.trim();

            // Slash command overrides (all configurable via [tool_commands], defaults below).
            let default_cmds = ToolCommandsConfig::default();
            let cmds = tool_commands.unwrap_or(&default_cmds);
            if Self::starts_with_any(trimmed, &[&cmds.bypass, &cmds.direct, &cmds.pass_route]) {
                return RouteDecision::FastPassThrough;
            }

            // Per-tool slash-command flags: force the agentic loop with the
            // flagged tool armed (explicit user intent beats keyword detection).
            if let Some((tools, query)) = Self::match_tool_command(&content, cmds) {
                return RouteDecision::AgenticToolForced { tools, query };
            }

            // Image requests route through the agentic loop so the image_generate
            // tool can be armed (independent of the 0.70 intent threshold).
            if crate::imagegen::is_image_request(messages) {
                return RouteDecision::AgenticToolLoop;
            }

            // File-write requests route through the agentic loop so the write_file
            // tool can be armed, mirroring the image check above.
            if crate::filegen::is_file_request(messages) {
                return RouteDecision::AgenticToolLoop;
            }

            // Check model-targeting slash commands
            if Self::starts_with_any(trimmed, &[&cmds.rag, &cmds.knowledge, &cmds.docs]) {
                let intent = detect_intent(&content);
                if matches!(intent, RouteIntent::RAG) {
                    let clean_query = Self::clean_rag_query(&content, cmds);
                    return RouteDecision::RAGAugmented {
                        query: if clean_query.is_empty() { content.to_string() } else { clean_query },
                    };
                }
            }

            // Intent-based routing
            let routed = Self::route_by_intent(&content, intent_classifier, intent_config, Some(cmds));
            if let Some(decision) = routed {
                return decision;
            }
        }

        // 4. Client-provided tools → AgenticToolLoop
        if let Some(Value::Array(tools)) = tools_payload {
            if !tools.is_empty() {
                return RouteDecision::AgenticToolLoop;
            }
        }

        // 5. Default: FastPassThrough
        RouteDecision::FastPassThrough
    }

/// Match the leading slash-command flag in `content` against the configured
/// per-tool commands. Returns the tool names to arm (empty = all internal tools
/// for the generic `/tools` flag) plus the message text with the flag stripped.
/// Longest flag wins so `/ragsearch` beats `/search`.
///
/// The generic `/tools` flag also accepts tool-name args (`/tools search fetch`)
/// to arm an exact subset; leading recognized tokens are consumed as tool
/// selectors and the remainder becomes the query.
pub fn match_tool_command(
    content: &str,
    commands: &ToolCommandsConfig,
) -> Option<(Vec<String>, String)> {
    let mut entries: Vec<(&str, &str)> = vec![
        ("agentic", &commands.agentic),
        ("web_search", &commands.web_search),
        ("web_fetch", &commands.web_fetch),
        ("rag_search", &commands.rag_search),
        ("rag_ingest", &commands.rag_ingest),
        ("system_time", &commands.system_time),
        ("image_generate", &commands.image_generate),
        ("write_file", &commands.write_file),
    ];
    entries.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    let trimmed = content.trim();
    for (tool, flag) in entries {
        if flag.is_empty() || !flag.starts_with('/') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(flag) {
            return Some(Self::parse_tool_command(tool, rest));
        }
    }
    None
}

/// Build the `RouteDecision` payload for a matched flag: resolve the canonical
/// tool name(s) and the cleaned query text.
fn parse_tool_command(tool: &str, rest: &str) -> (Vec<String>, String) {
    let rest = rest.trim();
    if tool != "agentic" {
        // Single-tool flags always arm exactly that tool.
        return (vec![tool.to_string()], rest.to_string());
    }
    // `/tools` — greedily consume leading tool-name tokens as selectors.
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    let mut tools = Vec::new();
    let mut consumed = 0;
    for token in &tokens {
        match Self::resolve_tool_alias(token) {
            Some(canonical) => {
                if !tools.contains(&canonical) {
                    tools.push(canonical);
                }
                consumed += 1;
            }
            None => break,
        }
    }
    let query = tokens[consumed..].join(" ");
    (tools, query)
}

/// Map user-facing tool selectors and aliases to canonical tool names.
fn resolve_tool_alias(token: &str) -> Option<String> {
    let t = token.to_lowercase();
    let canonical = match t.as_str() {
        "search" | "web_search" => "web_search",
        "fetch" | "web_fetch" => "web_fetch",
        "rag" | "ragsearch" | "rag_search" | "knowledge" | "docs" => "rag_search",
        "ingest" | "rag_ingest" => "rag_ingest",
        "time" | "date" | "system_time" => "system_time",
        "image" | "img" | "image_generate" => "image_generate",
        "file" | "write_file" => "write_file",
        _ => return None,
    };
    Some(canonical.to_string())
}

/// True when `s` starts with any of the non-empty configured flags.
fn starts_with_any(s: &str, flags: &[&String]) -> bool {
    flags.iter().any(|f| !f.is_empty() && s.starts_with(f.as_str()))
}

/// Strip configured RAG flags and legacy phrases from the query text.
fn clean_rag_query(content: &str, commands: &ToolCommandsConfig) -> String {
    let mut q = content.to_string();
    for f in [&commands.rag, &commands.knowledge, &commands.docs] {
        if !f.is_empty() {
            q = q.replace(f.as_str(), "");
        }
    }
    for phrase in ["search docs", "in knowledge base"] {
        q = q.replace(phrase, "");
    }
    q.trim().to_string()
}

    /// Route based on intent detection (hybrid keyword + embedding).
    fn route_by_intent(
        content: &str,
        classifier: Option<&IntentClassifier>,
        _config: Option<&IntentConfig>,
        commands: Option<&ToolCommandsConfig>,
    ) -> Option<RouteDecision> {
        // If no classifier provided, use pure keyword detection
        let intent = match classifier {
            Some(classifier) => {
                let (cat, _confidence) = classifier.classify(content);
                match cat {
                    IntentCategory::AgenticTool => RouteIntent::AgenticTool,
                    IntentCategory::ImageGeneration => RouteIntent::AgenticTool,
                    IntentCategory::FileGeneration => RouteIntent::AgenticTool,
                    IntentCategory::RAGSearch => RouteIntent::RAG,
                    IntentCategory::RAGINGEST => RouteIntent::RAGIngestion,
                    IntentCategory::Passthrough => RouteIntent::None,
                }
            }
            None => detect_intent(content),
        };

        match intent {
            RouteIntent::AgenticTool => {
                Some(RouteDecision::AgenticToolLoop)
            }
            RouteIntent::RAG => {
                let clean_query = match commands {
                    Some(cmds) => Self::clean_rag_query(content, cmds),
                    None => content
                        .replace("/rag", "")
                        .replace("search docs", "")
                        .replace("in knowledge base", "")
                        .trim()
                        .to_string(),
                };
                Some(RouteDecision::RAGAugmented {
                    query: if clean_query.is_empty() { content.to_string() } else { clean_query },
                })
            }
            RouteIntent::RAGIngestion => {
                let content = clean_ingest_content(content);
                Some(RouteDecision::RAGIngestion { content })
            }
            RouteIntent::None => None,
        }
    }
}

/// Strips the leading instruction wording from a RAG-ingest request, keeping the
/// actual payload that should be stored. Handles inline payloads
/// ("save this: <data>" / "store this; <data>") as well as dictation
/// ("store this text into the RAG database" → "text").
fn clean_ingest_content(raw: &str) -> String {
    let s = raw.trim();

    // 1. Prefer the payload that follows an explicit separator.
    if let Some(idx) = s.find([':', ';']) {
        let after = s[idx + 1..].trim();
        if !after.is_empty() {
            return after.to_string();
        }
    }

    // 2. Repeatedly strip leading instruction phrases.
    let instruction_phrases = [
        "ingest into the rag", "ingest into rag", "ingest this into",
        "ingest this", "save to knowledge base", "save this into my knowledge base",
        "save this to my knowledge base", "save this in my knowledge base",
        "save this to the rag database", "save this in the rag database",
        "save this into the rag database", "save this to",
        "save this in", "save this into", "save this",
        "store this in my knowledge base", "store this into my knowledge base",
        "store this in the rag database", "store this into the rag database",
        "store this to the rag database", "store this in rag",
        "store this in", "store this into", "store this to", "store this",
        "store it in", "store it into", "store it",
        "add this to the knowledge base", "add this to", "add this",
        "index this into", "index this for", "index this",
        "remember that", "remember this", "take a note",
        "store the following", "save the following", "store the below",
        "into the vector store", "to the vector store", "in the vector store",
        "please",
    ];
    let mut cleaned = s.to_string();
    loop {
        let t = cleaned.trim_start();
        let mut stripped = false;
        for p in instruction_phrases {
            if let Some(rest) = t.strip_prefix(p) {
                cleaned = rest.trim().to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }

    // 3. Strip a trailing target/temporal phrase.
    let trailing_phrases = [
        " into the rag database", " into my knowledge base", " into the vector store",
        " to the rag database", " to my knowledge base", " into my notes",
        " for later", " so i can find it later", " so i can search for it later",
        " for future searching",
    ];
    let mut out = cleaned;
    for t in trailing_phrases {
        if let Some(rest) = out.strip_suffix(t) {
            out = rest.trim().to_string();
            break;
        }
    }
    out.trim().to_string()
}

/// Detects the user's intent from message content using string matching heuristics.
fn detect_intent(content: &str) -> RouteIntent {
    let c = content.to_lowercase();

    // Agentic tool keywords (web search, fetch, weather, time)
    let agentic_patterns = [
        "search the web", "check the web", "search online", "search the internet",
        "web search", "google for", "look up online",
        "search for ", "search for the ", "look up ", "find out ",
        "latest news", "lastest news", "breaking news", "latest headlines",
        "latest world news", "latest tech news",
        "news today", "news now", "news right now",
        "what's new", "what's happening", "what is the current",
        "current price of", "weather in", "recent developments on",
        "get me the news", "get me news", "get latest news",
        "get breaking news", "give me the news", "give me news",
        "what time is it", "current date", "today's date",
        "what day is it", "system time", "fetch", "summarize",
        "summarise", "read this link", "load https",
    ];
    for pattern in &agentic_patterns {
        if c.contains(pattern) {
            return RouteIntent::AgenticTool;
        }
    }

    // RAG ingestion keywords — MUST be checked before RAG search, because some
    // ingest phrasings (e.g. "add this to my notes") also match search keywords
    // like "my notes". Ingestion is about *storing* content, not querying it.
    let rag_ingest_patterns = [
        "ingest into rag", "save to knowledge base", "index this",
        "index this into", "index this for", "store in rag",
        "store this in knowledge base", "store this in the rag",
        "store this into the rag", "save this to the rag",
        "store this in rag", "save this in the rag", "save this into the rag",
        "store this in", "store this into", "store this to",
        "save this to", "save this in", "save this into",
        // bare verb forms — tolerate an intervening object:
        // "store this data into the RAG database", "save this text to ..."
        "store this ", "save this ", "add this ", "store it ", "save it ",
        "add this to the knowledge base", "store this in my knowledge base",
        "save this to my knowledge base", "add this to my notes",
        "store this in my notes", "save this to my notes",
        "remember this", "remember that", "take a note",
        "store the following", "save the following", "store the below",
        "into the vector store", "to the vector store", "in the vector store",
        "save it to", "store it in", "store it into",
    ];
    for pattern in &rag_ingest_patterns {
        if c.contains(pattern) {
            return RouteIntent::RAGIngestion;
        }
    }

    // RAG search keywords
    let rag_search_patterns = [
        "/rag", "search docs", "in knowledge base", "from my files",
        "my notes", "my docs", "search my notes", "search my docs",
        "search knowledge base",
    ];
    for pattern in &rag_search_patterns {
        if c.contains(pattern) || c.starts_with(pattern) {
            return RouteIntent::RAG;
        }
    }

    RouteIntent::None
}
