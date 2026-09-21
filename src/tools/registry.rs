use std::sync::Arc;
use crate::rag::RagEngine;
use crate::search::SearchService;
use super::parser::ExtractedToolCall;

pub struct ToolRegistry {
    search_service: Arc<SearchService>,
    rag_engine: Arc<RagEngine>,
}

impl ToolRegistry {
    pub fn new(search_service: Arc<SearchService>, rag_engine: Arc<RagEngine>) -> Self {
        Self {
            search_service,
            rag_engine,
        }
    }

    /// Returns the JSON schema definitions of all registered internal tools
    pub fn get_internal_tools_definitions(&self) -> serde_json::Value {
        serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Perform a low-overhead local web search for queries requiring up-to-date information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search keywords or query"
                            },
                            "count": {
                                "type": "integer",
                                "description": "Maximum number of search hits (default 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract clean, readable text/markdown from a specific URL without browser overhead.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The HTTP/HTTPS URL to fetch and parse"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_query",
                    "description": "Query the local embedded vector database to retrieve knowledge chunks, user notes, and indexed documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language question or search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": "Search the local in-memory/embedded SQLite vector store for indexed documents and knowledge chunks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language question or search query"
                            },
                            "collection": {
                                "type": "string",
                                "description": "Optional collection filter"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum chunks to return (default 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rag_ingest",
                    "description": "Ingest and index a text document or notes into the local vector database with CPU embeddings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The raw text content to chunk and index"
                            },
                            "collection": {
                                "type": "string",
                                "description": "Collection name (e.g. 'notes', 'docs')"
                            },
                            "source_uri": {
                                "type": "string",
                                "description": "Source reference URI or document title"
                            }
                        },
                        "required": ["content", "source_uri"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current host system time, timezone, and date.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "system_time",
                    "description": "Get current host system time, timezone, and date.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ])
    }

    /// Executes a detected tool call
    pub async fn execute_tool(&self, call: &ExtractedToolCall) -> String {
        tracing::info!("Executing tool: {} with args: {}", call.name, call.arguments);

        match call.name.as_str() {
            "web_search" => {
                let query = call.arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let count = call.arguments.get("count").and_then(|v| v.as_u64()).map(|c| c as usize);

                if query.is_empty() {
                    return "Error: 'query' parameter is required".to_string();
                }

                match self.search_service.search(query, count).await {
                    Ok(results) => serde_json::to_string_pretty(&results).unwrap_or_default(),
                    Err(e) => format!("Search error: {e}"),
                }
            }
            "web_fetch" => {
                let url = call.arguments.get("url").and_then(|v| v.as_str()).unwrap_or("");
                if url.is_empty() {
                    return "Error: 'url' parameter is required".to_string();
                }

                match self.search_service.scraper().fetch_and_clean(url).await {
                    Ok(page) => {
                        format!("Title: {}\n\nContent:\n{}", page.title, page.content)
                    }
                    Err(e) => format!("Scrape error: {e}"),
                }
            }
            "rag_query" | "rag_search" => {
                let query = call.arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let collection = call.arguments.get("collection").and_then(|v| v.as_str());
                let limit = call.arguments.get("limit").and_then(|v| v.as_u64()).map(|l| l as usize).unwrap_or(5);

                if query.is_empty() {
                    return "Error: 'query' parameter is required".to_string();
                }

                match self.rag_engine.query_rag(query, collection, limit) {
                    Ok(results) => self.rag_engine.format_rag_context(&results),
                    Err(e) => format!("RAG search error: {e}"),
                }
            }
            "rag_ingest" => {
                let content = call.arguments.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let source_uri = call.arguments.get("source_uri").and_then(|v| v.as_str()).unwrap_or("inline");
                let collection = call.arguments.get("collection").and_then(|v| v.as_str()).unwrap_or("default");

                if content.is_empty() {
                    return "Error: 'content' parameter is required".to_string();
                }

                match self.rag_engine.ingest_document(collection, source_uri, content) {
                    Ok(chunks) => format!("Successfully indexed {} chunks into collection '{}'", chunks, collection),
                    Err(e) => format!("Ingestion error: {e}"),
                }
            }
            "get_current_time" | "system_time" => {
                let now = chrono::Utc::now();
                let local = now.with_timezone(&chrono::Local);
                format!(
                    "Current UTC time: {}. Local time: {}",
                    now.format("%Y-%m-%d %H:%M:%S UTC"),
                    local.format("%Y-%m-%d %H:%M:%S %Z (UTC%z)")
                )
            }
            _ => format!("Unknown tool '{}'", call.name),
        }
    }
}
