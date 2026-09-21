use a_prox::context::{
    inject_system_instructions, ChatMessage, ContextManager, FastTokenizer,
    ANTI_HALLUCINATION_SYSTEM_PROMPT,
};
use a_prox::rag::TextChunker;
use a_prox::router::{RequestRouter, RouteDecision};
use a_prox::search::readability::clean_html;
use a_prox::server::models::{ChatCompletionChunk, ChatCompletionRequest};
use a_prox::server::ApiKey;
use a_prox::tools::{ExtractedToolCall, ToolParser};
use axum::extract::FromRequestParts;
use axum::http::{HeaderMap, Request};
use serde_json::json;

/// Builds a uniquely-named SQLite path under the OS temp dir so tests never
/// write artifacts into the repo's `data/` directory.
fn test_db_path(label: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir()
        .join(format!("a_prox_{}_{}.db", label, nanos))
        .to_string_lossy()
        .into_owned()
}

/// Clean-up protocol: after a test finishes with a temp SQLite database, remove
/// the main file plus its `-wal` and `-shm` WAL sidecar files so no testing
/// artifacts are left behind. Must be called only after every `VectorStore`
/// connection to the file has been dropped, otherwise the sidecars can be
/// re-created or remain locked.
fn cleanup_test_db(db_path: &str) {
    let _ = std::fs::remove_file(format!("{db_path}-wal"));
    let _ = std::fs::remove_file(format!("{db_path}-shm"));
    let _ = std::fs::remove_file(db_path);
}

#[test]
fn test_tool_parser_qwen_xml() {
    let content = r#"I will now search for the latest documentation.
<tool_call>
{"name": "web_search", "arguments": {"query": "Rust axum tutorial 2026"}}
</tool_call>
Let me know if you need more details."#;

    let calls = ToolParser::parse_tool_calls(content, None);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "web_search");
    assert_eq!(
        calls[0].arguments.get("query").and_then(|v| v.as_str()),
        Some("Rust axum tutorial 2026")
    );
}

#[test]
fn test_tool_parser_openai_json() {
    let tool_calls_field = json!([
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "rag_search",
                "arguments": "{\"query\": \"neural networks\", \"limit\": 3}"
            }
        }
    ]);

    let calls = ToolParser::parse_tool_calls("", Some(&tool_calls_field));
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_123");
    assert_eq!(calls[0].name, "rag_search");
    assert_eq!(
        calls[0].arguments.get("query").and_then(|v| v.as_str()),
        Some("neural networks")
    );
}

#[test]
fn test_context_manager_pruning() {
    let tokenizer = FastTokenizer::new::<&str>(None);
    // Tight budget of 100 tokens with 20 reserve
    let mgr = ContextManager::new(tokenizer, 100, 20, 2);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(json!("You are an AI assistant.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(json!("Old message turn 1 with some filler text to consume tokens.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: Some(json!("Old reply turn 1 with more filler text to exceed budget.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(json!("Latest urgent user prompt that must be preserved.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let (pruned, was_pruned) = mgr.prune_messages(messages);
    assert!(was_pruned);
    assert_eq!(pruned[0].role, "system");
    assert_eq!(pruned.last().unwrap().role, "user");
}

#[test]
fn test_context_manager_preserves_multimodal_content() {
    let tokenizer = FastTokenizer::new::<&str>(None);
    // Tight budget of 100 tokens with 20 reserve — forces pruning to trigger.
    let mgr = ContextManager::new(tokenizer, 100, 20, 2);

    // ~200 KB of base64 image data (the raw JSON would be ~57k+ heuristic tokens).
    let base64_img = format!("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD{}", "A".repeat(200_000));

    let multimodal_content = json!([
        {"type": "text", "text": "What is in this photo?"},
        {"type": "image_url", "image_url": {"url": base64_img}}
    ]);

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(json!("You are an AI assistant.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(multimodal_content.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    // Base64 image data must NOT inflate the token count as text.
    let total = mgr.calculate_total_tokens(&messages);
    assert!(total < 1000, "image base64 inflated token count to {total}");

    // Even when pruning trips, the image_url part must survive fully intact.
    let (pruned, was_pruned) = mgr.prune_messages(messages);
    assert!(was_pruned);
    let parts = pruned
        .last()
        .unwrap()
        .content
        .as_ref()
        .expect("multimodal user content must not be replaced by truncated text")
        .as_array()
        .expect("multimodal content must stay an array of parts");
    let img = parts
        .iter()
        .find(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"))
        .expect("image_url part must be preserved");
    let url = img.pointer("/image_url/url").and_then(|u| u.as_str()).unwrap();
    assert!(url.starts_with("data:image/jpeg;base64,"));
    assert_eq!(url.len(), format!("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD{}", "A".repeat(200_000)).len());
}

#[test]
fn test_text_chunker() {
    let chunker = TextChunker::new(50, 10);
    let text = "Paragraph 1 is here.\n\nParagraph 2 contains more information.\n\nParagraph 3 wraps up.";
    let chunks = chunker.chunk_text(text);
    assert!(!chunks.is_empty());
}

#[test]
fn test_html_readability() {
    let raw_html = r#"
    <!DOCTYPE html>
    <html>
    <head><title>Test Article Page</title></head>
    <body>
        <nav><a href="/">Home</a><a href="/about">About</a></nav>
        <script>console.log("Malicious or tracking script");</script>
        <style>body { color: red; }</style>
        <article>
            <h1>Understanding Memory Hierarchies</h1>
            <p>DDR4 RAM operates with lower latency than NVMe SSDs, making in-memory caches crucial.</p>
        </article>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    "#;

    let page = clean_html(raw_html);
    assert_eq!(page.title, "Test Article Page");
    assert!(page.content.contains("Understanding Memory Hierarchies"));
    assert!(page.content.contains("DDR4 RAM operates"));
    assert!(!page.content.contains("Malicious"));
    assert!(!page.content.contains("Copyright 2026"));
}

#[test]
fn test_onnx_cpu_embeddings_and_vector_store() {
    use a_prox::embeddings::CpuEmbedder;
    use a_prox::db::VectorStore;

    let model_path = "models/bge-small-en-v1.5-int8.onnx";
    let tokenizer_path = "models/tokenizer.json";

    let embedder = CpuEmbedder::new(model_path, Some(tokenizer_path), 4, 384);
    let vector = embedder.embed_text("High-performance hardware optimization").unwrap();
    assert_eq!(vector.len(), 384);

    // Verify vector is normalized (magnitude approx 1.0)
    let mag: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((mag - 1.0).abs() < 1e-3);

    // Test SQLite + sqlite-vec insertion and hybrid search in-memory
    let temp_db = test_db_path("embed");
    let store = VectorStore::new(&temp_db, 384, 128, 64).unwrap();

    let chunk_id = store.insert_chunk(
        "test_col",
        "doc1.txt",
        0,
        "High-performance hardware optimization on Zen 3 CPUs",
        10,
        &vector,
    ).unwrap();
    assert!(chunk_id > 0);

    let hits = store.hybrid_search("hardware optimization", &vector, None, 5).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].collection, "test_col");
    assert!(hits[0].content.contains("Zen 3"));

    // Clean-up protocol: close the connection, then remove the DB and its WAL/SHM sidecars
    drop(store);
    cleanup_test_db(&temp_db);
}

#[test]
fn test_api_key_extractor_valid_x_api_key() {
    let mut req = Request::new(());
    *req.headers_mut() = HeaderMap::new();
    req.headers_mut().insert(
        "x-api-key",
        axum::http::HeaderValue::from_static("test-key-123"),
    );
    let mut parts = req.into_parts().0;

    let result = tokio::runtime::Runtime::new().unwrap().block_on(ApiKey::from_request_parts(&mut parts, &()));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().0, "test-key-123");
}

#[test]
fn test_api_key_extractor_valid_bearer() {
    let mut req = Request::new(());
    *req.headers_mut() = HeaderMap::new();
    req.headers_mut().insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer my-secret-key"),
    );
    let mut parts = req.into_parts().0;

    let result = tokio::runtime::Runtime::new().unwrap().block_on(ApiKey::from_request_parts(&mut parts, &()));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().0, "my-secret-key");
}

#[test]
fn test_api_key_extractor_missing() {
    let mut req = Request::new(());
    *req.headers_mut() = HeaderMap::new();
    let mut parts = req.into_parts().0;

    let result = tokio::runtime::Runtime::new().unwrap().block_on(ApiKey::from_request_parts(&mut parts, &()));
    assert!(result.is_err());
}

#[test]
fn test_api_key_extractor_x_api_key_takes_precedence() {
    let mut req = Request::new(());
    *req.headers_mut() = HeaderMap::new();
    req.headers_mut().insert(
        "x-api-key",
        axum::http::HeaderValue::from_static("x-key-wins"),
    );
    req.headers_mut().insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer bearer-loses"),
    );
    let mut parts = req.into_parts().0;

    let result = tokio::runtime::Runtime::new().unwrap().block_on(ApiKey::from_request_parts(&mut parts, &()));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().0, "x-key-wins");
}

#[test]
fn test_router_fast_passthrough() {
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: Some(json!("Write a Rust function that sorts a vector.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: Some(json!("Explain how async/await works in Rust")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: Some(json!("What is the capital of France?")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);
}

#[test]
fn test_router_agentic_triggers() {
    // Web search triggers
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("search the web for latest news")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("search online for Rust 2026 updates")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("google for axum framework tutorial")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("look up online the weather today")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("what is the latest news on AI")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("current price of Bitcoin")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    // System time triggers
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("what time is it")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("what is the current date")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("today's date please")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("what day is it today")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    assert_eq!(RequestRouter::classify_request(&messages, None, false, false, None, None, None), RouteDecision::AgenticToolLoop);
}

#[test]
fn test_router_bypasses() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("search the web for anything")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    // Header bypass should force FastPassThrough even with search intent
    let decision = RequestRouter::classify_request(&messages, None, true, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    // Slash command bypasses
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("/bypass search the web")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("/direct what time is it")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("/pass search the web")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);
}

#[test]
fn test_sse_chunk_format() {
    use a_prox::router::RouteDecision;

    // Validate that FastPassThrough and RAGAugmented have correct variants
    let fast = RouteDecision::FastPassThrough;
    assert!(matches!(fast, RouteDecision::FastPassThrough));

    let rag = RouteDecision::RAGAugmented { query: "test query".to_string() };
    if let RouteDecision::RAGAugmented { query } = rag {
        assert_eq!(query, "test query");
    } else {
        panic!("Expected RAGAugmented variant");
    }

    let agentic = RouteDecision::AgenticToolLoop;
    assert!(matches!(agentic, RouteDecision::AgenticToolLoop));

    let ingest = RouteDecision::RAGIngestion { content: "test content".to_string() };
    if let RouteDecision::RAGIngestion { content } = ingest {
        assert_eq!(content, "test content");
    } else {
        panic!("Expected RAGIngestion variant");
    }

    // Validate that RouteDecision has all expected variants
    let all_variants = vec![
        RouteDecision::FastPassThrough,
        RouteDecision::RAGAugmented { query: "test".to_string() },
        RouteDecision::RAGIngestion { content: "test".to_string() },
        RouteDecision::AgenticToolLoop,
    ];
    assert_eq!(all_variants.len(), 4);
}

#[test]
fn test_router_rag_intent() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("/rag search my knowledge base")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    match decision {
        RouteDecision::RAGAugmented { query } => {
            assert!(query.contains("knowledge base"));
        }
        _ => panic!("Expected RAGAugmented, got {:?}", decision),
    }

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("search docs about Rust")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    match decision {
        RouteDecision::RAGAugmented { .. } => {}
        _ => panic!("Expected RAGAugmented, got {:?}", decision),
    }

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("search in knowledge base")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    match decision {
        RouteDecision::RAGAugmented { .. } => {}
        _ => panic!("Expected RAGAugmented, got {:?}", decision),
    }
}

#[test]
fn test_router_tools_trigger() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("Hello")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {"type": "object"}
            }
        }
    ]);

    let decision = RequestRouter::classify_request(&messages, Some(&tools), false, false, None, None, None);
    assert_eq!(decision, RouteDecision::AgenticToolLoop);

    // Empty tools array should not trigger agentic
    let empty_tools = json!([]);
    let decision = RequestRouter::classify_request(&messages, Some(&empty_tools), false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    // No tools and no agentic tools enabled = FastPassThrough
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);
}

#[test]
fn test_router_model_name_targeting() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("Hello world")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    // a-prox-direct should force FastPassThrough regardless of message content
    let decision = RequestRouter::classify_request(&messages, None, false, true, Some("a-prox-direct"), None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);

    // a-prox-rag should route to RAGAugmented
    let decision = RequestRouter::classify_request(&messages, None, false, false, Some("a-prox-rag"), None, None);
    assert!(matches!(decision, RouteDecision::RAGAugmented { .. }));

    // a-prox-agent should route to AgenticToolLoop
    let decision = RequestRouter::classify_request(&messages, None, false, false, Some("a-prox-agent"), None, None);
    assert_eq!(decision, RouteDecision::AgenticToolLoop);

    // a-prox-rag with specific message content
    let rag_messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("What is the capital of France?")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&rag_messages, None, false, false, Some("a-prox-rag"), None, None);
    if let RouteDecision::RAGAugmented { query } = decision {
        assert!(query.contains("capital of France"));
    } else {
        panic!("Expected RAGAugmented, got {:?}", decision);
    }
}

#[test]
fn test_router_rag_ingestion_intent() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("ingest into rag this important document content")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    if let RouteDecision::RAGIngestion { content } = decision {
        assert!(content.contains("important document content"));
    } else {
        panic!("Expected RAGIngestion for ingestion, got {:?}", decision);
    }

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("save to knowledge base this meeting notes")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    if let RouteDecision::RAGIngestion { content } = decision {
        assert!(content.contains("meeting notes"));
    } else {
        panic!("Expected RAGIngestion for ingestion, got {:?}", decision);
    }

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("index this into the vector store")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert!(matches!(decision, RouteDecision::RAGIngestion { .. }));
}

#[test]
fn test_router_rag_ingestion_natural_phrasings() {
    // Natural phrasings that previously fell through to FastPassThrough because
    // the keyword matcher only knew a few narrow verbatim phrases and the
    // embedding-based classifier had no centroids (empty examples in the config).
    let ingest_cases = vec![
        "please store this data into the RAG database",
        "store this in the RAG database",
        "save this text to my knowledge base",
        "add this to my notes so I can find it later",
        "remember this for me",
        "save the following to my notes: the deploy command is make deploy",
        "store this in my local docs",
        "please remember that the API key is stored in the env file",
    ];

    for content in ingest_cases {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
        assert!(
            matches!(&decision, RouteDecision::RAGIngestion { .. }),
            "Expected RAGIngestion for '{}' but got {:?}",
            content,
            decision
        );
    }

    // Pure search queries must NOT be reclassified as ingestion.
    let search_cases = vec![
        "search my notes for the API endpoint",
        "what do my notes say about authentication",
        "search docs about the migration",
    ];
    for content in search_cases {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
        assert!(
            matches!(&decision, RouteDecision::RAGAugmented { .. }),
            "Expected RAGAugmented (search) for '{}' but got {:?}",
            content,
            decision
        );
    }
}

#[test]
fn test_router_additional_web_search_keywords() {
    let test_cases = vec![
        ("breaking news on AI regulation", true),
        ("what is the weather in Tokyo", true),
        ("recent developments on quantum computing", true),
        ("what is the current state of the market", true),
        ("summarize https://example.com/article", true),
        ("read this link: https://example.com", true),
        ("what is the system time", true),
    ];

    for (content, expect_agentic) in test_cases {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
        if expect_agentic {
            assert!(
                matches!(decision, RouteDecision::AgenticToolLoop),
                "Expected AgenticToolLoop for '{}' but got {:?}",
                content, decision
            );
        }
    }
}

#[test]
fn test_router_agentic_disabled_no_tools() {
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("search the web for news")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    // With agentic_tools_enabled = false, should still detect search intent and route to AgenticToolLoop
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::AgenticToolLoop);

    // Web search without agentic_tools_enabled should still go to AgenticToolLoop (intent detection is independent)
    let decision = RequestRouter::classify_request(&messages, None, false, true, None, None, None);
    assert_eq!(decision, RouteDecision::AgenticToolLoop);

    // General chat with agentic_tools_enabled = false stays FastPassThrough
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("Hello, how are you?")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];
    let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
    assert_eq!(decision, RouteDecision::FastPassThrough);
}

#[test]
fn test_router_broad_web_search_patterns() {
    let web_search_patterns = vec![
        "search the web",
        "search online",
        "search the internet",
        "web search",
        "google for latest updates",
        "look up online",
        "search for Rust 2026",
        "search for the answer",
        "look up the weather",
        "find out what happened",
        "what's new in tech",
        "what's happening in AI",
        "latest headlines",
        "get me the news",
        "get me news",
        "get latest news",
        "get breaking news",
        "give me the news",
        "give me news",
    ];

    for pattern in web_search_patterns {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!(pattern)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
        assert!(
            matches!(decision, RouteDecision::AgenticToolLoop),
            "Expected AgenticToolLoop for '{}' but got {:?}",
            pattern, decision
        );
    }
}

#[test]
fn test_router_broad_web_fetch_patterns() {
    let fetch_patterns = vec![
        "summarize https://example.com",
        "summarise this https://example.com",
        "read this link: https://example.com",
        "fetch https://example.com",
        "load https://example.com",
    ];

    for pattern in fetch_patterns {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!(pattern)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
        assert!(
            matches!(decision, RouteDecision::AgenticToolLoop),
            "Expected AgenticToolLoop for '{}' but got {:?}",
            pattern, decision
        );
    }
}

#[test]
fn test_router_news_with_intermediate_words() {
    // Tricky cases where words appear between "latest" and "news"
    let news_cases = vec![
        "latest Middle East news today",
        "latest technology news right now",
        "latest world news now",
        "latest sports news today",
        "news today about AI",
        "what's the latest tech news",
        "can you check the web for the latest news",
        "check the web for lastest news",
    ];

    for content in news_cases {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let decision = RequestRouter::classify_request(&messages, None, false, false, None, None, None);
        assert!(
            matches!(decision, RouteDecision::AgenticToolLoop),
            "Expected AgenticToolLoop for '{}' but got {:?}",
            content, decision
        );
    }
}

#[test]
fn test_system_prompt_injection_empty_messages() {
    let mut messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("What is the capital of France?")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    inject_system_instructions(&mut messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    let content = messages[0].content_as_str();
    assert!(content.contains("CRITICAL OPERATIONAL RULES"));
    assert!(content.contains("ANTI-HALLUCINATION"));
    assert!(content.contains("web_search"));
    assert!(content.contains("rag_query"));
    assert_eq!(messages[1].role, "user");
}

#[test]
fn test_system_prompt_injection_existing_system() {
    let mut messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: Some(json!("You are an expert Rust programming tutor.")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: Some(json!("How do lifetimes work?")),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    inject_system_instructions(&mut messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    let content = messages[0].content_as_str();
    assert!(content.contains("expert Rust programming tutor"));
    assert!(content.contains("[CRITICAL OPERATIONAL RULES]"));
    assert!(content.contains("ANTI-HALLUCINATION"));
}

#[test]
fn test_system_prompt_injection_idempotent() {
    let mut messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(json!("Hello")),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }];

    inject_system_instructions(&mut messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);
    let len_after_first = messages.len();
    let content_after_first = messages[0].content_as_str();

    // Second invocation should not duplicate
    inject_system_instructions(&mut messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);
    assert_eq!(messages.len(), len_after_first);
    assert_eq!(messages[0].content_as_str(), content_after_first);
}

#[test]
fn test_tool_definitions_contain_aliases() {
    use a_prox::db::VectorStore;
    use a_prox::embeddings::CpuEmbedder;
    use a_prox::rag::RagEngine;
    use a_prox::search::SearchService;
    use a_prox::tools::ToolRegistry;
    use std::sync::Arc;

    let temp_db = test_db_path("reg");
    let defs = {
        let store = Arc::new(VectorStore::new(&temp_db, 384, 64, 32).unwrap());
        let embedder = Arc::new(CpuEmbedder::new::<&str>("models/bge-small-en-v1.5-int8.onnx", None, 1, 384));
        let tokenizer = Arc::new(FastTokenizer::new::<&str>(None));
        let rag = Arc::new(RagEngine::new(store, embedder, tokenizer));
        let search = Arc::new(SearchService::new("http://127.0.0.1:8888".to_string(), 3, 5, 60));
        let registry = ToolRegistry::new(search, rag);
        registry.get_internal_tools_definitions()
    };

    let defs_array = defs.as_array().expect("Expected array of tools");
    let names: Vec<&str> = defs_array.iter()
        .filter_map(|t| t.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
        .collect();

    assert!(names.contains(&"web_search"), "web_search must be registered");
    assert!(names.contains(&"rag_query"), "rag_query must be registered");
    assert!(names.contains(&"rag_search"), "rag_search must be registered");
    assert!(names.contains(&"get_current_time"), "get_current_time must be registered");
    assert!(names.contains(&"system_time"), "system_time must be registered");
    assert!(names.contains(&"web_fetch"), "web_fetch must be registered");
    assert!(names.contains(&"rag_ingest"), "rag_ingest must be registered");

    cleanup_test_db(&temp_db);
}

#[tokio::test]
async fn test_tool_registry_get_current_time_and_system_time() {
    use a_prox::db::VectorStore;
    use a_prox::embeddings::CpuEmbedder;
    use a_prox::rag::RagEngine;
    use a_prox::search::SearchService;
    use a_prox::tools::ToolRegistry;
    use std::sync::Arc;

    let temp_db = test_db_path("time");

    let (output_alias, output_orig) = {
        let store = Arc::new(VectorStore::new(&temp_db, 384, 64, 32).unwrap());
        let embedder = Arc::new(CpuEmbedder::new::<&str>("models/bge-small-en-v1.5-int8.onnx", None, 1, 384));
        let tokenizer = Arc::new(FastTokenizer::new::<&str>(None));
        let rag = Arc::new(RagEngine::new(store, embedder, tokenizer));
        let search = Arc::new(SearchService::new("http://127.0.0.1:8888".to_string(), 3, 5, 60));

        let registry = ToolRegistry::new(search, rag);

        let call_alias = ExtractedToolCall {
            id: "call_1".to_string(),
            name: "get_current_time".to_string(),
            arguments: json!({}),
        };
        let output_alias = registry.execute_tool(&call_alias).await;
        assert!(output_alias.contains("Current UTC time"));

        let call_orig = ExtractedToolCall {
            id: "call_2".to_string(),
            name: "system_time".to_string(),
            arguments: json!({}),
        };
        let output_orig = registry.execute_tool(&call_orig).await;
        assert!(output_orig.contains("Current UTC time"));
        (output_alias, output_orig)
    };

    // Keep asserts outside the connection scope; drop() already closed the store.
    assert!(output_alias.contains("Current UTC time"));
    assert!(output_orig.contains("Current UTC time"));
    cleanup_test_db(&temp_db);
}

#[test]
fn test_openai_models_serialization() {
    let req_json = json!({
        "model": "qwen3.6-35b-moe",
        "messages": [
            {"role": "user", "content": "What is the weather?"}
        ],
        "stream": true,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {"type": "object"}
                }
            }
        ]
    });

    let req: ChatCompletionRequest = serde_json::from_value(req_json).unwrap();
    assert_eq!(req.model, Some("qwen3.6-35b-moe".to_string()));
    assert_eq!(req.stream, Some(true));
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.tools.unwrap().len(), 1);
}

#[test]
fn test_sse_chunk_constructors_and_formatting() {
    let chunk = ChatCompletionChunk::role_chunk("chatcmpl-test", "a-prox-agent", "assistant");
    let sse_str = chunk.to_sse_event();
    assert!(sse_str.starts_with("data: "));
    assert!(sse_str.ends_with("\n\n"));
    assert!(sse_str.contains("chatcmpl-test"));
    assert!(sse_str.contains("assistant"));

    let delta = ChatCompletionChunk::content_delta("chatcmpl-test", "a-prox-agent", "Hello world");
    let delta_str = delta.to_sse_event();
    assert!(delta_str.contains("Hello world"));

    let finish = ChatCompletionChunk::finish_chunk("chatcmpl-test", "a-prox-agent", "stop");
    let finish_str = finish.to_sse_event();
    assert!(finish_str.contains("\"finish_reason\":\"stop\""));
}

#[test]
fn test_tool_parser_with_rag_query_and_get_current_time() {
    let openai_payload = json!([
        {
            "id": "call_rag_1",
            "type": "function",
            "function": {
                "name": "rag_query",
                "arguments": "{\"query\":\"quarterly earnings\"}"
            }
        },
        {
            "id": "call_time_1",
            "type": "function",
            "function": {
                "name": "get_current_time",
                "arguments": "{}"
            }
        }
    ]);

    let calls = ToolParser::parse_tool_calls("", Some(&openai_payload));
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "rag_query");
    assert_eq!(calls[0].arguments.get("query").and_then(|v| v.as_str()), Some("quarterly earnings"));
    assert_eq!(calls[1].name, "get_current_time");

    let qwen_xml = r#"<tool_call>
{"name": "get_current_time", "arguments": {}}
</tool_call>"#;
    let xml_calls = ToolParser::parse_tool_calls(qwen_xml, None);
    assert_eq!(xml_calls.len(), 1);
    assert_eq!(xml_calls[0].name, "get_current_time");
}

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

#[test]
fn test_parse_result_no_think_block() {
    let plain_text = "Here is the answer to your question about Rust.";
    let result = ToolParser::parse_response(plain_text, None);
    assert!(!result.has_think_block);
    assert!(!result.is_think_only());
    assert_eq!(result.raw_content, plain_text);
}

#[test]
fn test_parse_result_think_with_tool_call() {
    let content = r#"<think>I need to search the web for this.</think>
<tool_call>
{"name": "web_search", "arguments": {"query": "Rust programming language"}}
</tool_call>
The results show that Rust is a systems programming language."#;
    let result = ToolParser::parse_response(content, None);
    assert!(result.has_think_block);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].name, "web_search");
    assert!(!result.is_think_only());
}

#[test]
fn test_parse_result_openai_tool_calls_with_think() {
    let tool_calls_field = json!([
        {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "rag_search",
                "arguments": "{\"query\": \"vector databases\"}"
            }
        }
    ]);
    let content = "<think>Looking up vector database information in my knowledge base.</think>";
    let result = ToolParser::parse_response(content, Some(&tool_calls_field));
    assert!(result.has_think_block);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].name, "rag_search");
    assert!(!result.is_think_only());
}



