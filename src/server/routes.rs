use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use axum::response::sse::{KeepAlive, Sse};
use async_stream;
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};
use std::sync::Arc;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::convert::Infallible;
use tokio::sync::{broadcast, mpsc};
use tokio::time::{self, interval, Duration};
use crate::context::{
    inject_system_instructions, ChatMessage, ContextManager, ANTI_HALLUCINATION_SYSTEM_PROMPT,
    SYNTHESIS_FORCING_PROMPT,
};
use crate::comfy_ui::WorkflowKind;
use crate::db::SearchResult;
use crate::error::AppError;
use crate::guardrails::InferencePermitGuard;
use crate::imagegen::{
    detect_image_request, extract_reference_image, load_prompt_file,
    try_parse_image_generate_json, GenerateRequest, GeneratedImage, HARNESS_DIRECTIVE,
};
use crate::filegen::{
    clean_filename, is_file_request, is_denied_extension, try_parse_write_file_json, FILE_DIRECTIVE,
};
use crate::files::content_type_for as file_content_type_for;
use crate::files::GeneratedFile;
use crate::images::content_type_for;
use crate::monitor::{
    ActiveRequest, DbStats, SearXNGStatus,
};
use crate::router::{has_roleplay_marker, restore_upstream_model, RequestRouter, RouteDecision, ROLEPLAY_ALLOWED_TOOLS};
use crate::server::models::ChatCompletionChunk;
use crate::state::AppState;
use crate::tools::{ExtractedToolCall, ToolParser};

pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (active_inferences, queued) = state.concurrency_limiter.stats();
    let (total_ram, free_ram, cpu_usage) = state.watchdog.get_telemetry();

    // Advertised so clients (e.g. the CLAN Flutter app) can detect A-PROX and
    // gate A-PROX-only features without a separate capability probe. `rag` is
    // always available; the rest mirror live config.
    let mut capabilities = vec!["rag"];
    if state.config.image_generation.enabled {
        capabilities.push("image");
    }
    if state.config.file_generation.enabled {
        capabilities.push("file");
    }

    Json(json!({
        "status": "healthy",
        "service": "A-PROX",
        "version": env!("CARGO_PKG_VERSION"),
        "capabilities": capabilities,
        "hardware": {
            "total_ram_gb": format!("{:.1}", total_ram),
            "available_ram_gb": format!("{:.1}", free_ram),
            "global_cpu_usage_pct": format!("{:.1}%", cpu_usage),
            "gpu_mode": "Zero-VRAM (Pure CPU Middleware)",
        },
        "concurrency": {
            "active_inferences": active_inferences,
            "queued_requests": queued,
            "max_slots": state.config.guardrails.max_concurrent_inferences,
        },
        "upstream": {
            "endpoint": state.config.upstream.base_url,
            "model_alias": state.config.upstream.model_alias,
        }
    }))
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Result<Response, AppError> {
    let upstream_url = format!("{}/v1/models", state.config.upstream.base_url.trim_end_matches('/'));

    let resp = state.http_client
        .get(&upstream_url)
        .header("Authorization", format!("Bearer {}", state.config.upstream.api_key))
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(AppError::Upstream(format!("Upstream models endpoint returned {}", resp.status())));
    }

    let mut data: Value = resp.json().await.map_err(|e| AppError::Upstream(e.to_string()))?;

    if let Some(list) = data.get_mut("data").and_then(|v| v.as_array_mut()) {
        list.push(json!({
            "id": "a-prox-auto",
            "object": "model",
            "owned_by": "a-prox",
            "description": "Dynamic Intelligent Routing with RAG and Local Web Search"
        }));
        list.push(json!({
            "id": "a-prox-rag",
            "object": "model",
            "owned_by": "a-prox",
            "description": "Embedded Vector Knowledge Store Search"
        }));
    }

    Ok(Json(data).into_response())
}

pub async fn monitor_dashboard(_state: State<Arc<AppState>>) -> impl IntoResponse {
    let html = include_str!("../monitor/dashboard.html");
    (StatusCode::OK, [("Content-Type", "text/html; charset=utf-8")], html)
}

pub async fn monitor_api(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let provided_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|s| s.to_string())
        });

    if let Some(key) = provided_key {
        if key != state.config.upstream.api_key {
            return Err(AppError::BadRequest("Invalid API key".to_string()));
        }
    }

    let searxng_running = if state.config.searxng.enabled {
        state.searxng_manager.is_running()
    } else {
        false
    };

    let searxng_status = SearXNGStatus {
        enabled: state.config.searxng.enabled,
        running: searxng_running,
        listen_port: state.config.searxng.listen_port,
    };

    let db_stats = match state.rag_engine.stats() {
        Ok((total, collections)) => DbStats {
            total_chunks: total,
            collections,
        },
        Err(_) => DbStats {
            total_chunks: -1,
            collections: Default::default(),
        },
    };

    let api_data = state.monitor.get_api_data(
        &state.watchdog,
        &state.concurrency_limiter,
        db_stats,
        searxng_status,
        &state.config,
    ).await;

    Ok(Json(api_data))
}

pub async fn monitor_statistics(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let _provided_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|s| s.to_string())
        });

    let stats = state.monitor.compute_statistics();
    Ok(Json(stats))
}

pub async fn monitor_sse(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let provided_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|s| s.to_string())
        });

    if let Some(key) = provided_key {
        if key != state.config.upstream.api_key {
            return Err(AppError::BadRequest("Invalid API key".to_string()));
        }
    }

    let receiver = state.monitor.broadcast_receiver();

    let stream = MonitorStream {
        receiver,
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

struct MonitorStream {
    receiver: broadcast::Receiver<String>,
}

impl Stream for MonitorStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.receiver.try_recv() {
            Ok(_event) => {
                Poll::Ready(Some(Ok(Event::default().data(format!("{}:{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(), _event)))))
            }
            Err(broadcast::error::TryRecvError::Empty) => Poll::Pending,
            Err(broadcast::error::TryRecvError::Lagged(_n)) => {
                Poll::Ready(Some(Ok(Event::default().data("lagged"))))
            }
            Err(broadcast::error::TryRecvError::Closed) => Poll::Ready(None),
        }
    }
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> Result<Response, AppError> {
    let (is_safe, free_gb) = state.watchdog.is_memory_safe();
    if !is_safe {
        return Err(AppError::ResourceExhausted(format!(
            "Host memory critical: only {:.2} GB RAM available (minimum required: {:.2} GB)",
            free_gb, state.config.guardrails.min_free_ram_gb
        )));
    }

    let raw_messages = payload.get("messages")
        .cloned()
        .ok_or_else(|| AppError::BadRequest("Missing 'messages' array".to_string()))?;

    let messages: Vec<ChatMessage> = serde_json::from_value(raw_messages)
        .map_err(|e| AppError::BadRequest(format!("Invalid messages format: {e}")))?;

    let (mut pruned_messages, was_pruned) = state.context_mgr.prune_messages(messages);
    if was_pruned {
        payload["messages"] = serde_json::to_value(&pruned_messages)
            .map_err(|e| AppError::Context(e.to_string()))?;
    }

    let _permit = state.concurrency_limiter.acquire_permit().await?;

    let bypass_header = headers.get("x-proxy-bypass")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let tools_payload = payload.get("tools");
    let model_name = payload.get("model").and_then(|v| v.as_str());
    
    // Debug: log what the router sees
    let latest_user_content = pruned_messages.iter().rev().find(|m| m.role == "user").map(|m| m.content_as_str()).unwrap_or_default();
    tracing::debug!("Router input: model={:?}, tools={:?}, latest_user_msg={}", model_name, tools_payload.is_some(), latest_user_content.chars().take(200).collect::<String>());
    
    let decision = RequestRouter::classify_request(&pruned_messages, tools_payload, bypass_header, state.config.guardrails.enable_agentic_tools, model_name, Some(&*state.intent_classifier), Some(&state.config.intent), Some(&state.config.tool_commands));

    // A routing alias (`a-prox-rag`, `a-prox-agent`, …) selects a strategy, not
    // a model. Swap it back to the configured upstream model before anything is
    // forwarded, so it never reaches llama.cpp or a strict OpenAI backend.
    restore_upstream_model(&mut payload, &state.config.upstream.model_alias);

    // Optional retrieval tuning for the `RAGAugmented` route. Stripped from the
    // payload so it is never forwarded upstream as an unknown field.
    let rag = RagOptions::from_payload(&payload);
    if let Some(obj) = payload.get_mut("rag") {
        *obj = Value::Null;
    }

    // Optional visual style for image generation. Also stripped: it is an
    // A-PROX-only field, applied when the workflow is built (see
    // `ImageGenService::resolve_style`).
    let image_style = payload
        .get("image_style")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    if let Some(obj) = payload.get_mut("image_style") {
        *obj = Value::Null;
    }

    // Opt-in: the client only wants the generated artifact, not a caption or a
    // synthesized reply. Skips two upstream turns that such a client discards.
    let image_only = payload
        .get("image_only")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if let Some(obj) = payload.get_mut("image_only") {
        *obj = Value::Null;
    }


    let is_streaming = payload.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Resolve the externally-addressable base URLs for generated images/files
    // once per request, from the address the client actually dialed through
    // (config `public_base_url` → forwarded headers → `Host` → loopback).
    let image_base = artifact_base_url(
        &state.config.image_generation.public_base_url,
        &headers,
        state.config.server.port,
    );
    let file_base = artifact_base_url(
        &state.config.file_generation.public_base_url,
        &headers,
        state.config.server.port,
    );

    // Normalize route decision string for consistent statistics grouping
    let route_label = match &decision {
        RouteDecision::FastPassThrough => "FastPassThrough".to_string(),
        RouteDecision::RAGAugmented { .. } => "RAGAugmented".to_string(),
        RouteDecision::RAGIngestion { .. } => "RAGIngestion".to_string(),
        RouteDecision::AgenticToolLoop | RouteDecision::AgenticToolForced { .. } => "AgenticToolLoop".to_string(),
    };

    let request_id = state.monitor.next_request_id().await;

    let monitor_req = ActiveRequest {
        request_id,
        route_decision: route_label,
        start_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        messages_count: pruned_messages.len(),
        tokens_sent: state.context_mgr.calculate_total_tokens(&pruned_messages),
        tokens_received: 0,
        tool_calls_executed: Vec::new(),
        rag_hits: 0,
        upstream_latency_ms: None,
        is_streaming,
    };

    state.monitor.set_active(monitor_req).await;

    let start = std::time::Instant::now();
    let result = match decision {
        RouteDecision::FastPassThrough => {
            tracing::info!("Routing via Fast Pass-Through");
            forward_to_upstream(&state, payload, is_streaming, request_id, _permit).await
        }
        RouteDecision::RAGAugmented { query } => {
            tracing::info!("Routing via RAG Augmented Search for query: {:?}", query);
            let search_hits = state.rag_engine.query_rag(&query, rag.collection.as_deref(), rag.top_k)
                .unwrap_or_default();

            let rag_hits_count = search_hits.len();
            if !search_hits.is_empty() {
                let kept: Vec<SearchResult> = search_hits
                    .into_iter()
                    .filter(|r| r.score >= rag.min_score)
                    .take(rag.top_k)
                    .collect();
                if !kept.is_empty() {
                    let context_block = state.rag_engine.format_rag_context(&kept);
                    inject_context_into_messages(&mut pruned_messages, &context_block);
                    payload["messages"] = serde_json::to_value(&pruned_messages)
                        .map_err(|e| AppError::Context(e.to_string()))?;
                }
            }

            state.monitor.update_rag_hits(request_id, rag_hits_count).await;

            forward_to_upstream(&state, payload, is_streaming, request_id, _permit).await
        }
        RouteDecision::RAGIngestion { content } => {
            tracing::info!("Routing via RAG Ingestion");
            let content = content.trim();
            // Guard against storing an empty or instruction-only payload
            // (e.g. "remember this" with no data, or a leftover target phrase
            // like "the vector store"). In that case forward upstream so the
            // model can ask the user for the content to store.
            let trivial = content.is_empty() || content.split_whitespace().count() < 3;
            if trivial {
                tracing::warn!("RAG ingestion requested but message contained no substantial content to store; forwarding upstream");
                forward_to_upstream(&state, payload, is_streaming, request_id, _permit).await
            } else {
                let source_uri = format!(
                    "chat-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                );
                let chunks = state
                    .rag_engine
                    .ingest_document("default", &source_uri, content)
                    .map_err(AppError::Internal)?;

                tracing::info!(
                    "Stored user-requested content into RAG ({} chunks, source {:?})",
                    chunks,
                    source_uri
                );

                // Inform the model so it can acknowledge the save to the user.
                let note = format!(
                    "### System Note: The content from the user's latest message was successfully \
                     stored into the RAG knowledge base ({} chunks, collection 'default', source '{}'). \
                     If the user asked you to save or remember this information, confirm that it has \
                     been saved and tell them how to retrieve it later.",
                    chunks, source_uri
                );
                inject_context_into_messages(&mut pruned_messages, &note);
                payload["messages"] = serde_json::to_value(&pruned_messages)
                    .map_err(|e| AppError::Context(e.to_string()))?;

                forward_to_upstream(&state, payload, is_streaming, request_id, _permit).await
            }
        }
        RouteDecision::AgenticToolLoop => {
            tracing::info!("Routing via Agentic Function Calling Loop");
            // Inject anti-hallucination & tool guidance directives into system prompt
            inject_system_instructions(&mut pruned_messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);
            payload["messages"] = serde_json::to_value(&pruned_messages)
                .map_err(|e| AppError::Context(e.to_string()))?;

            // Detect image-generation requests and arm the image pipeline context.
            let image_gen_ctx = prepare_image_request(&state, &mut pruned_messages, &image_base);

            // Detect file-write requests and arm the write_file pipeline context.
            let file_gen_ctx = prepare_file_request(&state, &mut pruned_messages, &file_base);

            // Ensure internal tool schemas are registered if not provided
            if payload.get("tools").is_none() {
                payload["tools"] = state.tool_registry.get_internal_tools_definitions();
            }

            execute_agentic_loop(&state, payload, pruned_messages, is_streaming, request_id, _permit, image_gen_ctx, file_gen_ctx, Vec::new(), image_style.clone(), image_only).await
        }
        RouteDecision::AgenticToolForced { tools, query } => {
            tracing::info!("Routing via Agentic Function Calling Loop (forced flags {:?})", tools);
            inject_system_instructions(&mut pruned_messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);

            // Strip the flag from the latest user message so the model sees clean text.
            rewrite_latest_user_text(&mut pruned_messages, &query);
            payload["messages"] = serde_json::to_value(&pruned_messages)
                .map_err(|e| AppError::Context(e.to_string()))?;

            // Image/file flags arm their respective pipelines as well.
            let mut image_gen_ctx = None;
            let mut file_gen_ctx = None;
            if tools.iter().any(|t| t == "image_generate") {
                image_gen_ctx = prepare_forced_image_request(&state, &mut pruned_messages, &image_base);
            }
            if tools.iter().any(|t| t == "write_file") {
                file_gen_ctx = prepare_forced_file_request(&state, &mut pruned_messages, &file_base);
            }

            if payload.get("tools").is_none() {
                payload["tools"] = state.tool_registry.get_internal_tools_definitions();
            }

            execute_agentic_loop(&state, payload, pruned_messages, is_streaming, request_id, _permit, image_gen_ctx, file_gen_ctx, tools, image_style.clone(), image_only).await
        }
    };

    let duration = start.elapsed().as_secs_f64() * 1000.0;

    // Streaming responses finalize themselves once their body is fully delivered:
    // CountingStream for pass-through/RAG, and the background task for the agentic loop.
    // Only complete here for non-streaming requests, or when setup failed synchronously.
    if !is_streaming || result.is_err() {
        match &result {
            Ok(_) => {
                state.monitor.complete_request(request_id, "success", duration, None).await;
            }
            Err(e) => {
                state.monitor.complete_request(request_id, "error", duration, Some(e.to_string())).await;
            }
        }
    }

    result
}

struct KeepAliveStream<S> {
    inner: S,
    interval: time::Interval,
}

#[derive(Debug)]
struct StreamError(String);

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StreamError {}

impl<S> Stream for KeepAliveStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin + Send + 'static,
{
    type Item = Result<bytes::Bytes, StreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match Pin::new(&mut self.interval).poll_tick(cx) {
                Poll::Ready(_) => {
                    let comment = ": keepalive\n\n";
                    return Poll::Ready(Some(Ok(bytes::Bytes::from(comment))));
                }
                Poll::Pending => {}
            }

            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => return Poll::Ready(Some(Ok(chunk))),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(StreamError(e.to_string())))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Wraps an upstream SSE stream for pass-through/RAG routes. Accumulates generated
/// content to count output tokens and finalizes the monitor entry when the stream ends
/// (or is dropped because the client disconnected), so streaming requests are not left
/// stuck in "processing" with zero token/latency stats.
struct CountingStream<S> {
    inner: S,
    _permit: Option<InferencePermitGuard>,
    monitor: Arc<crate::monitor::MonitorState>,
    context_mgr: Arc<ContextManager>,
    request_id: u64,
    accumulated: String,
    leftover: String,
    latency_ms: f64,
    start: std::time::Instant,
    completed: bool,
}

impl<S> CountingStream<S> {
    fn finalize(&mut self, status: &str, error: Option<String>) {
        if self.completed {
            return;
        }
        self.completed = true;
        let tokens_received = if self.accumulated.is_empty() {
            0
        } else {
            self.context_mgr.tokenizer().count_tokens(&self.accumulated)
        };
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.monitor.finalize_stream_sync(
            self.request_id,
            tokens_received,
            Some(self.latency_ms),
            status,
            duration_ms,
            error,
        );
        self._permit.take();
    }

    fn ingest(&mut self, chunk: &[u8]) {
        let text = String::from_utf8_lossy(chunk);
        let mut buf = std::mem::take(&mut self.leftover);
        buf.push_str(&text);

        // Keep the trailing partial line for the next chunk; SSE chunks may split lines.
        let mut boundary = 0;
        for (i, _) in buf.match_indices('\n') {
            boundary = i + 1;
        }
        if boundary == 0 {
            self.leftover = buf;
            return;
        }
        let complete = buf[..boundary].to_string();
        self.leftover = buf[boundary..].to_string();

        for line in complete.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    if let Some(content) = v
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        self.accumulated.push_str(content);
                    }
                }
            }
        }
    }
}

impl<S> Stream for CountingStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, StreamError>> + Unpin + Send + 'static,
{
    type Item = Result<bytes::Bytes, StreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.ingest(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                let msg = e.to_string();
                this.finalize("error", Some(msg.clone()));
                Poll::Ready(Some(Err(StreamError(msg))))
            }
            Poll::Ready(None) => {
                this.finalize("success", None);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for CountingStream<S> {
    fn drop(&mut self) {
        self.finalize("success", None);
    }
}

async fn forward_to_upstream(
    state: &Arc<AppState>,
    payload: Value,
    is_streaming: bool,
    request_id: u64,
    permit: InferencePermitGuard,
) -> Result<Response, AppError> {
    let upstream_url = format!("{}/v1/chat/completions", state.config.upstream.base_url.trim_end_matches('/'));

    let start = std::time::Instant::now();
    let resp = state.http_client
        .post(&upstream_url)
        .header("Authorization", format!("Bearer {}", state.config.upstream.api_key))
        .json(&payload)
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;

    let status = StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    if is_streaming {
        let ttfb_ms = start.elapsed().as_secs_f64() * 1000.0;
        let upstream_stream = resp.bytes_stream();
        let keepalive_stream = KeepAliveStream {
            inner: upstream_stream,
            interval: interval(Duration::from_secs(3)),
        };
        let counting_stream = CountingStream {
            inner: keepalive_stream,
            _permit: Some(permit),
            monitor: state.monitor.clone(),
            context_mgr: state.context_mgr.clone(),
            request_id,
            accumulated: String::new(),
            leftover: String::new(),
            latency_ms: ttfb_ms,
            start,
            completed: false,
        };
        let body = Body::from_stream(counting_stream);

        Response::builder()
            .status(status)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(body)
            .map_err(|e| AppError::Internal(e.into()))
    } else {
        drop(permit);
        let data: Value = resp.json().await
            .map_err(|e| AppError::Upstream(e.to_string()))?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let tokens_received = data
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                data.get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|s| state.context_mgr.tokenizer().count_tokens(s))
                    .unwrap_or(0)
            });

        state.monitor.set_upstream_metrics(request_id, tokens_received, Some(latency_ms)).await;

        Ok((status, Json(data)).into_response())
    }
}

/// State carried through the agentic loop for an image-generation request.
/// Populated by `prepare_image_request` when the latest user turn looks like an
/// image request (attached photo → i2i, generation phrase → t2i).
struct ImageGenContext {
    request_kind: WorkflowKind,
    /// (bytes, (width, height)) of the attached reference image, if any.
    reference_image: Option<(Vec<u8>, (u32, u32))>,
    /// Filled in by `maybe_execute_image_generate` once the tool has run.
    result: Option<GeneratedImage>,
    failed: bool,
    /// The rewritten prompt used for the generation (surface in the tool result).
    last_prompt: String,
    /// Externally-addressable base URL for the served image (config override →
    /// forwarded headers → request `Host` → loopback fallback).
    public_base: String,
    /// Visual style key from the request's `image_style` field, resolved against
    /// `image_generation.styles` at generation time (after the prompt enhancer
    /// has rewritten the prompt, so the style cannot be diluted).
    style: Option<String>,
}

/// State carried through the agentic loop for a file-write request. Populated by
/// `prepare_file_request`. `active` tracks the file being appended to so that
/// `mode=append` chunks continue the same logical file across turns.
struct FileGenContext {
    active: Option<GeneratedFile>,
    result: Option<GeneratedFile>,
    failed: bool,
    /// Externally-addressable base URL for the served file (config override →
    /// forwarded headers → request `Host` → loopback fallback).
    public_base: String,
}

/// Resolve the externally-addressable base URL for generated artifacts, using
/// the request the client actually dialed through. Priority:
///   1. non-empty `cfg_override` (`[image_generation]`/`[file_generation]`
///      `public_base_url`) — explicit config always wins;
///   2. `X-Forwarded-Proto` (first of a comma-list, default `http`) +
///      `X-Forwarded-Host` (first value) — TLS reverse proxies;
///   3. the request `Host` header (LAN IP, public IP, or proxy domain);
///   4. loopback fallback `http://127.0.0.1:{server_port}` for direct local use.
fn artifact_base_url(cfg_override: &str, headers: &HeaderMap, server_port: u16) -> String {
    let trimmed = cfg_override.trim();
    if !trimmed.is_empty() {
        return trimmed.trim_end_matches('/').to_string();
    }

    let proto = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "http".to_string());

    let host = headers
        .get("x-forwarded-host")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            headers
                .get("host")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        });

    match host {
        // Reject values with whitespace; they are never a valid host.
        Some(h) if !h.chars().any(char::is_whitespace) => format!("{proto}://{h}"),
        _ => format!("http://127.0.0.1:{server_port}"),
    }
}

/// Detect an image request in the latest user turn and prepare the pipeline:
///  - selects t2i vs i2i from content/image parts (the model never chooses),
///  - appends the workflow-specific image-generator system prompt + harness
///    directive as a dedicated system message (verbatim, not merged),
///  - captures the attached reference image so `image_generate` can upload it.
fn prepare_image_request(
    state: &Arc<AppState>,
    messages: &mut Vec<ChatMessage>,
    public_base: &str,
) -> Option<ImageGenContext> {
    if !state.config.image_generation.enabled {
        return None;
    }
    let kind = detect_image_request(messages)?;

    let prompt_file = match kind {
        WorkflowKind::TextToImage => &state.config.image_generation.t2i_prompt_file,
        WorkflowKind::ImageToImage => &state.config.image_generation.i2i_prompt_file,
    };
    let system_prompt = load_prompt_file(kind, prompt_file);
    let full = format!("{}\n{}", system_prompt.trim(), HARNESS_DIRECTIVE);
    append_system_message(messages, &full);
    tracing::info!(
        "Image request detected ({:?}); injected image-generator system prompt",
        kind
    );

    let reference_image = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(extract_reference_image);

    Some(ImageGenContext {
        request_kind: kind,
        reference_image,
        result: None,
        failed: false,
        last_prompt: String::new(),
        public_base: public_base.to_string(),
        style: None,
    })
}

/// Detect a file-write request in the latest user turn and arm the write_file
/// pipeline context (mirrors `prepare_image_request`, no backend involved).
fn prepare_file_request(
    state: &Arc<AppState>,
    messages: &mut Vec<ChatMessage>,
    public_base: &str,
) -> Option<FileGenContext> {
    if !state.config.file_generation.enabled {
        return None;
    }
    if !is_file_request(messages) {
        return None;
    }
    append_system_message(messages, FILE_DIRECTIVE);
    tracing::info!("File write request detected; injected write_file directive");
    Some(FileGenContext {
        active: None,
        result: None,
        failed: false,
        public_base: public_base.to_string(),
    })
}

/// Forced `/image` flag: arm the image pipeline without keyword detection.
/// Files as i2i when the latest user turn has an attached image part (using the
/// reference dims), otherwise t2i; the model rewrites the prompt in Phase A.
fn prepare_forced_image_request(
    state: &Arc<AppState>,
    messages: &mut Vec<ChatMessage>,
    public_base: &str,
) -> Option<ImageGenContext> {
    if !state.config.image_generation.enabled {
        return None;
    }
    let reference_image = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(extract_reference_image);
    let (kind, prompt_file) = match &reference_image {
        Some(_) => (WorkflowKind::ImageToImage, &state.config.image_generation.i2i_prompt_file),
        None => (WorkflowKind::TextToImage, &state.config.image_generation.t2i_prompt_file),
    };
    let system_prompt = load_prompt_file(kind, prompt_file);
    let full = format!("{}\n{}", system_prompt.trim(), HARNESS_DIRECTIVE);
    append_system_message(messages, &full);
    tracing::info!("Forced /image flag ({:?}); injected image-generator system prompt", kind);
    Some(ImageGenContext {
        request_kind: kind,
        reference_image,
        result: None,
        failed: false,
        last_prompt: String::new(),
        public_base: public_base.to_string(),
        style: None,
    })
}

/// Forced `/file` flag: arm the write_file pipeline without keyword detection.
fn prepare_forced_file_request(
    state: &Arc<AppState>,
    messages: &mut Vec<ChatMessage>,
    public_base: &str,
) -> Option<FileGenContext> {
    if !state.config.file_generation.enabled {
        return None;
    }
    append_system_message(messages, FILE_DIRECTIVE);
    tracing::info!("Forced /file flag; injected write_file directive");
    Some(FileGenContext {
        active: None,
        result: None,
        failed: false,
        public_base: public_base.to_string(),
    })
}

/// Rewrite the latest user message text after a `/flag` is stripped. For
/// multimodal (array) content the `text` part is replaced in place so any
/// attached `image_url` parts survive (needed for `/image` i2i).
fn rewrite_latest_user_text(messages: &mut Vec<ChatMessage>, text: &str) {
    let Some(last_user) = messages.iter_mut().rev().find(|m| m.role == "user") else {
        return;
    };
    match &mut last_user.content {
        Some(Value::Array(parts)) => {
            if text.trim().is_empty() {
                return;
            }
            match parts.iter_mut().find(|p| {
                p.get("type").and_then(|t| t.as_str()) == Some("text")
            }) {
                Some(text_part) => {
                    text_part["text"] = json!(text.trim());
                }
                None => parts.insert(0, json!({"type": "text", "text": text.trim()})),
            }
        }
        _ => {
            let cleaned = text.trim();
            last_user.content = Some(json!(if cleaned.is_empty() {
                last_user.content_as_str()
            } else {
                cleaned.to_string()
            }));
        }
    }
}

/// Merge the image-generator directive into the leading system message (llama.cpp
/// 3.6's chat template raises `System message must be at the beginning` when more
/// than one system message precedes the user turn, so we cannot insert a second
/// one). Inserts a fresh system message only when none exists.
fn append_system_message(messages: &mut Vec<ChatMessage>, text: &str) {
    if let Some(first) = messages.first_mut() {
        if first.role == "system" {
            let existing = first.content_as_str();
            let merged = if existing.trim().is_empty() {
                text.to_string()
            } else {
                format!("{}\n\n{}", existing.trim_end(), text)
            };
            first.content = Some(json!(merged));
            return;
        }
    }
    messages.insert(
        0,
        ChatMessage {
            role: "system".to_string(),
            content: Some(json!(text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            roleplay: None,
        },
    );
}

/// OpenAI-compatible SSE data payload for the single `delta.image_url` event.
fn image_url_sse_payload(url: &str) -> String {
    json!({
        "choices": [{ "delta": { "image_url": { "url": url } } }]
    })
    .to_string()
}

/// OpenAI-compatible SSE data payload for the single `delta.file_url` event.
fn file_url_sse_payload(url: &str, name: &str, mime: &str) -> String {
    json!({
        "choices": [{ "delta": { "file_url": { "url": url, "name": name, "mime": mime } } }]
    })
    .to_string()
}

/// Intercept `image_generate` tool calls in the agentic loop. Returns `None`
/// when the call is a normal internal tool (delegate to the registry), otherwise
/// `Some(output_string)` — running the full llama-stop / ComfyUI / llama-restart
/// pipeline and recording the result into the context.
async fn maybe_execute_image_generate(
    state: &Arc<AppState>,
    call: &ExtractedToolCall,
    ctx: &mut Option<ImageGenContext>,
) -> Option<String> {
    if call.name != "image_generate" {
        return None;
    }
    let image_ctx = match ctx {
        Some(c) => c,
        None => {
            return Some(
                "Error: image_generate was requested but no image request context is active."
                    .to_string(),
            );
        }
    };

    if !state.config.image_generation.enabled {
        return Some("Error: image generation is disabled in the config.".to_string());
    }
    if image_ctx.failed {
        return Some(
            "Error: image generation previously failed on this request. Ask the user to retry."
                .to_string(),
        );
    }
    if let Some(img) = &image_ctx.result {
        return Some(result_json(img, &image_ctx.last_prompt));
    }

    let raw_prompt = call
        .arguments
        .get("rewritten_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if raw_prompt.is_empty() {
        image_ctx.failed = true;
        return Some("Error: image_generate requires a non-empty 'rewritten_prompt'.".to_string());
    }

    let wh_ratio = call
        .arguments
        .get("wh_ratio")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let ratio_follow = call
        .arguments
        .get("ratio_follow")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();
    let extra_negative = call
        .arguments
        .get("negative_prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let reference_dims = if ratio_follow.starts_with("<image") {
        image_ctx.reference_image.as_ref().map(|(_, d)| *d)
    } else {
        None
    };
    let reference_bytes = image_ctx
        .reference_image
        .as_ref()
        .map(|(b, _)| b.clone());

    let negative_prompt = if extra_negative.is_empty() {
        state.config.image_generation.default_negative_prompt.clone()
    } else {
        format!(
            "{}, {}",
            state.config.image_generation.default_negative_prompt.trim(),
            extra_negative
        )
    };

    let request = GenerateRequest {
        kind: image_ctx.request_kind,
        prompt: raw_prompt.clone(),
        negative_prompt,
        wh_ratio,
        reference_dims,
        reference_image: if image_ctx.request_kind == WorkflowKind::ImageToImage {
            reference_bytes
        } else {
            None
        },
        public_base: image_ctx.public_base.clone(),
        style: image_ctx.style.clone(),
    };

    match state.image_service.generate(request).await {
        Ok(img) => {
            tracing::info!("Image generated: {}", img.public_url);
            image_ctx.last_prompt = raw_prompt;
            let output = result_json(&img, &image_ctx.last_prompt);
            image_ctx.result = Some(img);
            Some(output)
        }
        Err(e) => {
            tracing::error!("Image generation error: {e}");
            image_ctx.failed = true;
            Some(format!("Error: image generation failed: {e}"))
        }
    }
}

fn result_json(img: &GeneratedImage, prompt: &str) -> String {
    json!({
        "status": "ok",
        "image_url": img.public_url,
        "image_path": img.file_path.display().to_string(),
        "prompt": prompt,
    })
    .to_string()
}

fn file_result_json(f: &GeneratedFile) -> String {
    json!({
        "status": "ok",
        "file_url": f.public_url,
        "file_name": f.name,
        "size": f.size,
        "mime": f.mime,
    })
    .to_string()
}

/// Monotonic suffix + epoch millis, mirroring the image id scheme (`gen_*`).
fn next_file_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "file_{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
        FILE_COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

/// Resolve the URLs for a written file, returned as `(served_url, client_data_url)`:
/// - `served_url` is the `{base}/files/{name}` URL (request-derived base: config
///   `public_base_url` → forwarded headers → `Host` → loopback). It is used in
///   the LLM-facing tool result so the KV cache only ever sees a small URL.
/// - `client_data_url` is a base64 data-URL of the full stored bytes, but only
///   when `[file_generation].inline_data_url` is enabled; it is emitted to the
///   client, never sent to the LLM.
fn resolve_file_public_url(
    state: &Arc<AppState>,
    fctx: &FileGenContext,
    servable_name: &str,
    fallback_bytes: &[u8],
    mime: &str,
) -> (String, Option<String>) {
    let served = file_served_url(&fctx.public_base, state.config.server.port, servable_name);
    let data = if state.config.file_generation.inline_data_url {
        let bytes = state
            .file_store
            .read(servable_name)
            .unwrap_or_else(|| fallback_bytes.to_vec());
        Some(file_data_url(&bytes, mime))
    } else {
        None
    };
    (served, data)
}

/// Base64 data-URL form of a generated file (`data:<mime>;base64,…`).
fn file_data_url(bytes: &[u8], mime: &str) -> String {
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;
    format!("data:{mime};base64,{}", B64.encode(bytes))
}

/// Served-URL form of a generated file, rooted at the resolved public base
/// (empty base → loopback fallback `http://127.0.0.1:{server_port}`).
fn file_served_url(base: &str, server_port: u16, servable_name: &str) -> String {
    let base = base.trim();
    let base = if base.is_empty() {
        format!("http://127.0.0.1:{server_port}")
    } else {
        base.trim_end_matches('/').to_string()
    };
    format!("{base}/files/{servable_name}")
}

/// URL string emitted to the client for a generated image: the inline base64
/// data-URL when `inline_data_url` is enabled, else the served URL.
fn image_client_url(img: &GeneratedImage) -> String {
    img.data_url.clone().unwrap_or_else(|| img.public_url.clone())
}

/// URL string emitted to the client for a generated file: the inline base64
/// data-URL when `inline_data_url` is enabled, else the served URL.
fn file_client_url(f: &GeneratedFile) -> String {
    f.data_url.clone().unwrap_or_else(|| f.public_url.clone())
}

/// Intercept `write_file` tool calls in the agentic loop. Returns `None` when
/// the call is not a file call (delegate to image/registry dispatch), otherwise
/// `Some(output_string)` — persisting the file, updating the context, and
/// returning metadata including the publicly reachable `/files/{name}` URL.
async fn maybe_execute_write_file(
    state: &Arc<AppState>,
    call: &ExtractedToolCall,
    ctx: &mut Option<FileGenContext>,
) -> Option<String> {
    if call.name != "write_file" {
        return None;
    }
    let fctx = match ctx {
        Some(c) => c,
        None => {
            return Some(
                "Error: write_file was requested but no file request context is active."
                    .to_string(),
            );
        }
    };

    if !state.config.file_generation.enabled {
        return Some("Error: file generation is disabled in the config.".to_string());
    }
    if fctx.failed {
        return Some(
            "Error: file writing previously failed on this request. Ask the user to retry."
                .to_string(),
        );
    }

    let filename = call
        .arguments
        .get("filename")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let content = call
        .arguments
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if content.trim().is_empty() {
        fctx.failed = true;
        return Some("Error: write_file requires a non-empty 'content' argument.".to_string());
    }

    let clean = clean_filename(&filename);
    if is_denied_extension(&clean, &state.config.file_generation.deny_exts) {
        fctx.failed = true;
        return Some(format!(
            "Error: writing files with the '{}' extension is not allowed.",
            crate::filegen::extension_of(&clean)
        ));
    }

    let mode = call
        .arguments
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("overwrite");
    let append = mode == "append";

    let cap = state.config.file_generation.max_content_chars;
    let len = content.chars().count();
    if len > cap {
        return Some(format!(
            "Error: content is {len} characters, exceeding the per-call limit of {cap}. Send the content in smaller chunks: first chunk with mode=\"overwrite\", remaining chunks with mode=\"append\" (all with the same filename)."
        ));
    }

    // Append continues the same logical file only when the filename matches the
    // active file; any other mode starts a fresh file.
    let reuse = append && fctx.active.as_ref().map(|f| f.name == clean).unwrap_or(false);
    let id = if reuse {
        fctx.active.as_ref().unwrap().id.clone()
    } else {
        next_file_id()
    };
    let servable_name = format!("{id}_{clean}");

    let path = match state.file_store.save(&servable_name, content.as_bytes(), reuse) {
        Ok(p) => p,
        Err(e) => {
            fctx.failed = true;
            return Some(format!("Error: failed to write file: {e}"));
        }
    };

    // Always resolve the emit-URLs for the artifact regardless of which table
    // (append or fresh) stored it: a served URL for LLM context, plus (in inline
    // mode) a client-only base64 data-URL.
    let mime = file_content_type_for(&servable_name);
    let (public_url, data_url) =
        resolve_file_public_url(state, fctx, &servable_name, content.as_bytes(), &mime);
    let size = state
        .file_store
        .size(&servable_name)
        .unwrap_or(content.len() as u64);
    let file = GeneratedFile {
        id,
        name: clean,
        servable_name,
        public_url,
        data_url,
        file_path: path,
        mime,
        size,
    };
    fctx.active = Some(file.clone());
    fctx.result = Some(file);
    let output = match fctx.result.as_ref() {
        Some(f) => file_result_json(f),
        None => "Error: unknown write_file state".to_string(),
    };
    Some(output)
}

/// Route one tool call: file → image → registry. Keeps the three dispatch sites
/// (loop, streaming loop, finalize) identical.
async fn dispatch_tool_execution(
    state: &Arc<AppState>,
    call: &ExtractedToolCall,
    image_ctx: &mut Option<ImageGenContext>,
    file_ctx: &mut Option<FileGenContext>,
) -> String {
    if let Some(out) = maybe_execute_write_file(state, call, file_ctx).await {
        return out;
    }
    if let Some(out) = maybe_execute_image_generate(state, call, image_ctx).await {
        return out;
    }
    state.tool_registry.execute_tool(call).await
}

/// Redact the `content` argument of successfully-executed write_file tool calls
/// when storing the assistant message in history, so multi-chunk file bodies
/// don't bloat the context window across turns. Failed calls keep their content
/// so the model can retry. `outputs` is aligned with the tool_calls array.
fn sanitize_tool_calls_for_history(tool_calls: Option<&Value>, outputs: &[String]) -> Value {
    let mut v = tool_calls.cloned().unwrap_or(Value::Array(vec![]));
    if let Some(arr) = v.as_array_mut() {
        for (i, tc) in arr.iter_mut().enumerate() {
            let name = tc.pointer("/function/name").and_then(|n| n.as_str()).unwrap_or("");
            if name != "write_file" {
                continue;
            }
            let wrote = outputs.get(i).map(|o| !o.starts_with("Error: ")).unwrap_or(false);
            if !wrote {
                continue;
            }
            match tc.pointer_mut("/function/arguments") {
                Some(Value::Object(args)) => {
                    args.insert("content".to_string(), json!("[saved to file; withheld from context]"));
                }
                Some(Value::String(s)) => {
                    if let Ok(mut obj) = serde_json::from_str::<Value>(s) {
                        if let Some(o) = obj.as_object_mut() {
                            o.insert("content".to_string(), json!("[saved to file; withheld from context]"));
                        }
                        *s = obj.to_string();
                    }
                }
                _ => {}
            }
        }
    }
    v
}

/// Push the synthetic vision user message (text + the generated image as an
/// `image_url` part pointing at A-PROX's own loopback serve endpoint) so
/// llama.cpp can caption it during the final synthesis turn. llama.cpp fetches
/// `http://127.0.0.1:{port}/images/{id}.png` as a real image (bounded vision
/// tokens via mmproj), so the multi-MB PNG never enters the KV cache as base64
/// text. This is independent of `inline_data_url`, which only affects what is
/// emitted to the client.
fn inject_generated_image_as_user_msg(
    messages: &mut Vec<ChatMessage>,
    img: &GeneratedImage,
    server_port: u16,
) {
    let caption_url = format!("http://127.0.0.1:{server_port}/images/{}.png", img.id);
    let parts = json!([
        { "type": "text", "text": "The image you were asked to generate is above. Keep your reply to the user concise: confirm the image was created, describe what it shows, and mention the image URL if helpful." },
        { "type": "image_url", "image_url": { "url": caption_url } }
    ]);
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: Some(parts),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        roleplay: None,
    });
}

/// Applies the default per-turn generation budget to an agentic-loop payload.
///
/// A-PROX deliberately does not invent a token budget for clients that set their
/// own `max_tokens` — that always wins. This only fills in the gap, because
/// otherwise the upstream default governs: llama.cpp caps at 2048, which is
/// enough for a turn that calls a tool immediately but not for an image turn,
/// where Phase A has to plan a prompt before calling `image_generate`. A
/// planning model that exceeds the cap is cut off mid-thought, emits no tool
/// call, and the request silently yields no image.
///
/// Configured via `guardrails.max_generation_tokens`; `0` keeps the upstream
/// default.
fn apply_generation_budget(state: &AppState, payload: &mut Value) {
    if payload.get("max_tokens").map_or(false, |v| !v.is_null()) {
        return; // The client asked for a specific budget.
    }
    let budget = state.config.guardrails.max_generation_tokens;
    if budget > 0 {
        payload["max_tokens"] = json!(budget);
    }
}

/// Removes any tool schema in `payload["tools"]` that is not in
/// [`ROLEPLAY_ALLOWED_TOOLS`]. Returns how many were dropped.
///
/// Filtering the serialized schemas (rather than the tool-name list) makes this
/// work uniformly no matter how the tools were armed — a `/flag` command, a
/// client-supplied `tools` array, the all-internal default, or the image/file
/// pipeline arming.
pub fn filter_armed_tools_for_roleplay(payload: &mut Value) -> usize {
    let Some(tools) = payload.get_mut("tools").and_then(|t| t.as_array_mut()) else {
        return 0;
    };
    let before = tools.len();
    tools.retain(|t| {
        t.pointer("/function/name")
            .and_then(|n| n.as_str())
            .map(|name| ROLEPLAY_ALLOWED_TOOLS.contains(&name))
            .unwrap_or(false)
    });
    before - tools.len()
}

async fn execute_agentic_loop(
    state: &Arc<AppState>,
    mut payload: Value,
    mut messages: Vec<ChatMessage>,
    is_streaming: bool,
    request_id: u64,
    permit: InferencePermitGuard,
    mut image_ctx: Option<ImageGenContext>,
    mut file_ctx: Option<FileGenContext>,
    forced_tools: Vec<String>,
    image_style: Option<String>,
    // Return the image artifact and stop — no caption turn, no synthesis turn.
    image_only: bool,
) -> Result<Response, AppError> {
    let context_mgr = state.context_mgr.clone();
    let state_ref = state.clone();
    let mut tools_in_payload = false;

    // Carry the request's visual style into the image context so the workflow
    // is built with it (the style is applied after the prompt rewrite).
    if let Some(style) = image_style {
        if let Some(ctx) = image_ctx.as_mut() {
            ctx.style = Some(style);
        }
    }

    // A /flag command restricts the armed surface to exactly the flagged tools.
    if !forced_tools.is_empty() {
        let defs: Vec<Value> = forced_tools
            .iter()
            .filter_map(|t| state_ref.tool_registry.definition_for(t))
            .collect();
        payload["tools"] = serde_json::json!(defs);
    } else if payload.get("tools").is_some() {
        tools_in_payload = true;
    } else {
        payload["tools"] = state_ref.tool_registry.get_internal_tools_definitions();
    }

    // Arm the image_generate tool schema when a context is active, so the model
    // already has it in its function-calling surface regardless of client tools.
    if image_ctx.is_some() {
        match payload.get_mut("tools").and_then(|t| t.as_array_mut()) {
            Some(arr) => {
                let armed = arr.iter().any(|t| {
                    t.pointer("/function/name").and_then(|n| n.as_str()) == Some("image_generate")
                });
                if !armed {
                    arr.push(state_ref.tool_registry.image_generate_definition());
                }
            }
            None => {
                payload["tools"] = serde_json::json!([
                    state_ref.tool_registry.image_generate_definition()
                ]);
            }
        }
    }

    // Arm the write_file tool schema when a context is active, mirroring the
    // image_generate arming above.
    if file_ctx.is_some() {
        match payload.get_mut("tools").and_then(|t| t.as_array_mut()) {
            Some(arr) => {
                let armed = arr.iter().any(|t| {
                    t.pointer("/function/name").and_then(|n| n.as_str()) == Some("write_file")
                });
                if !armed {
                    arr.push(state_ref.tool_registry.write_file_definition());
                }
            }
            None => {
                payload["tools"] = serde_json::json!([
                    state_ref.tool_registry.write_file_definition()
                ]);
            }
        }
    }

    // Roleplay hard guarantee. Applied here, after every branch above has had a
    // chance to arm tools (forced flag, client-supplied, all-internal, plus the
    // image/file pipeline arming), so it is the single choke point. A turn
    // marked `"roleplay": true` can only ever reach rag_search, rag_ingest and
    // image_generate — never web_search, web_fetch, system_time or write_file.
    if has_roleplay_marker(&messages) {
        let before = payload
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let dropped = filter_armed_tools_for_roleplay(&mut payload);
        let after = payload
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        if dropped > 0 {
            tracing::info!(
                "Roleplay request: dropped {} tool(s) outside the roleplay allow-list ({} -> {} armed)",
                dropped,
                before,
                after
            );
        }
    }

    let upstream_url = format!("{}/v1/chat/completions", state_ref.config.upstream.base_url.trim_end_matches('/'));
    let max_turns = 5;

    if !is_streaming {
        // Non-streaming path: multi-turn tool execution, then final synthesis
        let _permit_guard = permit;
        let mut total_tool_calls = Vec::new();
        let mut total_rag_hits = 0;
        let mut upstream_latency_ms: Option<f64> = None;
        let mut tools_called = false;

        for turn in 0..max_turns {
            tracing::info!("Agentic tool loop iteration {}/{}", turn + 1, max_turns);
            state_ref.monitor.heartbeat();

            let mut loop_payload = payload.clone();
            apply_generation_budget(state, &mut loop_payload);
            loop_payload["stream"] = json!(false);
            loop_payload["messages"] = serde_json::to_value(&messages)
                .map_err(|e| AppError::Context(e.to_string()))?;

            let start = std::time::Instant::now();
            state_ref.monitor.heartbeat();
            let resp = state_ref.http_client
                .post(&upstream_url)
                .header("Authorization", format!("Bearer {}", state_ref.config.upstream.api_key))
                .json(&loop_payload)
                .send()
                .await
                .map_err(|e| AppError::Upstream(e.to_string()))?;
            state_ref.monitor.heartbeat();
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            if upstream_latency_ms.is_none() {
                upstream_latency_ms = Some(latency);
            }

            if !resp.status().is_success() {
                return Err(AppError::Upstream(format!("Upstream error during tool turn: {}", resp.status())));
            }

            let resp_json: Value = resp.json().await
                .map_err(|e| AppError::Upstream(e.to_string()))?;

            let choice = resp_json.get("choices")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first());

            let assistant_msg = choice.and_then(|c| c.get("message"));
            let content_str = assistant_msg.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("");
            let tool_calls_field = assistant_msg.and_then(|m| m.get("tool_calls"));

            let result = ToolParser::parse_response(content_str, tool_calls_field);

            // Handle think-only responses: continue the loop instead of terminating
            if result.is_think_only() {
                tracing::info!("Agentic turn {} produced think-only response, continuing loop", turn + 1);
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(json!(result.raw_content)),
                    name: None,
                    tool_calls: tool_calls_field.cloned(),
                    tool_call_id: None,
                    roleplay: None,
                });
                continue;
            }

            let detected_tools = result.tool_calls;

            // Bare-JSON fallback: when an image request is armed but the model did
            // not emit a structured tool call, treat single-line `image_generate`
            // JSON (with optional surrounding prose) as the tool action.
            if detected_tools.is_empty() && image_ctx.is_some() {
                if let Some(call) = try_parse_image_generate_json(&result.raw_content) {
                    tracing::info!(
                        "Bare image_generate JSON fallback triggered on turn {} (no structured tool call)",
                        turn + 1
                    );
                    tools_called = true;
                    total_tool_calls.push("image_generate".to_string());
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(json!(result.raw_content)),
                        name: None,
                        tool_calls: tool_calls_field.cloned(),
                        tool_call_id: None,
                        roleplay: None,
                    });

                    let tool_output =
                        maybe_execute_image_generate(&state_ref, &call, &mut image_ctx)
                            .await
                            .unwrap_or_else(|| "Error: image_generate unavailable".to_string());
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(json!(tool_output)),
                        name: Some("image_generate".to_string()),
                        tool_calls: None,
                        tool_call_id: Some(call.id.clone()),
                        roleplay: None,
                    });

                    state_ref
                        .monitor
                        .update_active_request(
                            request_id,
                            0,
                            total_tool_calls.clone(),
                            total_rag_hits,
                            upstream_latency_ms,
                        )
                        .await;
                    state_ref.monitor.heartbeat();
                    continue;
                }
            }

            // Bare-JSON fallback for file writes: the model skipped the structured
            // call and emitted `{"filename":...,"content":...}` (with or without
            // surrounding prose). Treat it as the write_file action.
            if detected_tools.is_empty() && file_ctx.is_some() {
                if let Some(call) = try_parse_write_file_json(&result.raw_content) {
                    tracing::info!(
                        "Bare write_file JSON fallback triggered on turn {} (no structured tool call)",
                        turn + 1
                    );
                    tools_called = true;
                    total_tool_calls.push("write_file".to_string());
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(json!(result.raw_content)),
                        name: None,
                        tool_calls: tool_calls_field.cloned(),
                        tool_call_id: None,
                        roleplay: None,
                    });

                    let tool_output =
                        dispatch_tool_execution(&state_ref, &call, &mut image_ctx, &mut file_ctx)
                            .await;
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(json!(tool_output)),
                        name: Some("write_file".to_string()),
                        tool_calls: None,
                        tool_call_id: Some(call.id.clone()),
                        roleplay: None,
                    });

                    state_ref
                        .monitor
                        .update_active_request(
                            request_id,
                            0,
                            total_tool_calls.clone(),
                            total_rag_hits,
                            upstream_latency_ms,
                        )
                        .await;
                    state_ref.monitor.heartbeat();
                    continue;
                }
            }

            if detected_tools.is_empty() && tools_called {
                // The model has finished requesting tools; break to the final synthesis turn.
                break;
            }

            if detected_tools.is_empty() {
                let tokens_received = context_mgr.tokenizer().count_tokens(&result.raw_content);
                state_ref.monitor.update_active_request(request_id, tokens_received, total_tool_calls.clone(), total_rag_hits, upstream_latency_ms).await;
                return Ok(Json(resp_json).into_response());
            }

            tools_called = true;
            for tool_call in &detected_tools {
                total_tool_calls.push(tool_call.name.clone());
            }

            // Execute every tool call first so write_file results can be redacted from
            // the assistant tool_calls payload before it is pushed to history.
            let mut outcomes: Vec<(String, String, String)> = Vec::new();
            for tool_call in &detected_tools {
                let tool_output =
                    dispatch_tool_execution(&state_ref, tool_call, &mut image_ctx, &mut file_ctx)
                        .await;
                if tool_call.name == "rag_query" || tool_call.name == "rag_search" {
                    if let Ok(results) = serde_json::from_str::<Vec<crate::db::SearchResult>>(&tool_output) {
                        total_rag_hits += results.len();
                    }
                }
                outcomes.push((tool_call.id.clone(), tool_call.name.clone(), tool_output));
            }

            // Preserve the full raw content (including think blocks) in message
            // history; write_file bodies are withheld from the tool_calls payload.
            let output_vals: Vec<String> =
                outcomes.iter().map(|o| o.2.clone()).collect();
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(json!(result.raw_content)),
                name: None,
                tool_calls: Some(sanitize_tool_calls_for_history(
                    tool_calls_field,
                    &output_vals,
                )),
                tool_call_id: None,
                roleplay: None,
            });

            for (tool_call_id, tool_name, tool_output) in outcomes {
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(json!(tool_output)),
                    name: Some(tool_name),
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id),
                    roleplay: None,
                });
            }

            state_ref.monitor.update_active_request(
                request_id, 0, total_tool_calls.clone(), total_rag_hits, upstream_latency_ms,
            ).await;
            state_ref.monitor.heartbeat();
        }

        // Final synthesis turn(s) after tool executions (max turns reached without natural
        // termination, or the model indicated it is done requesting tools). Some models keep
        // requesting tools even after the loop budget has been exhausted and even after
        // synthesis is forced. Handle that resiliently: execute any residual tool calls and
        // re-prompt for a final answer, bounded by a few extra rounds, so the client is never
        // handed a message whose `content` is empty but still contains tool calls.
        if tools_called {
            let mut finalize_round = 0;
            let mut vision_injected = false;
            loop {
                finalize_round += 1;
                // Phase C: hand the generated image back to llama.cpp (vision) so it
                // can caption it. Injected once, before the very first final payload.
                let generated = image_ctx.as_ref().and_then(|c| c.result.as_ref());
                if let Some(img) = generated {
                    if !vision_injected {
                        inject_generated_image_as_user_msg(&mut messages, img, state.config.server.port);
                        vision_injected = true;
                    }
                }
                inject_system_instructions(&mut messages, SYNTHESIS_FORCING_PROMPT);
                let mut final_payload = payload.clone();
                apply_generation_budget(state, &mut final_payload);
                final_payload["messages"] = serde_json::to_value(&messages)
                    .map_err(|e| AppError::Context(e.to_string()))?;
                final_payload["stream"] = json!(false);
                if !tools_in_payload {
                    final_payload["tools"] = serde_json::json!([]);
                }

                let resp = state_ref.http_client
                    .post(&upstream_url)
                    .header("Authorization", format!("Bearer {}", state_ref.config.upstream.api_key))
                    .json(&final_payload)
                    .send()
                    .await
                    .map_err(|e| AppError::Upstream(e.to_string()))?;

                if !resp.status().is_success() {
                    return Err(AppError::Upstream(format!("Upstream error on final synthesis turn: {}", resp.status())));
                }

                let data: Value = resp.json().await
                    .map_err(|e| AppError::Upstream(e.to_string()))?;

                let choice = data.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first());

                let assistant_msg = choice.and_then(|c| c.get("message"));
                let content_str = assistant_msg.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("");
                let tool_calls_field = assistant_msg.and_then(|m| m.get("tool_calls"));

                let result = ToolParser::parse_response(content_str, tool_calls_field);

                // The model still wants to call tools: execute them so the context is complete,
                // then force synthesis again (bounded by a few rounds).
                if !result.tool_calls.is_empty() && finalize_round < 4 {
                    tracing::info!("Finalize round {} still requested tools, executing before forcing synthesis", finalize_round);
                    for tool_call in &result.tool_calls {
                        total_tool_calls.push(tool_call.name.clone());
                    }
                    let mut outcomes: Vec<(String, String, String)> = Vec::new();
                    for tool_call in &result.tool_calls {
                        let tool_output = dispatch_tool_execution(
                            &state_ref, tool_call, &mut image_ctx, &mut file_ctx,
                        )
                        .await;
                        if tool_call.name == "rag_query" || tool_call.name == "rag_search" {
                            if let Ok(results) = serde_json::from_str::<Vec<crate::db::SearchResult>>(&tool_output) {
                                total_rag_hits += results.len();
                            }
                        }
                        outcomes.push((tool_call.id.clone(), tool_call.name.clone(), tool_output));
                    }

                    let output_vals: Vec<String> =
                        outcomes.iter().map(|o| o.2.clone()).collect();
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(json!(result.raw_content)),
                        name: None,
                        tool_calls: Some(sanitize_tool_calls_for_history(
                            tool_calls_field,
                            &output_vals,
                        )),
                        tool_call_id: None,
                        roleplay: None,
                    });

                    for (tool_call_id, tool_name, tool_output) in outcomes {
                        messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(json!(tool_output)),
                            name: Some(tool_name),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id),
                            roleplay: None,
                        });
                    }
                    continue;
                }

                let tokens_received = data.get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|s| context_mgr.tokenizer().count_tokens(s))
                    .unwrap_or(0);

                state_ref.monitor.update_active_request(
                    request_id, tokens_received, total_tool_calls, total_rag_hits, upstream_latency_ms,
                ).await;

                let mut final_data = data;
                let generated = image_ctx.as_ref().and_then(|c| c.result.as_ref());
                if let Some(img) = generated {
                    final_data["image_url"] = json!({ "url": image_client_url(img) });
                }
                if let Some(f) = file_ctx.as_ref().and_then(|c| c.result.as_ref()) {
                    final_data["file_url"] = json!({
                        "url": file_client_url(f),
                        "name": f.name,
                        "mime": f.mime,
                        "size": f.size,
                    });
                }

                return Ok(Json(final_data).into_response());
            }
        }

        Err(AppError::Upstream("Agentic loop finished without generating a response".to_string()))
    } else {
        // Streaming path: background task + channel + keepalive ticker + stream: true for final synthesis
        let (tx, mut rx) = mpsc::channel::<bytes::Bytes>(128);
        let tx_for_task = tx.clone();
        let stream_state = state.clone();
        let context_mgr_for_spawn = context_mgr.clone();

        let stream_task = tokio::spawn(async move {
            let _permit_guard = permit;
            let mut total_tool_calls: Vec<String> = Vec::new();
            let mut total_rag_hits = 0;
            let mut upstream_latency_ms: Option<f64> = None;
            let mut tools_called = false;
            let first_turn = true;

            // Immediate TTFB
            let _ = tx_for_task.send(bytes::Bytes::from(": keepalive\n\n")).await;

            for turn in 0..max_turns {
                stream_state.monitor.heartbeat();
                let _ = tx_for_task.send(bytes::Bytes::from(format!(": turn {} evaluating\n\n", turn + 1))).await;

                let mut loop_payload = payload.clone();
                apply_generation_budget(&stream_state, &mut loop_payload);
                loop_payload["stream"] = json!(false);
                loop_payload["messages"] = match serde_json::to_value(&messages) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":\"{}\"}}\n\n", escape_sse_string(&e.to_string())))).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                        let _ = stream_state.monitor.complete_request(request_id, "error", 0.0, Some(e.to_string())).await;
                        return;
                    }
                };

                let start = std::time::Instant::now();
                stream_state.monitor.heartbeat();
                let resp = match stream_state.http_client
                    .post(&upstream_url)
                    .header("Authorization", format!("Bearer {}", stream_state.config.upstream.api_key))
                    .json(&loop_payload)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":{{\"message\":\"{}\",\"type\":\"upstream_error\"}}}}\n\n", escape_sse_string(&e.to_string())))).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                        let _ = stream_state.monitor.complete_request(request_id, "error", 0.0, Some(e.to_string())).await;
                        return;
                    }
                };
                stream_state.monitor.heartbeat();
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                if upstream_latency_ms.is_none() {
                    upstream_latency_ms = Some(latency);
                }

                if !resp.status().is_success() {
                    let err_msg = format!("Upstream error: {}", resp.status());
                    let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":{{\"message\":\"{}\",\"type\":\"upstream_error\"}}}}\n\n", escape_sse_string(&err_msg)))).await;
                    let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                    let _ = stream_state.monitor.complete_request(request_id, "error", upstream_latency_ms.unwrap_or(0.0), Some(err_msg)).await;
                    return;
                }

                let resp_json: Value = match resp.json().await {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":{{\"message\":\"{}\",\"type\":\"upstream_error\"}}}}\n\n", escape_sse_string(&e.to_string())))).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                        let _ = stream_state.monitor.complete_request(request_id, "error", upstream_latency_ms.unwrap_or(0.0), Some(e.to_string())).await;
                        return;
                    }
                };

                let choice = resp_json.get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first());

                let assistant_msg = choice.and_then(|c| c.get("message"));
                let content_str = assistant_msg.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("");
                let tool_calls_field = assistant_msg.and_then(|m| m.get("tool_calls"));

                let result = ToolParser::parse_response(content_str, tool_calls_field);

                // Handle think-only responses: continue the loop
                if result.is_think_only() {
                    tracing::info!("Agentic turn {} produced think-only response, continuing loop", turn + 1);
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(json!(result.raw_content)),
                        name: None,
                        tool_calls: tool_calls_field.cloned(),
                        tool_call_id: None,
                        roleplay: None,
                    });
                    continue;
                }

                let detected_tools = result.tool_calls;

                // Bare-JSON fallback (mirrors the non-streaming path): when an image
                // request is armed and the model emitted bare `image_generate` JSON
                // instead of a structured tool call, execute it as the tool action.
                if detected_tools.is_empty() && image_ctx.is_some() {
                    if let Some(call) = try_parse_image_generate_json(&result.raw_content) {
                        tracing::info!(
                            "Streaming bare image_generate JSON fallback on turn {} (no structured tool call)",
                            turn + 1
                        );
                        tools_called = true;
                        total_tool_calls.push("image_generate".to_string());
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(json!(result.raw_content)),
                            name: None,
                            tool_calls: tool_calls_field.cloned(),
                            tool_call_id: None,
                            roleplay: None,
                        });

                        let tool_output =
                            maybe_execute_image_generate(&stream_state, &call, &mut image_ctx)
                                .await
                                .unwrap_or_else(|| "Error: image_generate unavailable".to_string());
                        messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(json!(tool_output)),
                            name: Some("image_generate".to_string()),
                            tool_calls: None,
                            tool_call_id: Some(call.id.clone()),
                            roleplay: None,
                        });

                        stream_state
                            .monitor
                            .update_active_request(
                                request_id,
                                0,
                                total_tool_calls.clone(),
                                total_rag_hits,
                                upstream_latency_ms,
                            )
                            .await;
                        stream_state.monitor.heartbeat();
                        continue;
                    }
                }

                // Bare-JSON fallback for file writes (mirrors the non-streaming path).
                if detected_tools.is_empty() && file_ctx.is_some() {
                    if let Some(call) = try_parse_write_file_json(&result.raw_content) {
                        tracing::info!(
                            "Streaming bare write_file JSON fallback on turn {} (no structured tool call)",
                            turn + 1
                        );
                        tools_called = true;
                        total_tool_calls.push("write_file".to_string());
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: Some(json!(result.raw_content)),
                            name: None,
                            tool_calls: tool_calls_field.cloned(),
                            tool_call_id: None,
                            roleplay: None,
                        });

                        let tool_output = dispatch_tool_execution(
                            &stream_state, &call, &mut image_ctx, &mut file_ctx,
                        )
                        .await;
                        messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(json!(tool_output)),
                            name: Some("write_file".to_string()),
                            tool_calls: None,
                            tool_call_id: Some(call.id.clone()),
                            roleplay: None,
                        });

                        stream_state
                            .monitor
                            .update_active_request(
                                request_id,
                                0,
                                total_tool_calls.clone(),
                                total_rag_hits,
                                upstream_latency_ms,
                            )
                            .await;
                        stream_state.monitor.heartbeat();
                        continue;
                    }
                }

                if detected_tools.is_empty() {
                    if !tools_called {
                        // Model directly answered without needing tools on the first turn
                        let chunk_id = format!("chatcmpl-{}", request_id);
                        let role_chunk = ChatCompletionChunk::role_chunk(&chunk_id, "a-prox-agent", "assistant");
                        if tx_for_task.send(bytes::Bytes::from(role_chunk.to_sse_event())).await.is_err() {
                            return;
                        }

                        for chunk in result.raw_content.as_bytes().chunks(32) {
                            if let Ok(s) = std::str::from_utf8(chunk) {
                                let delta = ChatCompletionChunk::content_delta(&chunk_id, "a-prox-agent", s);
                                if tx_for_task.send(bytes::Bytes::from(delta.to_sse_event())).await.is_err() {
                                    return;
                                }
                            }
                        }

                        let finish = ChatCompletionChunk::finish_chunk(&chunk_id, "a-prox-agent", "stop");
                        let _ = tx_for_task.send(bytes::Bytes::from(finish.to_sse_event())).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;

                        let tokens_received = context_mgr_for_spawn.tokenizer().count_tokens(&result.raw_content);
                        stream_state.monitor.update_active_request(request_id, tokens_received, total_tool_calls, total_rag_hits, upstream_latency_ms).await;
                        let _ = stream_state.monitor.complete_request(request_id, "success", upstream_latency_ms.unwrap_or(0.0), None).await;
                        return;
                    } else {
                        // Tools were called and executed in earlier turn(s). Break to execute final synthesis streaming.
                        break;
                    }
                }

                tools_called = true;
                for tool_call in &detected_tools {
                    total_tool_calls.push(tool_call.name.clone());
                }

                // Execute every tool call first so write_file results can be redacted
                // from the assistant tool_calls payload before it is pushed to history.
                let mut outcomes: Vec<(String, String, String)> = Vec::new();
                for tool_call in &detected_tools {
                    let _ = tx_for_task.send(bytes::Bytes::from(format!(": executing tool {}\n\n", tool_call.name))).await;
                    let tool_output = dispatch_tool_execution(
                        &stream_state, tool_call, &mut image_ctx, &mut file_ctx,
                    )
                    .await;

                    if tool_call.name == "rag_query" || tool_call.name == "rag_search" {
                        if let Ok(results) = serde_json::from_str::<Vec<crate::db::SearchResult>>(&tool_output) {
                            total_rag_hits += results.len();
                        }
                    }
                    outcomes.push((tool_call.id.clone(), tool_call.name.clone(), tool_output));
                }

                // Image-only mode: the client asked for the artifact and nothing
                // else, so return as soon as the image exists.
                //
                // Skipping here avoids two whole upstream turns that this client
                // would discard anyway:
                //   1. the next loop iteration (which would ask the model to
                //      synthesize an answer), and
                //   2. Phase C — handing the image back to the vision model so it
                //      can stream a caption (~9k prompt tokens), plus the
                //      non-streaming synthesis fallback when that caption comes
                //      back empty.
                //
                // The `delta.image_url` event is emitted by A-PROX, not by the
                // model, so the artifact does not depend on any of that.
                //
                // Only short-circuits on success: a failed generation falls
                // through so the model can report *why* it failed, exactly as
                // before.
                if image_only {
                    if let Some(img) = image_ctx.as_ref().and_then(|c| c.result.as_ref()) {
                        let url = image_client_url(img);
                        let event = image_url_sse_payload(&url);
                        let _ = tx_for_task
                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                            .await;
                        let chunk_id = format!("chatcmpl-{}", request_id);
                        let finish = ChatCompletionChunk::finish_chunk(&chunk_id, "a-prox-agent", "stop");
                        let _ = tx_for_task.send(bytes::Bytes::from(finish.to_sse_event())).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                        tracing::info!(
                            "Image-only mode: returned delta.image_url without a caption or synthesis turn"
                        );
                        stream_state.monitor.update_active_request(
                            request_id, 0, total_tool_calls.clone(), total_rag_hits, upstream_latency_ms,
                        ).await;
                        let _ = stream_state.monitor.complete_request(
                            request_id, "success", upstream_latency_ms.unwrap_or(0.0), None,
                        ).await;
                        return;
                    }
                }

                let output_vals: Vec<String> =
                    outcomes.iter().map(|o| o.2.clone()).collect();
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(json!(result.raw_content)),
                    name: None,
                    tool_calls: Some(sanitize_tool_calls_for_history(
                        tool_calls_field,
                        &output_vals,
                    )),
                    tool_call_id: None,
                    roleplay: None,
                });

                for (tool_call_id, tool_name, tool_output) in outcomes {
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(json!(tool_output)),
                        name: Some(tool_name),
                        tool_calls: None,
                        tool_call_id: Some(tool_call_id),
                        roleplay: None,
                    });
                }

                stream_state.monitor.update_active_request(
                    request_id, 0, total_tool_calls.clone(), total_rag_hits, upstream_latency_ms,
                ).await;
                stream_state.monitor.heartbeat();
            }

            // Phase 3: True token streaming for final synthesis with stream: true
            if tools_called || !first_turn {
                // Phase C: hand the generated image back to llama.cpp (vision) so it
                // can caption it during synthesis streaming.
                let generated = image_ctx.as_ref().and_then(|c| c.result.as_ref());
                if let Some(img) = generated {
                    inject_generated_image_as_user_msg(&mut messages, img, stream_state.config.server.port);
                }
                let pending_image_url = image_ctx
                    .as_ref()
                    .and_then(|c| c.result.as_ref())
                    .map(|i| image_client_url(i));
                let pending_file = file_ctx
                    .as_ref()
                    .and_then(|c| c.result.as_ref())
                    .cloned();

                let mut final_payload = payload.clone();
                if tools_called {
                    inject_system_instructions(&mut messages, SYNTHESIS_FORCING_PROMPT);
                    if !tools_in_payload {
                        final_payload["tools"] = serde_json::json!([]);
                    }
                }
                final_payload["messages"] = match serde_json::to_value(&messages) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":{{\"message\":\"{}\",\"type\":\"context_error\"}}}}\n\n", escape_sse_string(&e.to_string())))).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                        let _ = stream_state.monitor.complete_request(request_id, "error", upstream_latency_ms.unwrap_or(0.0), Some(e.to_string())).await;
                        return;
                    }
                };
                apply_generation_budget(&stream_state, &mut final_payload);
                final_payload["stream"] = json!(true);

                let stream_resp = match stream_state.http_client
                    .post(&upstream_url)
                    .header("Authorization", format!("Bearer {}", stream_state.config.upstream.api_key))
                    .json(&final_payload)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":{{\"message\":\"{}\",\"type\":\"upstream_error\"}}}}\n\n", escape_sse_string(&e.to_string())))).await;
                        let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                        let _ = stream_state.monitor.complete_request(request_id, "error", upstream_latency_ms.unwrap_or(0.0), Some(e.to_string())).await;
                        return;
                    }
                };

                if !stream_resp.status().is_success() {
                    let err_msg = format!("Upstream synthesis streaming error: {}", stream_resp.status());
                    let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":{{\"message\":\"{}\",\"type\":\"upstream_error\"}}}}\n\n", escape_sse_string(&err_msg)))).await;
                    let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                    let _ = stream_state.monitor.complete_request(request_id, "error", upstream_latency_ms.unwrap_or(0.0), Some(err_msg)).await;
                    return;
                }

                let mut byte_stream = stream_resp.bytes_stream();
                let mut accumulated_text = String::new();
                let mut has_done = false;
                let mut buffered: Vec<bytes::Bytes> = Vec::new();

                while let Some(chunk_res) = byte_stream.next().await {
                    match chunk_res {
                        Ok(chunk) => {
                            if chunk.windows(6).any(|w| w == b"[DONE]") {
                                has_done = true;
                            }
                            if let Ok(text) = std::str::from_utf8(&chunk) {
                                for line in text.lines() {
                                    if let Some(data_str) = line.strip_prefix("data: ") {
                                        if let Ok(v) = serde_json::from_str::<Value>(data_str) {
                                            if let Some(content) = v.get("choices")
                                                .and_then(|c| c.get(0))
                                                .and_then(|c| c.get("delta"))
                                                .and_then(|d| d.get("content"))
                                                .and_then(|c| c.as_str())
                                            {
                                                accumulated_text.push_str(content);
                                            }
                                        }
                                    }
                                }
                            }
                            buffered.push(chunk);
                        }
                        Err(e) => {
                            tracing::warn!("Error reading upstream chunk: {e}");
                            break;
                        }
                    }
                }

                // Some models keep emitting tool calls even after tools have been stripped and
                // synthesis was forced. If the streaming synthesis produced no content at all,
                // run one non-streaming forcing attempt so the client still receives an answer
                // instead of an empty, [DONE]-terminated stream.
                if accumulated_text.trim().is_empty() {
                    tracing::info!("Streaming synthesis produced no content, forcing a non-streaming fallback");
                    inject_system_instructions(&mut messages, SYNTHESIS_FORCING_PROMPT);
                    let mut fb_payload = payload.clone();
                    fb_payload["messages"] = match serde_json::to_value(&messages) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = tx_for_task.send(bytes::Bytes::from(format!("data: {{\"error\":\"{}\"}}\n\n", escape_sse_string(&e.to_string())))).await;
                            let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                            let _ = stream_state.monitor.complete_request(request_id, "error", upstream_latency_ms.unwrap_or(0.0), Some(e.to_string())).await;
                            return;
                        }
                    };
                    if !tools_in_payload {
                        fb_payload["tools"] = serde_json::json!([]);
                    }
                    apply_generation_budget(&stream_state, &mut fb_payload);
                    fb_payload["stream"] = json!(false);

                    if let Ok(fb_resp) = stream_state.http_client
                        .post(&upstream_url)
                        .header("Authorization", format!("Bearer {}", stream_state.config.upstream.api_key))
                        .json(&fb_payload)
                        .send()
                        .await
                    {
                        if fb_resp.status().is_success() {
                            if let Ok(fb_data) = fb_resp.json::<Value>().await {
                                let fb_content = fb_data.get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("message"))
                                    .and_then(|m| m.get("content"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("");
                                if !fb_content.trim().is_empty() {
                                    if let Some(url) = &pending_image_url {
                                        let event = image_url_sse_payload(url);
                                        let _ = tx_for_task
                                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                                            .await;
                                    }
                                    if let Some(f) = &pending_file {
                                        let f_url = file_client_url(f);
                                        let event = file_url_sse_payload(&f_url, &f.name, f.mime);
                                        let _ = tx_for_task
                                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                                            .await;
                                    }
                                    let chunk_id = format!("chatcmpl-{}", request_id);
                                    for chunk in fb_content.as_bytes().chunks(32) {
                                        if let Ok(s) = std::str::from_utf8(chunk) {
                                            let delta = ChatCompletionChunk::content_delta(&chunk_id, "a-prox-agent", s);
                                            if tx_for_task.send(bytes::Bytes::from(delta.to_sse_event())).await.is_err() {
                                                break;
                                            }
                                        }
                                    }
                                    accumulated_text.push_str(fb_content);
                                }
                            }
                        }
                    }
                }

                // Only forward the (possibly re-synthesized) content once we know it is non-empty.
                if accumulated_text.trim().is_empty() {
                    // Still deliver the artifact URLs (if any were generated) so the
                    // client never loses them, even when synthesis returned no words.
                    if let Some(url) = &pending_image_url {
                        let event = image_url_sse_payload(url);
                        let _ = tx_for_task
                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                            .await;
                    }
                    if let Some(f) = &pending_file {
                        let f_url = file_client_url(f);
                        let event = file_url_sse_payload(&f_url, &f.name, f.mime);
                        let _ = tx_for_task
                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                            .await;
                    }
                    let _ = tx_for_task.send(bytes::Bytes::from(": synthesis returned no content\n\n")).await;
                } else {
                    // Phase C wire contract: single `delta.image_url` (and `delta.file_url`)
                    // events as the very first deltas (after the upstream role chunk), before
                    // any text.
                    if let Some(url) = &pending_image_url {
                        let event = image_url_sse_payload(url);
                        let _ = tx_for_task
                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                            .await;
                    }
                    if let Some(f) = &pending_file {
                        let f_url = file_client_url(f);
                        let event = file_url_sse_payload(&f_url, &f.name, f.mime);
                        let _ = tx_for_task
                            .send(bytes::Bytes::from(format!("data: {event}\n\n")))
                            .await;
                    }
                    for chunk in buffered.iter() {
                        if tx_for_task.send(chunk.clone()).await.is_err() {
                            tracing::info!("Client disconnected during agentic streaming synthesis");
                            return;
                        }
                    }
                }

                if !has_done {
                    let _ = tx_for_task.send(bytes::Bytes::from("data: [DONE]\n\n")).await;
                }

                let tokens_received = context_mgr_for_spawn.tokenizer().count_tokens(&accumulated_text);
                stream_state.monitor.update_active_request(
                    request_id, tokens_received, total_tool_calls, total_rag_hits, upstream_latency_ms,
                ).await;
                let _ = stream_state.monitor.complete_request(request_id, "success", upstream_latency_ms.unwrap_or(0.0), None).await;
            }
        });

        let mut ticker = interval(Duration::from_secs(2));
        ticker.tick().await;

        let combined = async_stream::stream! {
            loop {
                tokio::select! {
                    biased;

                    received = rx.recv() => {
                        match received {
                            Some(chunk) => {
                                yield Ok::<bytes::Bytes, Infallible>(chunk);
                            }
                            None => {
                                break;
                            }
                        }
                    }

                    _ = ticker.tick() => {
                        yield Ok::<bytes::Bytes, Infallible>(bytes::Bytes::from(": keepalive\n\n"));
                    }
                }
            }
        };

        let body = Body::from_stream(combined);
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("X-Accel-Buffering", "no")
            .body(body)
            .map_err(|e| AppError::Internal(e.into()))?;

        tokio::spawn(async move {
            drop(tx);
            let _ = stream_task.await;
        });

        Ok(response)
    }
}

/// Escape special characters in SSE string values
fn escape_sse_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn inject_context_into_messages(messages: &mut Vec<ChatMessage>, context_block: &str) {
    if let Some(user_msg) = messages.iter_mut().rev().find(|m| m.role == "user") {
        let current_text = user_msg.content_as_str();
        let augmented = format!("{}\n\n{}", context_block, current_text);
        user_msg.content = Some(json!(augmented));
    } else {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: Some(json!(context_block)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            roleplay: None,
        });
    }
}

/// Serve a stored generated image directly from the ImageStore directory.
pub async fn serve_image(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    match state.image_store.read(&name) {
        Some(bytes) => (
            StatusCode::OK,
            [
                ("Content-Type", content_type_for(&name)),
                ("Cache-Control", "public, max-age=31536000, immutable"),
            ],
            bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Serve a stored generated text file directly from the FileStore directory.
pub async fn serve_file(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    match state.file_store.read(&name) {
        Some(bytes) => (
            StatusCode::OK,
            [
                ("Content-Type".to_string(), file_content_type_for(&name).to_string()),
                ("Content-Disposition".to_string(), format!("inline; filename=\"{}\"", name)),
                ("Cache-Control".to_string(), "public, max-age=86400".to_string()),
            ],
            bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Reindex all configured directories
pub async fn ingestion_reindex(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = state.directory_ingestor.reindex_all().await;
    Json(json!({
        "status": "success",
        "files_indexed": result.files_indexed,
        "files_skipped": result.files_skipped,
        "files_errors": result.files_errors,
        "chunks_total": result.chunks_total
    }))
}

/// Get ingestion status and statistics
pub async fn ingestion_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.directory_ingestor.get_stats().await;
    let indexed_files = state.directory_ingestor.list_indexed_files().unwrap_or_default();

    Json(json!({
        "status": "success",
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "files_errors": stats.files_errors,
        "chunks_total": stats.chunks_total,
        "indexed_files_count": indexed_files.len(),
        "indexed_files": indexed_files.iter().take(50).map(|f| json!({
            "path": f.path,
            "collection": f.collection,
            "chunks": f.chunks_count,
            "last_indexed": f.last_indexed
        })).collect::<Vec<_>>()
    }))
}

// ---------------------------------------------------------------------------
// Direct RAG store access
//
// These endpoints exist so a client can use A-PROX's RAG as a *memory backend*
// without routing a chat request through the LLM. Unlike the `/ingest` flag and
// the `RAGIngestion` route (which both forward upstream for a model
// acknowledgement, and the latter hardcodes the `default` collection), these
// are pure store operations: no generation, no inference permit, no LLM cost.
//
// A client that scopes its own memories to a `collection` per conversation
// (e.g. `clan_<characterId>_<threadId>`) keeps unrelated indexed documents out
// of its retrievals — `query_rag` searches *all* collections when the
// collection filter is `None`.
// ---------------------------------------------------------------------------

/// Default retrieval depth, matching the hardcoded value the `RAGAugmented`
/// route has always used.
const DEFAULT_RAG_TOP_K: usize = 5;
/// Upper bound on `top_k` so a client cannot ask for the entire store.
const MAX_RAG_TOP_K: usize = 50;

/// Client-supplied retrieval tuning for the `RAGAugmented` route, read from the
/// request's top-level `"rag"` object:
///
/// ```json
/// "rag": { "collection": "clan_abc_xyz", "top_k": 3, "min_score": 0.35 }
/// ```
///
/// Every field is optional; omitting the object (or any field) preserves the
/// historical behaviour of searching all collections and taking 5 hits.
#[derive(Debug, Clone)]
pub struct RagOptions {
    pub collection: Option<String>,
    pub top_k: usize,
    pub min_score: f32,
}

impl Default for RagOptions {
    fn default() -> Self {
        Self {
            collection: None,
            top_k: DEFAULT_RAG_TOP_K,
            min_score: 0.0,
        }
    }
}

impl RagOptions {
    /// Parses and clamps the `"rag"` object out of a request payload.
    pub fn from_payload(payload: &Value) -> Self {
        let mut opts = RagOptions::default();
        let Some(obj) = payload.get("rag").and_then(|v| v.as_object()) else {
            return opts;
        };

        if let Some(collection) = obj.get("collection").and_then(|v| v.as_str()) {
            let trimmed = collection.trim();
            if !trimmed.is_empty() {
                opts.collection = Some(trimmed.to_string());
            }
        }
        if let Some(top_k) = obj.get("top_k").and_then(|v| v.as_u64()) {
            opts.top_k = (top_k as usize).clamp(1, MAX_RAG_TOP_K);
        }
        if let Some(min_score) = obj.get("min_score").and_then(|v| v.as_f64()) {
            opts.min_score = min_score.clamp(0.0, 1.0) as f32;
        }
        opts
    }
}

/// Request body for `POST /rag/ingest`.
#[derive(serde::Deserialize)]
pub struct RagIngestRequest {
    pub collection: String,
    #[serde(default)]
    pub source_uri: Option<String>,
    pub content: String,
}

/// Request body for `POST /rag/query`.
#[derive(serde::Deserialize)]
pub struct RagQueryRequest {
    pub query: String,
    #[serde(default)]
    pub collection: Option<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_score: Option<f32>,
}

/// Chunk a document into the RAG store, replacing any prior version of the same
/// `(collection, source_uri)` pair.
pub async fn rag_ingest(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<RagIngestRequest>,
) -> Result<Json<Value>, AppError> {
    let collection = req.collection.trim();
    if collection.is_empty() {
        return Err(AppError::BadRequest("`collection` must not be empty".into()));
    }
    let content = req.content.trim();
    if content.is_empty() {
        return Err(AppError::BadRequest("`content` must not be empty".into()));
    }

    let source_uri = req
        .source_uri
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("api-{}", unique_source_suffix()));

    // The embedder is CPU-only, so keep it off the async runtime's core lanes
    // the same way the ingestion path does. Chunk counts are small (512-token
    // chunks), so a blocking hop is cheap and avoids starving the runtime.
    let engine = state.rag_engine.clone();
    let collection_owned = collection.to_string();
    let source_uri_clone = source_uri.clone();
    let content_owned = content.to_string();
    let chunks = tokio::task::spawn_blocking(move || {
        engine.ingest_document(&collection_owned, &source_uri_clone, &content_owned)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("ingest task failed: {e}")))??;

    tracing::info!(
        "POST /rag/ingest -> {} chunk(s) into collection {:?} (source {:?})",
        chunks,
        collection,
        source_uri
    );
    Ok(Json(json!({
        "status": "ok",
        "chunks": chunks,
        "collection": collection,
        "source_uri": source_uri,
    })))
}

/// Hybrid-search the RAG store and return the matching chunks.
pub async fn rag_query(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<RagQueryRequest>,
) -> Result<Json<Value>, AppError> {
    let query = req.query.trim();
    if query.is_empty() {
        return Err(AppError::BadRequest("`query` must not be empty".into()));
    }

    let top_k = req
        .top_k
        .unwrap_or(DEFAULT_RAG_TOP_K)
        .clamp(1, MAX_RAG_TOP_K);
    let min_score = req.min_score.unwrap_or(0.0);
    let collection = req
        .collection
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let engine = state.rag_engine.clone();
    let query_owned = query.to_string();
    let results = tokio::task::spawn_blocking(move || {
        engine.query_rag(&query_owned, collection.as_deref(), top_k)
    })
    .await
    .map_err(|e| AppError::Internal(anyhow::anyhow!("query task failed: {e}")))??;

    // Fetch a slightly deeper page than requested so the score floor discards
    // weak hits without shrinking the result set below `top_k` for free.
    let kept: Vec<Value> = results
        .into_iter()
        .filter(|r| r.score >= min_score)
        .take(top_k)
        .map(|r| {
            json!({
                "chunk_id": r.chunk_id,
                "collection": r.collection,
                "source_uri": r.source_uri,
                "chunk_index": r.chunk_index,
                "content": r.content,
                "score": r.score,
            })
        })
        .collect();

    Ok(Json(json!({
        "status": "ok",
        "results": kept,
    })))
}

/// Number of chunks stored in a collection, so a client can tell an empty
/// collection (needing a backfill) from a populated one.
pub async fn rag_collection_count(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<Json<Value>, AppError> {
    let engine = state.rag_engine.clone();
    let stats = tokio::task::spawn_blocking(move || engine.stats())
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("stats task failed: {e}")))??;
    let (total, collections) = stats;
    let chunks = collections.get(&collection).copied().unwrap_or(0);
    Ok(Json(json!({
        "status": "ok",
        "collection": collection,
        "chunks": chunks,
        "chunks_total": total,
    })))
}

/// Monotonic-ish suffix for auto-generated `source_uri` values, so two
/// ingests of different content in the same second don't collide.
fn unique_source_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod filegen_tests {
    use super::*;

    #[test]
    fn sanitize_redacts_successful_write_file_content() {
        let tool_calls = json!([
            {
                "id": "call_a",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": {"filename": "a.txt", "content": "VERY LONG BODY", "mode": "append"}
                }
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": {"query": "x", "count": 2}
                }
            },
            {
                "id": "call_c",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": {"filename": "b.txt", "content": "also long"}
                }
            }
        ]);

        // call_a succeeded, call_b is a registry tool, call_c failed (Error prefixed).
        let outputs = vec![
            "{\"status\":\"ok\",\"file_url\":\"http://x/files/f_a.txt\"}".to_string(),
            "search results here".to_string(),
            "Error: content exceeds limit 24000".to_string(),
        ];
        let sanitized = sanitize_tool_calls_for_history(Some(&tool_calls), &outputs);
        let arr = sanitized.as_array().unwrap();

        assert_eq!(
            arr[0]["function"]["arguments"]["content"],
            json!("[saved to file; withheld from context]")
        );
        assert_eq!(arr[1]["function"]["arguments"]["content"], json!(null));
        assert_eq!(arr[1]["function"]["arguments"]["query"], json!("x"));
        assert_eq!(
            arr[2]["function"]["arguments"]["content"],
            json!("also long")
        );
    }

    #[test]
    fn sanitize_handles_string_arguments() {
        let tool_calls = json!([
            {
                "id": "call_x",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": "{\"filename\":\"s.py\",\"content\":\"print(1)\"}"
                }
            }
        ]);
        let sanitized = sanitize_tool_calls_for_history(Some(&tool_calls), &["{\"status\":\"ok\"}".to_string()]);
        let args = sanitized[0]["function"]["arguments"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["content"], json!("[saved to file; withheld from context]"));
        assert_eq!(parsed["filename"], json!("s.py"));
    }

    #[test]
    fn sanitize_none_is_empty_array() {
        let out = sanitize_tool_calls_for_history(None, &[]);
        assert_eq!(out, json!([]));
    }
}

#[cfg(test)]
mod base_url_tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderName, HeaderValue};

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                k.parse::<HeaderName>().unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn forwarded_proto_and_host_used_when_present() {
        let h = headers(&[
            ("x-forwarded-proto", "https"),
            ("x-forwarded-host", "chat.example.com"),
            ("host", "127.0.0.1:8000"),
        ]);
        assert_eq!(artifact_base_url("", &h, 8000), "https://chat.example.com");
    }

    #[test]
    fn forwarded_host_wins_over_plain_host() {
        let h = headers(&[("x-forwarded-host", "proxy.local"), ("host", "127.0.0.1:8000")]);
        assert_eq!(artifact_base_url("", &h, 8000), "http://proxy.local");
    }

    #[test]
    fn multi_value_forwarded_headers_take_first() {
        let h = headers(&[
            ("x-forwarded-proto", "https, http"),
            ("x-forwarded-host", "a.example.com, b.example.com"),
        ]);
        assert_eq!(artifact_base_url("", &h, 8000), "https://a.example.com");
    }

    #[test]
    fn plain_host_header_used_when_no_forwarded_headers() {
        let h = headers(&[("host", "192.168.1.5:8000")]);
        assert_eq!(artifact_base_url("", &h, 8000), "http://192.168.1.5:8000");
    }

    #[test]
    fn missing_headers_fall_back_to_loopback() {
        assert_eq!(artifact_base_url("", &HeaderMap::new(), 7777), "http://127.0.0.1:7777");
    }

    #[test]
    fn whitespace_host_is_rejected() {
        let h = headers(&[("host", "bad host with spaces")]);
        assert_eq!(artifact_base_url("", &h, 8000), "http://127.0.0.1:8000");
    }

    #[test]
    fn config_override_wins_over_headers() {
        let h = headers(&[
            ("x-forwarded-proto", "https"),
            ("x-forwarded-host", "chat.example.com"),
        ]);
        assert_eq!(
            artifact_base_url("https://cdn.example.com/base/", &h, 8000),
            "https://cdn.example.com/base"
        );
    }

    #[test]
    fn default_proto_is_http() {
        let h = headers(&[("x-forwarded-host", "a.example.com")]);
        assert_eq!(artifact_base_url("", &h, 8000), "http://a.example.com");
    }

    #[test]
    fn file_served_url_uses_base_and_trims_slash() {
        assert_eq!(file_served_url("http://host:8000/", 8000, "f_1_a.txt"), "http://host:8000/files/f_1_a.txt");
        assert_eq!(file_served_url("http://host:8000", 8000, "f_1_a.txt"), "http://host:8000/files/f_1_a.txt");
    }

    #[test]
    fn file_served_url_empty_base_falls_back_to_loopback() {
        assert_eq!(file_served_url("", 8000, "f_1_a.txt"), "http://127.0.0.1:8000/files/f_1_a.txt");
    }

    #[test]
    fn file_data_url_emits_data_prefix() {
        let url = file_data_url(b"hello world", "text/plain");
        assert!(url.starts_with("data:text/plain;base64,"));
        // base64("hello world") == "aGVsbG8gd29ybGQ="
        assert_eq!(url, "data:text/plain;base64,aGVsbG8gd29ybGQ=");
    }
}
