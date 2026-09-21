use axum::{
    body::Body,
    extract::State,
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
use crate::error::AppError;
use crate::guardrails::InferencePermitGuard;
use crate::monitor::{
    ActiveRequest, DbStats, SearXNGStatus,
};
use crate::router::{RequestRouter, RouteDecision};
use crate::server::models::ChatCompletionChunk;
use crate::state::AppState;
use crate::tools::ToolParser;

pub async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (active_inferences, queued) = state.concurrency_limiter.stats();
    let (total_ram, free_ram, cpu_usage) = state.watchdog.get_telemetry();

    Json(json!({
        "status": "healthy",
        "service": "A-PROX",
        "version": "0.1.0",
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
    
    let decision = RequestRouter::classify_request(&pruned_messages, tools_payload, bypass_header, state.config.guardrails.enable_agentic_tools, model_name, Some(&*state.intent_classifier), Some(&state.config.intent));

    let is_streaming = payload.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Normalize route decision string for consistent statistics grouping
    let route_label = match &decision {
        RouteDecision::FastPassThrough => "FastPassThrough".to_string(),
        RouteDecision::RAGAugmented { .. } => "RAGAugmented".to_string(),
        RouteDecision::RAGIngestion { .. } => "RAGIngestion".to_string(),
        RouteDecision::AgenticToolLoop => "AgenticToolLoop".to_string(),
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
            let search_hits = state.rag_engine.query_rag(&query, None, 5)
                .unwrap_or_default();

            let rag_hits_count = search_hits.len();
            if !search_hits.is_empty() {
                let context_block = state.rag_engine.format_rag_context(&search_hits);
                inject_context_into_messages(&mut pruned_messages, &context_block);
                payload["messages"] = serde_json::to_value(&pruned_messages)
                    .map_err(|e| AppError::Context(e.to_string()))?;
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

            // Ensure internal tool schemas are registered if not provided
            if payload.get("tools").is_none() {
                payload["tools"] = state.tool_registry.get_internal_tools_definitions();
            }

            execute_agentic_loop(&state, payload, pruned_messages, is_streaming, request_id, _permit).await
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

async fn execute_agentic_loop(
    state: &Arc<AppState>,
    mut payload: Value,
    mut messages: Vec<ChatMessage>,
    is_streaming: bool,
    request_id: u64,
    permit: InferencePermitGuard,
) -> Result<Response, AppError> {
    let context_mgr = state.context_mgr.clone();
    let state_ref = state.clone();
    let mut tools_in_payload = false;
    if payload.get("tools").is_some() {
        tools_in_payload = true;
    } else {
        payload["tools"] = state_ref.tool_registry.get_internal_tools_definitions();
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
                });
                continue;
            }

            let detected_tools = result.tool_calls;

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

            // Preserve the full raw content (including think blocks) in message history
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: Some(json!(result.raw_content)),
                name: None,
                tool_calls: tool_calls_field.cloned(),
                tool_call_id: None,
            });

            for tool_call in detected_tools {
                let tool_output = state_ref.tool_registry.execute_tool(&tool_call).await;

                if tool_call.name == "rag_query" || tool_call.name == "rag_search" {
                    if let Ok(results) = serde_json::from_str::<Vec<crate::db::SearchResult>>(&tool_output) {
                        total_rag_hits += results.len();
                    }
                }

                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(json!(tool_output)),
                    name: Some(tool_call.name.clone()),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id.clone()),
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
            loop {
                finalize_round += 1;
                inject_system_instructions(&mut messages, SYNTHESIS_FORCING_PROMPT);
                let mut final_payload = payload.clone();
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
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: Some(json!(result.raw_content)),
                        name: None,
                        tool_calls: tool_calls_field.cloned(),
                        tool_call_id: None,
                    });

                    for tool_call in result.tool_calls {
                        let tool_output = state_ref.tool_registry.execute_tool(&tool_call).await;
                        if tool_call.name == "rag_query" || tool_call.name == "rag_search" {
                            if let Ok(results) = serde_json::from_str::<Vec<crate::db::SearchResult>>(&tool_output) {
                                total_rag_hits += results.len();
                            }
                        }
                        messages.push(ChatMessage {
                            role: "tool".to_string(),
                            content: Some(json!(tool_output)),
                            name: Some(tool_call.name.clone()),
                            tool_calls: None,
                            tool_call_id: Some(tool_call.id.clone()),
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

                return Ok(Json(data).into_response());
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
                    });
                    continue;
                }

                let detected_tools = result.tool_calls;

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

                // Preserve the full raw content (including think blocks) in message history
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(json!(result.raw_content)),
                    name: None,
                    tool_calls: tool_calls_field.cloned(),
                    tool_call_id: None,
                });

                for tool_call in detected_tools {
                    let _ = tx_for_task.send(bytes::Bytes::from(format!(": executing tool {}\n\n", tool_call.name))).await;
                    let tool_output = stream_state.tool_registry.execute_tool(&tool_call).await;

                    if tool_call.name == "rag_query" || tool_call.name == "rag_search" {
                        if let Ok(results) = serde_json::from_str::<Vec<crate::db::SearchResult>>(&tool_output) {
                            total_rag_hits += results.len();
                        }
                    }

                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(json!(tool_output)),
                        name: Some(tool_call.name.clone()),
                        tool_calls: None,
                        tool_call_id: Some(tool_call.id.clone()),
                    });
                }

                stream_state.monitor.update_active_request(
                    request_id, 0, total_tool_calls.clone(), total_rag_hits, upstream_latency_ms,
                ).await;
                stream_state.monitor.heartbeat();
            }

            // Phase 3: True token streaming for final synthesis with stream: true
            if tools_called || !first_turn {
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
                    let _ = tx_for_task.send(bytes::Bytes::from(": synthesis returned no content\n\n")).await;
                } else {
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
        });
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
