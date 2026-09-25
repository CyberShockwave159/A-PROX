use crate::async_queue::models::{AsyncRequest, AsyncRequestStatus};
use crate::context::{ChatMessage, inject_system_instructions, ANTI_HALLUCINATION_SYSTEM_PROMPT};
use crate::guardrails::InferencePermitGuard;
use crate::router::{RequestRouter, RouteDecision};
use crate::server::routes::{
    execute_agentic_loop, forward_to_upstream, prepare_image_request, prepare_file_request,
    artifact_base_url, ImageGenContext, FileGenContext,
};
use crate::state::AppState;
use axum::{
    http::HeaderMap,
    response::Response,
};
use futures_util::TryStreamExt;
use serde_json::{json, Value};
use sqlx::{Row, SqlitePool};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use tracing::{error, info};

/// Background worker that processes async requests from the queue
pub struct AsyncQueueWorker {
    pool: SqlitePool,
    state: Arc<AppState>,
    semaphore: Arc<Semaphore>,
    cleanup_interval: Duration,
    ttl_hours: u64,
}

impl AsyncQueueWorker {
    pub fn new(
        pool: SqlitePool,
        state: Arc<AppState>,
        max_concurrent: usize,
        cleanup_interval_minutes: u64,
        ttl_hours: u64,
    ) -> Self {
        Self {
            pool,
            state,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            cleanup_interval: Duration::from_secs(cleanup_interval_minutes * 60),
            ttl_hours,
        }
    }

    /// Start the worker loop
    pub async fn run(self: Arc<Self>) {
        let worker = self.clone();
        tokio::spawn(async move {
            worker.cleanup_loop().await;
        });

        let worker = self.clone();
        tokio::spawn(async move {
            worker.process_loop().await;
        });

        info!("Async queue worker started (max_concurrent={}, ttl_hours={}, cleanup_interval={:?})",
            self.semaphore.available_permits(), self.ttl_hours, self.cleanup_interval);
    }

    /// Periodic cleanup of expired requests
    async fn cleanup_loop(&self) {
        let mut interval = tokio::time::interval(self.cleanup_interval);
        loop {
            interval.tick().await;
            if let Err(e) = self.cleanup_expired().await {
                error!("Async queue cleanup failed: {}", e);
            }
        }
    }

    /// Main processing loop - pulls queued requests and processes them
    async fn process_loop(self: Arc<Self>) {
        loop {
            // Fetch next queued request
            let request = match self.fetch_next_queued().await {
                Ok(Some(req)) => req,
                Ok(None) => {
                    // No work, wait a bit
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }
                Err(e) => {
                    error!("Failed to fetch queued request: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            };

            // Acquire permit and process the request
            let worker = self.clone();
            tokio::spawn(async move {
                // Acquire permit (limits concurrent processing)
                let permit = match worker.semaphore.acquire().await {
                    Ok(p) => p,
                    Err(_) => {
                        error!("Semaphore closed, stopping worker");
                        return;
                    }
                };

                worker.process_request(request).await;
                drop(permit); // Release when done
            });
        }
    }

    /// Fetch the next queued request (oldest first)
    async fn fetch_next_queued(&self) -> anyhow::Result<Option<AsyncRequest>> {
        let row = sqlx::query(
            r#"
            SELECT id, payload, status, result, created_at, updated_at, expires_at,
                   route_decision, tokens_received, error
            FROM async_requests
            WHERE status = 'queued' AND expires_at > ?
            ORDER BY created_at ASC
            LIMIT 1
            "#,
        )
        .bind(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(Self::row_to_request))
    }

    /// Mark request as processing
    async fn mark_processing(&self, request_id: &str, route_decision: Option<&str>) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        sqlx::query(
            r#"
            UPDATE async_requests
            SET status = 'processing', updated_at = ?, route_decision = ?
            WHERE id = ?
            "#,
        )
        .bind(now)
        .bind(route_decision)
        .bind(request_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update request with result
    async fn mark_completed(
        &self,
        request_id: &str,
        status: AsyncRequestStatus,
        result: Option<String>,
        route_decision: Option<&str>,
        tokens_received: u64,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        sqlx::query(
            r#"
            UPDATE async_requests
            SET status = ?, result = ?, updated_at = ?, route_decision = ?,
                tokens_received = ?, error = ?
            WHERE id = ?
            "#,
        )
        .bind(status.as_str())
        .bind(result)
        .bind(now)
        .bind(route_decision)
        .bind(tokens_received as i64)
        .bind(error)
        .bind(request_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Process a single async request
    async fn process_request(&self, request: AsyncRequest) {
        let request_id = request.id.clone();
        info!("Processing async request: {}", request_id);

        // Mark as processing
        if let Err(e) = self.mark_processing(&request_id, None).await {
            error!("Failed to mark request {} as processing: {}", request_id, e);
            return;
        }

        // Parse the payload
        let mut payload: Value = match serde_json::from_str(&request.payload) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to parse payload for request {}: {}", request_id, e);
                let _ = self.mark_completed(
                    &request_id,
                    AsyncRequestStatus::Failed,
                    None,
                    None,
                    0,
                    Some(&format!("Invalid payload: {}", e)),
                ).await;
                return;
            }
        };

        // Extract messages and prepare for routing (similar to chat_completions)
        let raw_messages = match payload.get("messages").cloned() {
            Some(m) => m,
            None => {
                error!("Request {} missing messages", request_id);
                let _ = self.mark_completed(
                    &request_id,
                    AsyncRequestStatus::Failed,
                    None,
                    None,
                    0,
                    Some("Missing 'messages' array"),
                ).await;
                return;
            }
        };

        let messages: Vec<ChatMessage> = match serde_json::from_value(raw_messages) {
            Ok(m) => m,
            Err(e) => {
                error!("Request {} invalid messages format: {}", request_id, e);
                let _ = self.mark_completed(
                    &request_id,
                    AsyncRequestStatus::Failed,
                    None,
                    None,
                    0,
                    Some(&format!("Invalid messages format: {}", e)),
                ).await;
                return;
            }
        };

        // Prune messages
        let (mut pruned_messages, was_pruned) = self.state.context_mgr.prune_messages(messages);
        if was_pruned {
            payload["messages"] = match serde_json::to_value(&pruned_messages) {
                Ok(v) => v,
                Err(e) => {
                    error!("Request {} failed to serialize pruned messages: {}", request_id, e);
                    let _ = self.mark_completed(
                        &request_id,
                        AsyncRequestStatus::Failed,
                        None,
                        None,
                        0,
                        Some(&format!("Context error: {}", e)),
                    ).await;
                    return;
                }
            };
        }

        // Classify request
        let tools_payload = payload.get("tools");
        let model_name = payload.get("model").and_then(|v| v.as_str());
        let bypass_header = false; // No headers in async context

        let decision = RequestRouter::classify_request(
            &pruned_messages,
            tools_payload,
            bypass_header,
            self.state.config.guardrails.enable_agentic_tools,
            model_name,
            Some(&*self.state.intent_classifier),
            Some(&self.state.config.intent),
            Some(&self.state.config.tool_commands),
        );

        // Resolve artifact base URLs
        let headers = HeaderMap::new(); // No real headers in background
        let image_base = artifact_base_url(
            &self.state.config.image_generation.public_base_url,
            &headers,
            self.state.config.server.port,
        );
        let file_base = artifact_base_url(
            &self.state.config.file_generation.public_base_url,
            &headers,
            self.state.config.server.port,
        );

        let route_label = match &decision {
            RouteDecision::FastPassThrough => "FastPassThrough".to_string(),
            RouteDecision::RAGAugmented { .. } => "RAGAugmented".to_string(),
            RouteDecision::RAGIngestion { .. } => "RAGIngestion".to_string(),
            RouteDecision::AgenticToolLoop | RouteDecision::AgenticToolForced { .. } => "AgenticToolLoop".to_string(),
        };

        // Update status with route decision
        if let Err(e) = self.mark_processing(&request_id, Some(&route_label)).await {
            error!("Failed to update route decision for {}: {}", request_id, e);
        }

        // Acquire inference permit
        let permit = match self.state.concurrency_limiter.acquire_permit().await {
            Ok(p) => p,
            Err(e) => {
                error!("Request {} failed to acquire permit: {}", request_id, e);
                let _ = self.mark_completed(
                    &request_id,
                    AsyncRequestStatus::Failed,
                    None,
                    Some(&route_label),
                    0,
                    Some(&format!("Failed to acquire inference permit: {}", e)),
                ).await;
                return;
            }
        };

        // Determine if streaming
        let is_streaming = payload.get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Execute based on route decision
        let start = Instant::now();
        let result = match decision {
            RouteDecision::FastPassThrough => {
                info!("Request {}: FastPassThrough", request_id);
                self.execute_passthrough(&payload, is_streaming, &request_id, permit).await
            }
            RouteDecision::RAGAugmented { query } => {
                info!("Request {}: RAGAugmented query={:?}", request_id, query);
                self.execute_rag_augmented(&mut pruned_messages, &payload, &query, is_streaming, &request_id, permit).await
            }
            RouteDecision::RAGIngestion { content } => {
                info!("Request {}: RAGIngestion", request_id);
                self.execute_rag_ingestion(&mut pruned_messages, &payload, &content, is_streaming, &request_id, permit).await
            }
            RouteDecision::AgenticToolLoop => {
                info!("Request {}: AgenticToolLoop", request_id);
                self.execute_agentic(&mut pruned_messages, payload.clone(), &image_base, &file_base, is_streaming, &request_id, permit, None, None, Vec::new()).await
            }
            RouteDecision::AgenticToolForced { tools, .. } => {
                info!("Request {}: AgenticToolForced tools={:?}", request_id, tools);
                self.execute_agentic(&mut pruned_messages, payload.clone(), &image_base, &file_base, is_streaming, &request_id, permit, None, None, tools).await
            }
        };

        let duration = start.elapsed().as_secs_f64() * 1000.0;

        // Handle result
        match result {
            Ok(response) => {
                // Extract result body for storage
                let result_body = self.extract_response_body(response, is_streaming).await;
                let tokens_received = self.estimate_tokens(&result_body);
                
                if let Err(e) = self.mark_completed(
                    &request_id,
                    AsyncRequestStatus::Completed,
                    result_body,
                    Some(&route_label),
                    tokens_received,
                    None,
                ).await {
                    error!("Failed to mark request {} completed: {}", request_id, e);
                }
                
                self.state.monitor.complete_request(
                    request_id.parse().unwrap_or(0),
                    "success",
                    duration,
                    None,
                ).await;
            }
            Err(e) => {
                error!("Request {} failed: {}", request_id, e);
                if let Err(e) = self.mark_completed(
                    &request_id,
                    AsyncRequestStatus::Failed,
                    None,
                    Some(&route_label),
                    0,
                    Some(&e.to_string()),
                ).await {
                    error!("Failed to mark request {} failed: {}", request_id, e);
                }
                
                self.state.monitor.complete_request(
                    request_id.parse().unwrap_or(0),
                    "error",
                    duration,
                    Some(e.to_string()),
                ).await;
            }
        }
    }

    /// Execute pass-through request
    async fn execute_passthrough(
        &self,
        payload: &Value,
        is_streaming: bool,
        request_id: &str,
        permit: InferencePermitGuard,
    ) -> anyhow::Result<Response> {
        forward_to_upstream(&self.state, payload.clone(), is_streaming, request_id.parse().unwrap_or(0), permit).await
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Execute RAG-augmented request
    async fn execute_rag_augmented(
        &self,
        messages: &mut Vec<ChatMessage>,
        payload: &Value,
        query: &str,
        is_streaming: bool,
        request_id: &str,
        permit: InferencePermitGuard,
    ) -> anyhow::Result<Response> {
        let search_hits = self.state.rag_engine.query_rag(query, None, 5)
            .unwrap_or_default();

        let rag_hits_count = search_hits.len();
        if !search_hits.is_empty() {
            let context_block = self.state.rag_engine.format_rag_context(&search_hits);
            crate::server::routes::inject_context_into_messages(messages, &context_block);
            let mut payload = payload.clone();
            payload["messages"] = serde_json::to_value(messages)?;
        }

        self.state.monitor.update_rag_hits(request_id.parse().unwrap_or(0), rag_hits_count).await;

        forward_to_upstream(&self.state, payload.clone(), is_streaming, request_id.parse().unwrap_or(0), permit).await
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Execute RAG ingestion request
    async fn execute_rag_ingestion(
        &self,
        messages: &mut Vec<ChatMessage>,
        payload: &Value,
        content: &str,
        is_streaming: bool,
        request_id: &str,
        permit: InferencePermitGuard,
    ) -> anyhow::Result<Response> {
        let content = content.trim();
        let trivial = content.is_empty() || content.split_whitespace().count() < 3;
        if trivial {
            return forward_to_upstream(&self.state, payload.clone(), is_streaming, request_id.parse().unwrap_or(0), permit).await
                .map_err(|e| anyhow::anyhow!(e));
        }

        let source_uri = format!(
            "chat-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        let chunks = self.state
            .rag_engine
            .ingest_document("default", &source_uri, content)?;

        let note = format!(
            "### System Note: The content from the user's latest message was successfully \
             stored into the RAG knowledge base ({} chunks, collection 'default', source '{}'). \
             If the user asked you to save or remember this information, confirm that it has \
             been saved and tell them how to retrieve it later.",
            chunks, source_uri
        );
        crate::server::routes::inject_context_into_messages(messages, &note);
        let mut payload = payload.clone();
        payload["messages"] = serde_json::to_value(messages)?;

        forward_to_upstream(&self.state, payload, is_streaming, request_id.parse().unwrap_or(0), permit).await
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Execute agentic tool loop request
    #[allow(clippy::too_many_arguments)]
    async fn execute_agentic(
        &self,
        messages: &mut Vec<ChatMessage>,
        mut payload: Value,
        image_base: &str,
        file_base: &str,
        is_streaming: bool,
        request_id: &str,
        permit: InferencePermitGuard,
        image_ctx: Option<ImageGenContext>,
        file_ctx: Option<FileGenContext>,
        forced_tools: Vec<String>,
    ) -> anyhow::Result<Response> {
        // Inject anti-hallucination system prompt
        inject_system_instructions(messages, ANTI_HALLUCINATION_SYSTEM_PROMPT);
        payload["messages"] = serde_json::to_value(&*messages)?;

        // Prepare image/file contexts
        let image_ctx = image_ctx.or_else(|| prepare_image_request(&self.state, messages, image_base));
        let file_ctx = file_ctx.or_else(|| prepare_file_request(&self.state, messages, file_base));

        // Ensure tools are in payload
        if !forced_tools.is_empty() {
            let defs: Vec<Value> = forced_tools
                .iter()
                .filter_map(|t| self.state.tool_registry.definition_for(t))
                .collect();
            payload["tools"] = json!(defs);
        } else if payload.get("tools").is_none() {
            payload["tools"] = self.state.tool_registry.get_internal_tools_definitions();
        }

        execute_agentic_loop(
            &self.state,
            payload.clone(),
            messages.to_vec(),
            is_streaming,
            request_id.parse().unwrap_or(0),
            permit,
            image_ctx,
            file_ctx,
            forced_tools,
        ).await.map_err(|e| anyhow::anyhow!(e))
    }

    /// Extract response body for storage
    async fn extract_response_body(&self, response: Response, is_streaming: bool) -> Option<String> {
        if is_streaming {
            // For streaming, we collect the SSE events and reconstruct a non-streaming response
            let body = response.into_body();
            let mut chunks = Vec::new();
            let mut stream = body.into_data_stream();
            while let Some(chunk) = stream.try_next().await.ok().flatten() {
                chunks.push(chunk);
            }
            let full = String::from_utf8_lossy(&chunks.concat()).to_string();
            // Convert streaming response to final result format
            Some(self.streaming_to_result(&full))
        } else {
            // Non-streaming: extract JSON body
            let body = response.into_body();
            let mut data = Vec::new();
            let mut stream = body.into_data_stream();
            while let Some(chunk) = stream.try_next().await.ok().flatten() {
                data.extend_from_slice(&chunk);
            }
            String::from_utf8(data).ok()
        }
    }

    /// Convert streaming SSE response to a final result JSON
    fn streaming_to_result(&self, sse_data: &str) -> String {
        let mut final_content = String::new();
        let mut finish_reason = None;
        let mut last_usage = None;

        for line in sse_data.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    break;
                }
                if let Ok(v) = serde_json::from_str::<Value>(data) {
                    if let Some(content) = v
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("delta"))
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        final_content.push_str(content);
                    }
                    if let Some(reason) = v
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("finish_reason"))
                        .and_then(|r| r.as_str())
                    {
                        finish_reason = Some(reason.to_string());
                    }
                    if let Some(usage) = v.get("usage") {
                        last_usage = Some(usage.clone());
                    }
                }
            }
        }

        let mut result = json!({
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_content
                },
                "finish_reason": finish_reason.unwrap_or_else(|| "stop".to_string())
            }]
        });

        if let Some(usage) = last_usage {
            result["usage"] = usage;
        }

        result.to_string()
    }

    /// Rough token estimate
    fn estimate_tokens(&self, text: &Option<String>) -> u64 {
        text.as_ref()
            .map(|s| s.len() / 4) // Rough: 4 chars per token
            .unwrap_or(0) as u64
    }

    /// Cleanup expired requests
    async fn cleanup_expired(&self) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result = sqlx::query(
            "DELETE FROM async_requests WHERE expires_at <= ?"
        )
        .bind(now)
        .execute(&self.pool)
        .await?;

        if result.rows_affected() > 0 {
            info!("Cleaned up {} expired async requests", result.rows_affected());
        }

        Ok(())
    }

    fn row_to_request(row: sqlx::sqlite::SqliteRow) -> AsyncRequest {
        AsyncRequest {
            id: row.get("id"),
            payload: row.get("payload"),
            status: match row.get::<String, _>("status").as_str() {
                "queued" => AsyncRequestStatus::Queued,
                "processing" => AsyncRequestStatus::Processing,
                "completed" => AsyncRequestStatus::Completed,
                "failed" => AsyncRequestStatus::Failed,
                "cancelled" => AsyncRequestStatus::Cancelled,
                _ => AsyncRequestStatus::Failed,
            },
            result: row.get("result"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            expires_at: row.get("expires_at"),
            route_decision: row.get("route_decision"),
            tokens_received: row.get("tokens_received"),
            error: row.get("error"),
        }
    }
}

/// Database operations for async requests
pub struct AsyncRequestRepo {
    pool: SqlitePool,
}

impl AsyncRequestRepo {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Create a new async request (idempotent - returns existing if same ID)
    pub async fn create(&self, request: AsyncRequest) -> anyhow::Result<AsyncRequest> {
        // Try to insert
        let result = sqlx::query(
            r#"
            INSERT OR IGNORE INTO async_requests
            (id, payload, status, result, created_at, updated_at, expires_at, route_decision, tokens_received, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&request.id)
        .bind(&request.payload)
        .bind(request.status.as_str())
        .bind(&request.result)
        .bind(request.created_at as i64)
        .bind(request.updated_at as i64)
        .bind(request.expires_at as i64)
        .bind(&request.route_decision)
        .bind(request.tokens_received as i64)
        .bind(&request.error)
        .execute(&self.pool)
        .await?;

        // If row was inserted, return the new request; otherwise fetch existing
        if result.rows_affected() > 0 {
            Ok(request)
        } else {
            self.get(&request.id).await?.ok_or_else(|| anyhow::anyhow!("Request not found after insert"))
        }
    }

    /// Get a request by ID
    pub async fn get(&self, id: &str) -> anyhow::Result<Option<AsyncRequest>> {
        let row = sqlx::query(
            r#"
            SELECT id, payload, status, result, created_at, updated_at, expires_at,
                   route_decision, tokens_received, error
            FROM async_requests
            WHERE id = ?
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(AsyncQueueWorker::row_to_request))
    }

    /// Update request status
    pub async fn update_status(&self, id: &str, status: AsyncRequestStatus) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        sqlx::query(
            "UPDATE async_requests SET status = ?, updated_at = ? WHERE id = ?"
        )
        .bind(status.as_str())
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update request with result
    pub async fn complete(
        &self,
        id: &str,
        status: AsyncRequestStatus,
        result: Option<String>,
        route_decision: Option<String>,
        tokens_received: u64,
        error: Option<String>,
    ) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        sqlx::query(
            r#"
            UPDATE async_requests
            SET status = ?, result = ?, updated_at = ?, route_decision = ?,
                tokens_received = ?, error = ?
            WHERE id = ?
            "#,
        )
        .bind(status.as_str())
        .bind(result)
        .bind(now)
        .bind(route_decision)
        .bind(tokens_received as i64)
        .bind(error)
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Cancel a request
    pub async fn cancel(&self, id: &str) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        sqlx::query(
            "UPDATE async_requests SET status = 'cancelled', updated_at = ? WHERE id = ?"
        )
        .bind(now)
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get request status for polling
    pub async fn get_status(&self, id: &str) -> anyhow::Result<Option<AsyncRequest>> {
        self.get(id).await
    }
}