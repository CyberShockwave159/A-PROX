use crate::async_queue::models::{AsyncRequest, AsyncRequestStatus, SubmitAsyncRequest, SubmitAsyncResponse, AsyncStatusResponse, AsyncResultResponse};
use crate::async_queue::worker::AsyncRequestRepo;
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures_util::Stream;
use reqwest;
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

/// Submit a new async chat completion request
pub async fn submit_async_completion(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Json(request): Json<SubmitAsyncRequest>,
) -> Result<Response, crate::error::AppError> {
    if !state.config.r#async.enabled {
        return Err(crate::error::AppError::BadRequest(
            "Async completions are disabled".to_string()
        ));
    }

    // Validate payload has required fields
    if request.payload.get("messages").is_none() {
        return Err(crate::error::AppError::BadRequest(
            "Missing 'messages' array in payload".to_string()
        ));
    }

    // Use client-provided ID or generate new one
    let request_id = request.request_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    // Check if request already exists (idempotent)
    let repo = AsyncRequestRepo::new(state.db_pool.clone());
    let existing = repo.get(&request_id).await?;
    
    if let Some(existing_req) = existing {
        // Return existing request status
        let response = SubmitAsyncResponse {
            request_id: existing_req.id,
            status: existing_req.status.as_str().to_string(),
        };
        return Ok((StatusCode::OK, Json(response)).into_response());
    }

    // Create new async request
    let async_request = AsyncRequest::new(
        request_id.clone(),
        request.payload.to_string(),
        state.config.r#async.cache_ttl_hours,
    );

    repo.create(async_request).await?;

    let response = SubmitAsyncResponse {
        request_id,
        status: AsyncRequestStatus::Queued.as_str().to_string(),
    };

    Ok((StatusCode::ACCEPTED, Json(response)).into_response())
}

/// Get the status of an async request
pub async fn get_async_status(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Path(request_id): Path<String>,
) -> Result<Response, crate::error::AppError> {
    if !state.config.r#async.enabled {
        return Err(crate::error::AppError::BadRequest(
            "Async completions are disabled".to_string()
        ));
    }

    let repo = AsyncRequestRepo::new(state.db_pool.clone());
    let request = repo.get_status(&request_id).await?;

    match request {
        Some(req) => {
            let response = AsyncStatusResponse {
                request_id: req.id,
                status: req.status.as_str().to_string(),
                route_decision: req.route_decision,
                tokens_received: req.tokens_received,
                error: req.error,
                created_at: req.created_at,
                updated_at: req.updated_at,
                expires_at: req.expires_at,
            };
            Ok((StatusCode::OK, Json(response)).into_response())
        }
        None => Err(crate::error::AppError::NotFound(format!(
            "Async request '{}' not found", request_id
        ))),
    }
}

/// Get the final result of a completed async request
pub async fn get_async_result(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Path(request_id): Path<String>,
) -> Result<Response, crate::error::AppError> {
    if !state.config.r#async.enabled {
        return Err(crate::error::AppError::BadRequest(
            "Async completions are disabled".to_string()
        ));
    }

    let repo = AsyncRequestRepo::new(state.db_pool.clone());
    let request = repo.get_status(&request_id).await?;

    match request {
        Some(req) => {
            if !req.is_terminal() {
                return Err(crate::error::AppError::BadRequest(format!(
                    "Request '{}' is not yet complete (status: {})", request_id, req.status.as_str()
                )));
            }

            let result = req.result.as_ref().and_then(|r| serde_json::from_str(r).ok());

            let response = AsyncResultResponse {
                request_id: req.id,
                status: req.status.as_str().to_string(),
                result,
                route_decision: req.route_decision,
                tokens_received: req.tokens_received,
                error: req.error,
                created_at: req.created_at,
                completed_at: req.updated_at,
            };
            Ok((StatusCode::OK, Json(response)).into_response())
        }
        None => Err(crate::error::AppError::NotFound(format!(
            "Async request '{}' not found", request_id
        ))),
    }
}

/// Stream the response of an async request (for reconnection/resume)
pub async fn stream_async_completion(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Path(request_id): Path<String>,
) -> Result<Response, crate::error::AppError> {
    if !state.config.r#async.enabled {
        return Err(crate::error::AppError::BadRequest(
            "Async completions are disabled".to_string()
        ));
    }

    let repo = AsyncRequestRepo::new(state.db_pool.clone());
    let request = repo.get_status(&request_id).await?;

    let req = match request {
        Some(r) => r,
        None => return Err(crate::error::AppError::NotFound(format!(
            "Async request '{}' not found", request_id
        ))),
    };

    match req.status {
        AsyncRequestStatus::Completed => {
            // Return cached result as SSE stream
            let result = req.result.ok_or_else(|| {
                crate::error::AppError::Internal(anyhow::anyhow!("Completed request has no result"))
            })?;

            // Convert stored result to SSE stream
            let sse_stream = create_result_sse_stream(&result);
            Ok(crate::server::sse::create_sse_response(sse_stream).into_response())
        }
        AsyncRequestStatus::Processing => {
            // Request is still being processed - we need to subscribe to updates
            // For now, return a stream that will emit the result when ready
            // This is a simplified implementation - a full implementation would use
            // a broadcast channel per request
            let sse_stream = create_waiting_sse_stream(state.clone(), request_id);
            Ok(crate::server::sse::create_sse_response(sse_stream).into_response())
        }
        AsyncRequestStatus::Failed => {
            let error = req.error.unwrap_or_else(|| "Request failed".to_string());
            let error_json = json!({
                "error": { "message": error, "type": "async_request_failed" }
            });
            let data = Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", error_json));
            let sse_stream = futures_util::stream::once(async move { Ok(data) });
            Ok(crate::server::sse::create_sse_response(sse_stream).into_response())
        }
        AsyncRequestStatus::Cancelled => {
            let error_json = json!({
                "error": { "message": "Request was cancelled", "type": "async_request_cancelled" }
            });
            let data = Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", error_json));
            let sse_stream = futures_util::stream::once(async move { Ok(data) });
            Ok(crate::server::sse::create_sse_response(sse_stream).into_response())
        }
        AsyncRequestStatus::Queued => {
            let sse_stream = create_waiting_sse_stream(state.clone(), request_id);
            Ok(crate::server::sse::create_sse_response(sse_stream).into_response())
        }
    }
}

/// Cancel an async request
pub async fn cancel_async_request(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Path(request_id): Path<String>,
) -> Result<Response, crate::error::AppError> {
    if !state.config.r#async.enabled {
        return Err(crate::error::AppError::BadRequest(
            "Async completions are disabled".to_string()
        ));
    }

    let repo = AsyncRequestRepo::new(state.db_pool.clone());
    let request = repo.get_status(&request_id).await?;

    match request {
        Some(req) => {
            if req.is_terminal() {
                return Err(crate::error::AppError::BadRequest(format!(
                    "Cannot cancel request '{}' in terminal state: {}", request_id, req.status.as_str()
                )));
            }

            repo.cancel(&request_id).await?;

            let response = json!({
                "request_id": request_id,
                "status": "cancelled"
            });
            Ok((StatusCode::OK, Json(response)).into_response())
        }
        None => Err(crate::error::AppError::NotFound(format!(
            "Async request '{}' not found", request_id
        ))),
    }
}

/// Create an SSE stream that emits the stored result as bytes
fn create_result_sse_stream(result_json: &str) -> impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static {
    let result: Value = serde_json::from_str(result_json).unwrap_or(json!({}));
    
    // Convert the final result to a streaming format that mimics the original SSE
    let content = result
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    
    let finish_reason = result
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str())
        .unwrap_or("stop");
    
    let usage = result.get("usage").cloned();
    
    // Create a stream that emits the content in chunks, then the final event
    let chunks: Vec<String> = content
        .chars()
        .collect::<Vec<_>>()
        .chunks(50)
        .map(|c| c.iter().collect::<String>())
        .collect();
    
    let stream_items: Vec<reqwest::Result<Bytes>> = chunks
        .into_iter()
        .map(|chunk| {
            let data = json!({
                "choices": [{
                    "index": 0,
                    "delta": { "content": chunk },
                    "finish_reason": null
                }]
            });
            Ok(Bytes::from(format!("data: {}\n\n", data)))
        })
        .collect::<Vec<_>>();
    
    // Add the final [DONE] event
    let mut final_items = stream_items;
    let final_data = json!({
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }],
        "usage": usage
    });
    final_items.push(Ok(Bytes::from(format!("data: {}\n\n", final_data))));
    final_items.push(Ok(Bytes::from("data: [DONE]\n\n")));
    
    futures_util::stream::iter(final_items)
}

/// Create an SSE stream that waits for the request to complete
fn create_waiting_sse_stream(
    state: Arc<AppState>,
    request_id: String,
) -> impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static {
    use futures_util::stream::self;
    use tokio::time::{interval, Duration};
    
    let repo = AsyncRequestRepo::new(state.db_pool.clone());
    let stream_interval = interval(Duration::from_millis(500));
    
    // Create a stream that polls the status periodically
    let stream = stream::unfold(
        (repo, request_id, stream_interval, false),
        move |(repo, request_id, mut interval, first_poll)| async move {
            // On first poll, check immediately; otherwise wait for tick
            if !first_poll {
                interval.tick().await;
            }
            
            match repo.get_status(&request_id).await {
                Ok(Some(req)) => {
                    if req.is_terminal() {
                        // Return the final result
                        let result = req.result.unwrap_or_default();
                        let data = Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", result));
                        return Some((Ok(data), (repo, request_id, interval, true)));
                    }
                    // Still processing - send a keepalive
                    let keepalive = Bytes::from(": keepalive\n\n");
                    Some((Ok(keepalive), (repo, request_id, interval, true)))
                }
                Ok(None) => {
                    let error_json = json!({
                        "error": { "message": "Request not found", "type": "async_request_not_found" }
                    });
                    let data = Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", error_json));
                    Some((Ok(data), (repo, request_id, interval, true)))
                }
                Err(e) => {
                    let error_json = json!({
                        "error": { "message": e.to_string(), "type": "async_request_error" }
                    });
                    let data = Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", error_json));
                    Some((Ok(data), (repo, request_id, interval, true)))
                }
            }
        }
    );
    
    stream
}