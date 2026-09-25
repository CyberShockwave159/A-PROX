use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Status of an async request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsyncRequestStatus {
    Queued,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

impl AsyncRequestStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            AsyncRequestStatus::Queued => "queued",
            AsyncRequestStatus::Processing => "processing",
            AsyncRequestStatus::Completed => "completed",
            AsyncRequestStatus::Failed => "failed",
            AsyncRequestStatus::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for AsyncRequestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Async request stored in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRequest {
    pub id: String,
    pub payload: String,
    pub status: AsyncRequestStatus,
    pub result: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub expires_at: u64,
    pub route_decision: Option<String>,
    pub tokens_received: u64,
    pub error: Option<String>,
}

impl AsyncRequest {
    /// Create a new async request with default timestamps and TTL
    pub fn new(
        id: String,
        payload: String,
        ttl_hours: u64,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            payload,
            status: AsyncRequestStatus::Queued,
            result: None,
            created_at: now,
            updated_at: now,
            expires_at: now + ttl_hours * 3600,
            route_decision: None,
            tokens_received: 0,
            error: None,
        }
    }

    /// Check if the request has expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at <= now
    }

    /// Check if the request is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            AsyncRequestStatus::Completed | AsyncRequestStatus::Failed | AsyncRequestStatus::Cancelled
        )
    }
}

/// Request to submit an async completion
#[derive(Debug, Deserialize)]
pub struct SubmitAsyncRequest {
    #[serde(flatten)]
    pub payload: serde_json::Value,
    /// Optional client-provided request ID. If not provided, server generates one.
    pub request_id: Option<String>,
}

/// Response for async submission
#[derive(Debug, Serialize)]
pub struct SubmitAsyncResponse {
    pub request_id: String,
    pub status: String,
}

/// Status response for polling
#[derive(Debug, Serialize)]
pub struct AsyncStatusResponse {
    pub request_id: String,
    pub status: String,
    pub route_decision: Option<String>,
    pub tokens_received: u64,
    pub error: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub expires_at: u64,
}

/// Result response for completed requests
#[derive(Debug, Serialize)]
pub struct AsyncResultResponse {
    pub request_id: String,
    pub status: String,
    pub result: Option<serde_json::Value>,
    pub route_decision: Option<String>,
    pub tokens_received: u64,
    pub error: Option<String>,
    pub created_at: u64,
    pub completed_at: u64,
}