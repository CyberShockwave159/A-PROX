use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Upstream error: {0}")]
    Upstream(String),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Context error: {0}")]
    Context(String),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Resource exhausted (RAM/Bus): {0}")]
    ResourceExhausted(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal server error: {0}")]
    Internal(#[from] anyhow::Error),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
}

#[derive(Serialize)]
struct OpenAIErrorWrapper {
    error: OpenAIErrorDetail,
}

#[derive(Serialize)]
struct OpenAIErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: Option<String>,
    code: Option<u16>,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, err_type) = match &self {
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            AppError::RateLimit(_) => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
            AppError::ResourceExhausted(_) => (StatusCode::SERVICE_UNAVAILABLE, "resource_exhausted"),
            AppError::Upstream(_) => (StatusCode::BAD_GATEWAY, "upstream_error"),
            AppError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "internal_server_error"),
        };

        let body = Json(OpenAIErrorWrapper {
            error: OpenAIErrorDetail {
                message: self.to_string(),
                error_type: err_type.to_string(),
                param: None,
                code: Some(status.as_u16()),
            },
        });

        (status, body).into_response()
    }
}
