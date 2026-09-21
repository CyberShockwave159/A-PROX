pub mod middleware;
pub mod models;
pub mod routes;
pub mod sse;

pub use middleware::ApiKey;
pub use models::*;

use std::net::SocketAddr;
use std::sync::Arc;
use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Maximum accepted request body size (100 MiB).
/// Large enough for multiple base64-encoded photo uploads through
/// `/v1/chat/completions` plus JSON overhead, with generous headroom.
pub const MAX_REQUEST_BODY_BYTES: usize = 100 * 1024 * 1024;

use crate::state::AppState;
use routes::{chat_completions, health_check, ingestion_reindex, ingestion_status, list_models, monitor_api, monitor_dashboard, monitor_sse, monitor_statistics};

pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(health_check))
        .route("/monitor", get(monitor_dashboard))
        .route("/monitor/api", get(monitor_api))
        .route("/monitor/stats", get(monitor_statistics))
        .route("/monitor/stream", get(monitor_sse))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/ingestion/reindex", post(ingestion_reindex))
        .route("/ingestion/status", get(ingestion_status))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn run_server(state: Arc<AppState>) -> anyhow::Result<()> {
    let host = state.config.server.host.clone();
    let port = state.config.server.port;
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("A-PROX server listening on http://{}", addr);
    tracing::info!("Monitoring webpage launched at http://{}{}", addr, "/monitor");

    axum::serve(listener, app).await?;
    Ok(())
}
