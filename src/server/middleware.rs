use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;

pub const API_KEY_HEADER: &str = "x-api-key";

#[derive(Debug, Clone)]
pub struct ApiKey(pub String);

impl ApiKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<S> FromRequestParts<S> for ApiKey
where
    S: Send + Sync,
{
    type Rejection = (axum::http::StatusCode, &'static str);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let headers = &parts.headers;

        // Check X-API-Key header first
        let provided_key = headers
            .get(API_KEY_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Fall back to Authorization: Bearer
        let provided_key = provided_key.or_else(|| {
            headers
                .get(AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(|s| s.to_string())
        });

        match provided_key {
            Some(key) => Ok(ApiKey(key)),
            None => {
                tracing::info!("No API key provided");
                Err((axum::http::StatusCode::UNAUTHORIZED, "API key required"))
            }
        }
    }
}
