use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::{Stream, StreamExt};
use std::convert::Infallible;

pub fn create_sse_response<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    let mapped_stream = stream.filter_map(|item| async move {
        match item {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes).to_string();
                // Strip SSE 'data: ' prefix if raw or forward as event
                Some(Ok(Event::default().data(text)))
            }
            Err(e) => {
                tracing::warn!("Error in upstream SSE stream: {e}");
                None
            }
        }
    });

    Sse::new(mapped_stream).keep_alive(KeepAlive::default())
}
