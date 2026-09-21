pub mod readability;
pub mod scraper;

use std::sync::Arc;
use reqwest::Client;
use serde::Deserialize;
pub use scraper::WebScraper;

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Deserialize)]
struct SearXNGResponse {
    #[serde(default)]
    results: Vec<SearXNGResult>,
}

#[derive(Deserialize)]
struct SearXNGResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

pub struct SearchService {
    searxng_url: String,
    client: Client,
    scraper: Arc<WebScraper>,
    max_results: usize,
}

impl SearchService {
    pub fn new(
        searxng_url: String,
        timeout_seconds: u64,
        max_results: usize,
        cache_ttl_seconds: u64,
    ) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .unwrap_or_else(|_| Client::new());

        let scraper = Arc::new(WebScraper::new(timeout_seconds, cache_ttl_seconds));

        Self {
            searxng_url,
            client,
            scraper,
            max_results,
        }
    }

    /// Searches local SearXNG instance or falls back gracefully
    pub async fn search(&self, query: &str, count: Option<usize>) -> anyhow::Result<Vec<SearchHit>> {
        let count = count.unwrap_or(self.max_results);
        let endpoint = format!("{}/search", self.searxng_url.trim_end_matches('/'));

        tracing::info!("Executing search query: {:?} at {}", query, endpoint);

        let resp = self.client
            .get(&endpoint)
            .query(&[("q", query), ("format", "json")])
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let parsed: SearXNGResponse = r.json().await?;
                let hits = parsed.results.into_iter().take(count).map(|item| SearchHit {
                    title: item.title,
                    url: item.url,
                    snippet: item.content,
                }).collect();
                Ok(hits)
            }
            Ok(r) => {
                tracing::warn!("SearXNG returned status {}. Using fallback mock results.", r.status());
                Ok(self.fallback_search(query))
            }
            Err(e) => {
                tracing::warn!("Failed to reach SearXNG at {}: {}. Using fallback search.", endpoint, e);
                Ok(self.fallback_search(query))
            }
        }
    }

    pub fn scraper(&self) -> &Arc<WebScraper> {
        &self.scraper
    }

    fn fallback_search(&self, query: &str) -> Vec<SearchHit> {
        vec![
            SearchHit {
                title: format!("Search query: {}", query),
                url: "https://local-proxy.search/results".to_string(),
                snippet: format!("Local meta-search placeholder for query: '{}'. SearXNG instance not detected on configured endpoint.", query),
            }
        ]
    }
}
