use std::sync::Arc;
use std::time::Duration;
use moka::future::Cache;
use reqwest::Client;
use super::readability::{clean_html, CleanPage};

#[derive(Clone)]
pub struct WebScraper {
    client: Client,
    cache: Cache<String, Arc<CleanPage>>,
    timeout: Duration,
}

impl WebScraper {
    pub fn new(timeout_seconds: u64, cache_ttl_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 A-PROX/0.1")
            .redirect(reqwest::redirect::Policy::limited(3))
            .gzip(true)
            .brotli(true)
            .zstd(true)
            .build()
            .unwrap_or_else(|_| Client::new());

        let cache = Cache::builder()
            .max_capacity(5000)
            .time_to_live(Duration::from_secs(cache_ttl_seconds))
            .build();

        Self {
            client,
            cache,
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    /// Fetches a URL, strips boilerplate, and returns cleaned text with caching
    pub async fn fetch_and_clean(&self, url: &str) -> anyhow::Result<Arc<CleanPage>> {
        if let Some(cached) = self.cache.get(url).await {
            tracing::debug!("Scraper cache hit for URL: {}", url);
            return Ok(cached);
        }

        tracing::info!("Scraping URL: {}", url);
        let resp = self.client
            .get(url)
            .timeout(self.timeout)
            .send()
            .await?;

        let html = resp.text().await?;
        let cleaned = clean_html(&html);
        let page = Arc::new(cleaned);

        self.cache.insert(url.to_string(), Arc::clone(&page)).await;
        Ok(page)
    }
}
