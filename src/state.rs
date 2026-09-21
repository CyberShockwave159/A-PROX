use std::sync::Arc;
use reqwest::Client;
use crate::config::AppConfig;
use crate::context::{ContextManager, FastTokenizer};
use crate::db::VectorStore;
use crate::embeddings::CpuEmbedder;
use crate::guardrails::{ConcurrencyLimiter, SystemWatchdog};
use crate::ingestion::watcher::DirectoryWatcher;
use crate::ingestion::DirectoryIngestor;
use crate::monitor::MonitorState;
use crate::rag::RagEngine;
use crate::search::SearchService;
use crate::searxng::SearXNGManager;
use crate::tools::ToolRegistry;
use crate::router::IntentClassifier;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub http_client: Client,
    pub context_mgr: Arc<ContextManager>,
    pub rag_engine: Arc<RagEngine>,
    pub search_service: Arc<SearchService>,
    pub tool_registry: Arc<ToolRegistry>,
    pub concurrency_limiter: Arc<ConcurrencyLimiter>,
    pub watchdog: Arc<SystemWatchdog>,
    pub searxng_manager: Arc<SearXNGManager>,
    pub monitor: Arc<MonitorState>,
    pub directory_ingestor: Arc<DirectoryIngestor>,
    pub directory_watcher: Arc<DirectoryWatcher>,
    pub intent_classifier: Arc<IntentClassifier>,
}

impl AppState {
    pub async fn new(config: AppConfig) -> anyhow::Result<Self> {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.upstream.timeout_seconds))
            .build()?;

        let tokenizer = Arc::new(FastTokenizer::new::<&str>(None));

        let embedder = Arc::new(CpuEmbedder::new::<&str>(
            &config.embeddings.model_path,
            None,
            config.embeddings.cpu_threads,
            config.embeddings.dimension,
        ));

        let vector_store = Arc::new(VectorStore::new(
            &config.db.path,
            config.embeddings.dimension,
            config.db.mmap_size_mb,
            config.db.cache_size_mb,
        )?);

        let rag_engine = Arc::new(RagEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&embedder),
            Arc::clone(&tokenizer),
        ));

        let search_service = Arc::new(SearchService::new(
            config.search.searxng_url.clone(),
            config.search.timeout_seconds,
            config.search.max_results,
            config.search.cache_ttl_seconds,
        ));

        let tool_registry = Arc::new(ToolRegistry::new(
            Arc::clone(&search_service),
            Arc::clone(&rag_engine),
        ));

        let context_mgr = Arc::new(ContextManager::new(
            FastTokenizer::new::<&str>(None),
            config.context.max_context_tokens,
            config.context.reserve_completion_tokens,
            config.context.sliding_window_turns,
        ));

        let concurrency_limiter = Arc::new(ConcurrencyLimiter::new(
            config.guardrails.max_concurrent_inferences,
            config.guardrails.queue_depth,
        ));

        let watchdog = Arc::new(SystemWatchdog::new(config.guardrails.min_free_ram_gb));

        let searxng_manager = Arc::new(SearXNGManager::new());
        if config.searxng.enabled {
            tracing::info!("SearXNG is enabled, starting...");
            searxng_manager.start(&config.searxng, &http_client).await?;
        }

        let monitor = Arc::new(MonitorState::new(
            config.monitor.max_history_entries,
        ));

        // Create directory ingestor and run initial ingestion
        let directory_ingestor = Arc::new(DirectoryIngestor::new(
            config.ingestion.clone(),
            Arc::clone(&vector_store),
            Arc::clone(&rag_engine),
            config.upstream.base_url.clone(),
            config.upstream.api_key.clone(),
            config.ingestion.pdf_upstream_model.clone(),
        ));

        // Run initial ingestion
        let ingestion_result = directory_ingestor.run_initial_ingestion().await;
        tracing::info!(
            "Directory ingestion complete: {} files indexed, {} skipped, {} errors, {} total chunks",
            ingestion_result.files_indexed,
            ingestion_result.files_skipped,
            ingestion_result.files_errors,
            ingestion_result.chunks_total
        );

        // Set up passive file watcher
        let directory_watcher = Arc::new(DirectoryWatcher::new(
            Arc::clone(&directory_ingestor),
            config.ingestion.watch_interval_secs,
        ));

        // Clone for spawning, keep original for AppState
        let watcher_for_spawn = Arc::clone(&directory_watcher);
        let watcher_clone = Arc::clone(&directory_watcher);
        watcher_for_spawn.spawn(&config.ingestion.directories);
        watcher_clone.watch_directories(&config.ingestion.directories).ok();

        // Create intent classifier
        let intent_classifier = Arc::new(IntentClassifier::new(
            Arc::clone(&embedder),
            config.intent.clone(),
        ));

        Ok(Self {
            config,
            http_client,
            context_mgr,
            rag_engine,
            search_service,
            tool_registry,
            concurrency_limiter,
            watchdog,
            searxng_manager,
            monitor,
            directory_ingestor,
            directory_watcher,
            intent_classifier,
        })
    }
}
