use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub upstream: UpstreamConfig,
    pub guardrails: GuardrailsConfig,
    pub context: ContextConfig,
    pub embeddings: EmbeddingsConfig,
    pub db: DbConfig,
    pub search: SearchConfig,
    pub searxng: SearXNGConfig,
    #[serde(default)]
    pub monitor: MonitorConfig,
    #[serde(default)]
    pub ingestion: IngestionConfig,
    #[serde(default)]
    pub intent: IntentConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IngestionConfig {
    #[serde(default)]
    pub directories: Vec<String>,
    #[serde(default)]
    pub file_extensions: Vec<String>,
    #[serde(default)]
    pub excluded_patterns: Vec<String>,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default = "default_max_file_size_kb")]
    pub max_file_size_kb: usize,
    #[serde(default = "default_watch_interval_secs")]
    pub watch_interval_secs: u64,
    #[serde(default = "default_pdf_enabled")]
    pub pdf_enabled: bool,
    #[serde(default = "default_pdf_upstream_model")]
    pub pdf_upstream_model: String,
}

fn default_collection() -> String { "auto-indexed".to_string() }
fn default_max_file_size_kb() -> usize { 512 }
fn default_watch_interval_secs() -> u64 { 30 }
fn default_pdf_enabled() -> bool { true }
fn default_pdf_upstream_model() -> String { "qwen3.6-35b-moe".to_string() }

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    pub base_url: String,
    pub api_key: String,
    pub model_alias: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GuardrailsConfig {
    #[serde(default = "default_max_concurrent_inferences")]
    pub max_concurrent_inferences: usize,
    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,
    #[serde(default = "default_rate_limit")]
    pub rate_limit_per_minute: u32,
    #[serde(default = "default_min_free_ram_gb")]
    pub min_free_ram_gb: f64,
    #[serde(default = "default_enable_agentic_tools")]
    pub enable_agentic_tools: bool,
}

fn default_max_concurrent_inferences() -> usize { 1 }
fn default_queue_depth() -> usize { 16 }
fn default_rate_limit() -> u32 { 120 }
fn default_min_free_ram_gb() -> f64 { 16.0 }
fn default_enable_agentic_tools() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
pub struct ContextConfig {
    pub max_context_tokens: usize,
    pub reserve_completion_tokens: usize,
    pub sliding_window_turns: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsConfig {
    pub model_path: String,
    pub cpu_threads: usize,
    pub dimension: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DbConfig {
    pub path: String,
    pub mmap_size_mb: usize,
    pub cache_size_mb: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchConfig {
    pub searxng_url: String,
    pub timeout_seconds: u64,
    pub max_results: usize,
    pub cache_ttl_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearXNGConfig {
    pub enabled: bool,
    pub port: u16,
    pub install_dir: String,
    pub listen_port: u16,
    pub search_engines: String,
    pub max_results: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MonitorConfig {
    #[serde(default = "default_max_history")]
    pub max_history_entries: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_history_entries: 200,
        }
    }
}

fn default_max_history() -> usize {
    200
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentConfig {
    #[serde(default)]
    pub categories: IntentCategories,
    #[serde(default = "default_intent_threshold")]
    pub default_threshold: f32,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            categories: IntentCategories::default(),
            default_threshold: 0.70,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentCategories {
    #[serde(default)]
    pub agentic_tool: IntentCategory,
    #[serde(default)]
    pub rag_search: IntentCategory,
    #[serde(default)]
    pub rag_ingest: IntentCategory,
}

impl Default for IntentCategories {
    fn default() -> Self {
        Self {
            agentic_tool: IntentCategory {
                examples: default_agentic_examples(),
                threshold: 0.70,
            },
            rag_search: IntentCategory {
                examples: default_rag_search_examples(),
                threshold: 0.70,
            },
            rag_ingest: IntentCategory {
                examples: default_rag_ingest_examples(),
                threshold: 0.70,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentCategory {
    #[serde(default)]
    pub examples: Vec<IntentExample>,
    #[serde(default = "default_intent_threshold")]
    pub threshold: f32,
}

impl Default for IntentCategory {
    fn default() -> Self {
        // NOTE: this generic default intentionally has NO examples. When a config
        // file defines `[intent.categories.*]` tables, serde fills missing `examples`
        // with an empty Vec (via #[serde(default)]), NOT with this Default impl.
        // The classifier falls back to the per-category default examples in that case.
        Self {
            examples: Vec::new(),
            threshold: 0.70,
        }
    }
}

/// Default example prompts for the agentic-tool intent category.
pub fn default_agentic_examples() -> Vec<IntentExample> {
    vec![
        IntentExample { text: "search the web for Rust async patterns".to_string() },
        IntentExample { text: "what's the weather like in Tokyo right now".to_string() },
        IntentExample { text: "fetch the content from this URL".to_string() },
        IntentExample { text: "look up the latest news about AI".to_string() },
        IntentExample { text: "check the current time".to_string() },
        IntentExample { text: "search online for Python tutorials".to_string() },
        IntentExample { text: "get me the latest headlines".to_string() },
        IntentExample { text: "find out what's trending today".to_string() },
        IntentExample { text: "search for the current price of Bitcoin".to_string() },
        IntentExample { text: "check the web for recent developments in quantum computing".to_string() },
    ]
}

/// Default example prompts for the RAG search intent category.
pub fn default_rag_search_examples() -> Vec<IntentExample> {
    vec![
        IntentExample { text: "search my docs for the API endpoint".to_string() },
        IntentExample { text: "what do my notes say about authentication".to_string() },
        IntentExample { text: "find info about the migration in my files".to_string() },
        IntentExample { text: "search knowledge base for deployment steps".to_string() },
        IntentExample { text: "what does the readme say about setup".to_string() },
        IntentExample { text: "look in my knowledge base for the config".to_string() },
        IntentExample { text: "search docs for the error handling section".to_string() },
        IntentExample { text: "what's in my files about Docker".to_string() },
    ]
}

/// Default example prompts for the RAG ingestion intent category.
pub fn default_rag_ingest_examples() -> Vec<IntentExample> {
    vec![
        IntentExample { text: "save this to my notes".to_string() },
        IntentExample { text: "index this document into my knowledge base".to_string() },
        IntentExample { text: "store this in my local docs".to_string() },
        IntentExample { text: "ingest this text into my vector store".to_string() },
        IntentExample { text: "save this article to my knowledge base".to_string() },
        IntentExample { text: "index this note for future searching".to_string() },
        IntentExample { text: "store this in my files".to_string() },
        IntentExample { text: "remember this for later".to_string() },
        IntentExample { text: "add this to my notes".to_string() },
    ]
}

fn default_intent_threshold() -> f32 {
    0.70
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentExample {
    pub text: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8000,
            },
            upstream: UpstreamConfig {
                base_url: "http://127.0.0.1:8080".to_string(),
                api_key: "change-me".to_string(),
                model_alias: "qwen3.6-35b-moe".to_string(),
                timeout_seconds: 180,
            },
            guardrails: GuardrailsConfig {
                max_concurrent_inferences: 1,
                queue_depth: 16,
                rate_limit_per_minute: 120,
                min_free_ram_gb: 16.0,
                enable_agentic_tools: true,
            },
            context: ContextConfig {
                max_context_tokens: 32768,
                reserve_completion_tokens: 4096,
                sliding_window_turns: 10,
            },
            embeddings: EmbeddingsConfig {
                model_path: "models/bge-small-en-v1.5-int8.onnx".to_string(),
                cpu_threads: 4,
                dimension: 384,
            },
            db: DbConfig {
                path: "data/a_prox.db".to_string(),
                mmap_size_mb: 16384,
                cache_size_mb: 1024,
            },
            search: SearchConfig {
                searxng_url: "http://127.0.0.1:8888".to_string(),
                timeout_seconds: 4,
                max_results: 5,
                cache_ttl_seconds: 86400,
            },
            searxng: SearXNGConfig {
                enabled: true,
                port: 8888,
                install_dir: String::from("~/.local/share/a-prox-searxng"),
                listen_port: 8888,
                search_engines: String::from("google,bing,duckduckgo,wikipedia,github"),
                max_results: 5,
            },
            monitor: MonitorConfig {
                max_history_entries: 200,
            },
            ingestion: IngestionConfig {
                directories: Vec::new(),
                file_extensions: vec![
                    "txt".to_string(), "md".to_string(), "json".to_string(),
                    "yaml".to_string(), "yml".to_string(), "toml".to_string(),
                    "html".to_string(), "htm".to_string(), "log".to_string(),
                    "csv".to_string(), "xml".to_string(),
                ],
                excluded_patterns: vec!["*.lock".to_string(), "*.swp".to_string(), "*.tmp".to_string(), ".DS_Store".to_string()],
                collection: "auto-indexed".to_string(),
                max_file_size_kb: 512,
                watch_interval_secs: 30,
                pdf_enabled: true,
                pdf_upstream_model: "qwen3.6-35b-moe".to_string(),
            },
            intent: IntentConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let config: AppConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }
}
