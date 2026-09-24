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
    #[serde(default)]
    pub llama_server: LlamaServerConfig,
    #[serde(default)]
    pub comfy_ui: ComfyUiConfig,
    #[serde(default)]
    pub image_generation: ImageGenerationConfig,
    #[serde(default)]
    pub file_generation: FileGenerationConfig,
    #[serde(default)]
    pub tool_commands: ToolCommandsConfig,
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

/// Managed llama.cpp server that A-PROX owns (spawns/stops around image jobs).
/// This mirrors the exact CLI flags used before, minus `--load-mode mlock`.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaServerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub executable: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub mmproj_path: String,
    #[serde(default = "default_llama_host")]
    pub host: String,
    #[serde(default = "default_llama_port")]
    pub port: u16,
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_zero_i32")]
    pub ctx_size: i32,
    #[serde(default = "default_true")]
    pub no_mmproj_offload: bool,
    #[serde(default = "default_on")]
    pub flash_attn: String,
    #[serde(default = "default_n_cpu_moe")]
    pub n_cpu_moe: u32,
    #[serde(default = "default_q4")]
    pub cache_type_k: String,
    #[serde(default = "default_q4")]
    pub cache_type_v: String,
    #[serde(default = "default_true")]
    pub reasoning_preserve: bool,
    #[serde(default = "default_true")]
    pub kv_unified: bool,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_ubatch_size")]
    pub ubatch_size: usize,
    #[serde(default = "default_image_min_tokens")]
    pub image_min_tokens: u32,
    #[serde(default = "default_image_max_tokens")]
    pub image_max_tokens: u32,
    #[serde(default)]
    pub cors_origins: String,
    /// `none` omits the flag entirely; anything else is passed as `--load-mode <v>`.
    #[serde(default = "default_load_mode")]
    pub load_mode: String,
    #[serde(default = "default_health_timeout_s")]
    pub health_timeout_s: u64,
    #[serde(default = "default_stop_grace_s")]
    pub stop_grace_s: u64,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            executable: String::new(),
            model_path: String::new(),
            mmproj_path: String::new(),
            host: default_llama_host(),
            port: default_llama_port(),
            api_key: String::new(),
            ctx_size: 0,
            no_mmproj_offload: true,
            flash_attn: "on".to_string(),
            n_cpu_moe: 34,
            cache_type_k: "q4_0".to_string(),
            cache_type_v: "q4_0".to_string(),
            reasoning_preserve: true,
            kv_unified: true,
            threads: 6,
            batch_size: 4096,
            ubatch_size: 1024,
            image_min_tokens: 1024,
            image_max_tokens: 2048,
            cors_origins: String::new(),
            load_mode: "none".to_string(),
            health_timeout_s: 600,
            stop_grace_s: 30,
        }
    }
}

/// Managed ComfyUI server used for image generation.
#[derive(Debug, Clone, Deserialize)]
pub struct ComfyUiConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_comfy_url")]
    pub url: String,
    #[serde(default)]
    pub workdir: String,
    #[serde(default)]
    pub python: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_health_timeout_s")]
    pub health_timeout_s: u64,
}

impl Default for ComfyUiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: default_comfy_url(),
            workdir: String::new(),
            python: String::new(),
            args: vec!["main.py".to_string(), "--enable-manager".to_string()],
            health_timeout_s: 600,
        }
    }
}

/// Image generation pipeline settings (ComfyUI Qwen-Image workflows).
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_mp")]
    pub mp: f64,
    #[serde(default = "default_multiple")]
    pub multiple: u32,
    #[serde(default = "default_max_side")]
    pub max_side: u32,
    #[serde(default)]
    pub unet: String,
    #[serde(default)]
    pub clip: String,
    #[serde(default)]
    pub vae: String,
    #[serde(default)]
    pub t2i_workflow: String,
    #[serde(default)]
    pub i2i_workflow: String,
    #[serde(default)]
    pub t2i_prompt_file: String,
    #[serde(default)]
    pub i2i_prompt_file: String,
    #[serde(default = "default_negative_prompt")]
    pub default_negative_prompt: String,
    #[serde(default = "default_serve_dir")]
    pub serve_dir: String,
    #[serde(default)]
    pub public_base_url: String,
    #[serde(default = "default_generation_timeout_s")]
    pub generation_timeout_s: u64,
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mp: 2.0,
            multiple: 16,
            max_side: 4096,
            unet: String::new(),
            clip: String::new(),
            vae: String::new(),
            t2i_workflow: "workflows/t2i.json".to_string(),
            i2i_workflow: "workflows/i2i.json".to_string(),
            t2i_prompt_file: "prompts/t-iprompt.txt".to_string(),
            i2i_prompt_file: "prompts/i-iprompt.txt".to_string(),
            default_negative_prompt: default_negative_prompt(),
            serve_dir: "data/generated_images".to_string(),
            public_base_url: String::new(),
            generation_timeout_s: 180,
            poll_interval_ms: 2000,
        }
    }
}

fn default_llama_host() -> String { "127.0.0.1".to_string() }
fn default_llama_port() -> u16 { 8080 }
fn default_zero_i32() -> i32 { 0 }
fn default_true() -> bool { true }
fn default_on() -> String { "on".to_string() }
fn default_n_cpu_moe() -> u32 { 34 }
fn default_q4() -> String { "q4_0".to_string() }
fn default_threads() -> usize { 6 }
fn default_batch_size() -> usize { 4096 }
fn default_ubatch_size() -> usize { 1024 }
fn default_image_min_tokens() -> u32 { 1024 }
fn default_image_max_tokens() -> u32 { 2048 }
fn default_load_mode() -> String { "none".to_string() }
fn default_health_timeout_s() -> u64 { 600 }
fn default_stop_grace_s() -> u64 { 30 }
fn default_comfy_url() -> String { "http://127.0.0.1:8188".to_string() }
fn default_mp() -> f64 { 2.0 }
fn default_multiple() -> u32 { 16 }
fn default_max_side() -> u32 { 4096 }
fn default_generation_timeout_s() -> u64 { 180 }
fn default_poll_interval_ms() -> u64 { 2000 }

fn default_max_content_chars() -> usize { 24_000 }
fn default_file_serve_dir() -> String { "data/generated_files".to_string() }

/// Generation of text files via the `write_file` tool (`/files/{name}`).
#[derive(Debug, Clone, Deserialize)]
pub struct FileGenerationConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_file_serve_dir")]
    pub serve_dir: String,
    #[serde(default)]
    pub public_base_url: String,
    #[serde(default = "default_max_content_chars")]
    pub max_content_chars: usize,
    #[serde(default)]
    pub deny_exts: Vec<String>,
}

impl Default for FileGenerationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            serve_dir: default_file_serve_dir(),
            public_base_url: String::new(),
            max_content_chars: default_max_content_chars(),
            deny_exts: Vec::new(),
        }
    }
}
fn default_serve_dir() -> String { "data/generated_images".to_string() }
fn default_negative_prompt() -> String {
    "bad anatomy, bad composition, bad lighting, distorted face, extra limbs, low quality, out of focus, overexposed, plastic, poor symmetry, signature, watermark, ugly, censored".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCommandsConfig {
    #[serde(default = "default_flag_agentic")]
    pub agentic: String,
    #[serde(default = "default_flag_search")]
    pub web_search: String,
    #[serde(default = "default_flag_fetch")]
    pub web_fetch: String,
    #[serde(default = "default_flag_rag_search")]
    pub rag_search: String,
    #[serde(default = "default_flag_rag_ingest")]
    pub rag_ingest: String,
    #[serde(default = "default_flag_system_time")]
    pub system_time: String,
    #[serde(default = "default_flag_image_generate")]
    pub image_generate: String,
    #[serde(default = "default_flag_write_file")]
    pub write_file: String,
    #[serde(default = "default_flag_bypass")]
    pub bypass: String,
    #[serde(default = "default_flag_direct")]
    pub direct: String,
    #[serde(default = "default_flag_pass", rename = "pass")]
    pub pass_route: String,
    #[serde(default = "default_flag_rag")]
    pub rag: String,
    #[serde(default = "default_flag_knowledge")]
    pub knowledge: String,
    #[serde(default = "default_flag_docs")]
    pub docs: String,
}

impl Default for ToolCommandsConfig {
    fn default() -> Self {
        Self {
            agentic: default_flag_agentic(),
            web_search: default_flag_search(),
            web_fetch: default_flag_fetch(),
            rag_search: default_flag_rag_search(),
            rag_ingest: default_flag_rag_ingest(),
            system_time: default_flag_system_time(),
            image_generate: default_flag_image_generate(),
            write_file: default_flag_write_file(),
            bypass: default_flag_bypass(),
            direct: default_flag_direct(),
            pass_route: default_flag_pass(),
            rag: default_flag_rag(),
            knowledge: default_flag_knowledge(),
            docs: default_flag_docs(),
        }
    }
}

fn default_flag_agentic() -> String { "/tools".to_string() }
fn default_flag_search() -> String { "/search".to_string() }
fn default_flag_fetch() -> String { "/fetch".to_string() }
fn default_flag_rag_search() -> String { "/ragsearch".to_string() }
fn default_flag_rag_ingest() -> String { "/ingest".to_string() }
fn default_flag_system_time() -> String { "/time".to_string() }
fn default_flag_image_generate() -> String { "/image".to_string() }
fn default_flag_write_file() -> String { "/file".to_string() }
fn default_flag_bypass() -> String { "/bypass".to_string() }
fn default_flag_direct() -> String { "/direct".to_string() }
fn default_flag_pass() -> String { "/pass".to_string() }
fn default_flag_rag() -> String { "/rag".to_string() }
fn default_flag_knowledge() -> String { "/knowledge".to_string() }
fn default_flag_docs() -> String { "/docs".to_string() }

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
    #[serde(default)]
    pub image_generation: IntentCategory,
    #[serde(default)]
    pub file_generation: IntentCategory,
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
            image_generation: IntentCategory {
                examples: default_image_generation_examples(),
                threshold: 0.70,
            },
            file_generation: IntentCategory {
                examples: default_file_generation_examples(),
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

/// Default example prompts for the image-generation intent category.
pub fn default_image_generation_examples() -> Vec<IntentExample> {
    vec![
        IntentExample { text: "generate an image of a futuristic city".to_string() },
        IntentExample { text: "create a picture of a cat astronaut".to_string() },
        IntentExample { text: "draw me a logo for my startup".to_string() },
        IntentExample { text: "make an image of a cyberpunk street".to_string() },
        IntentExample { text: "render a scene of a dragon in a forest".to_string() },
        IntentExample { text: "generate a portrait photo of a wizard".to_string() },
        IntentExample { text: "create a photo of a sunset over the ocean".to_string() },
        IntentExample { text: "design a poster for a concert".to_string() },
        IntentExample { text: "generate a wallpaper of space galaxies".to_string() },
        IntentExample { text: "edit the attached photo to add snow".to_string() },
        IntentExample { text: "redesign this room in a modern style".to_string() },
        IntentExample { text: "turn this sketch into a finished painting".to_string() },
        IntentExample { text: "make an illustration of my book cover".to_string() },
    ]
}

/// Default example prompts for the file-generation (write_file) intent category.
pub fn default_file_generation_examples() -> Vec<IntentExample> {
    vec![
        IntentExample { text: "write a python script to a file".to_string() },
        IntentExample { text: "create a markdown file with my meeting notes".to_string() },
        IntentExample { text: "save this as a text file".to_string() },
        IntentExample { text: "generate a csv file of this data".to_string() },
        IntentExample { text: "put my to-do list in a file".to_string() },
        IntentExample { text: "write a json file with the results".to_string() },
        IntentExample { text: "save a python script to notes.py".to_string() },
        IntentExample { text: "dump this into a markdown file".to_string() },
        IntentExample { text: "convert my notes to a text file".to_string() },
        IntentExample { text: "write a rust program and save it to a file".to_string() },
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
            llama_server: LlamaServerConfig::default(),
            comfy_ui: ComfyUiConfig::default(),
            image_generation: ImageGenerationConfig::default(),
            file_generation: FileGenerationConfig::default(),
            tool_commands: ToolCommandsConfig::default(),
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
