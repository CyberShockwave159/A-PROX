# A-PROX: Hardware-Optimized LLM Middleware Proxy

A high-performance proxy server that sits between your LLM frontends and `llama.cpp`. It adds intelligent features like web search, RAG (document retrieval), and tool execution — all running on CPU with zero GPU memory usage.

---

## How It Works (At a Glance)

```
Your Apps (SillyTavern, OpenWebUI, CLAN, curl, etc.)
       │
       ▼  port 8000 (A-PROX)
  ┌───────────────────────────┐
  │   A-PROX                  │  ← RAG/search/tool routing + image generation
  └───┬───────────┬───────────┘
      │ 8080      │ 8188 (managed, image jobs)
      ▼           ▼
 ┌────────────┐  ┌──────────────────┐
 │ llama.cpp  │  │ ComfyUI Qwen-IMG │  ← spawned lazily on first image job
 └────────────┘  └──────────────────┘
```

- **Port 8000** — A-PROX gateway. Clients connect here.
- **Port 8080** — llama.cpp server. Only reachable on `localhost` (behind A-PROX). A-PROX launches it from `[llama_server]` (paused during image jobs and restarted after).
- **Port 8188** — Managed ComfyUI backend used for `image_generate`; spawned on the first image job and left running.
- **API Key** — Not enforced on `/v1/*`, `/ingestion/*`, `/monitor/*`; when a key *is* provided to `/monitor/api` or `/monitor/stream` it is compared against `upstream.api_key`.

---

## Hardware Optimization

A-PROX was built for a machine where the GPU is fully occupied:
- **GPU (RTX 3080 Ti 12GB):** 0 MB used by A-PROX. All proxy logic runs on CPU.
- **CPU (Ryzen 9 5900X):** llama.cpp uses 6 threads; A-PROX uses the remaining 18.
- **RAM (128 GB):** 16 GB for SQLite memory-mapped I/O, 1 GB page cache, rest free.
- **Concurrency limit:** Max 1 active generation to prevent memory-bandwidth bottlenecks.

---

## Key Features

1. **Intelligent Request Routing** — Passes through simple requests directly; routes search/RAG queries through extra processing.
2. **CPU-Only Embeddings** — Uses ONNX Runtime on CPU with a quantized `bge-small-en-v1.5` model. Falls back to heuristic embeddings if the model file is missing.
3. **Hybrid Vector Search** — Combines vector similarity and full-text search using Reciprocal Rank Fusion (RRF).
4. **Context Management** — Automatically prunes long conversations while keeping system prompts and the latest user message.
5. **Web Search & Scraper** — Queries a local SearXNG instance or falls back to mock results; extracts clean text from web pages.
6. **Tool Execution** — The LLM can call tools (`web_search`, `web_fetch`, `rag_search`, `rag_ingest`, `system_time`) in an autonomous loop of up to 5 turns. Image requests automatically arm an `image_generate` tool backed by a managed ComfyUI (Qwen-Image 2.1) backend, with A-PROX pausing llama.cpp while the image job runs and resuming it afterward. File-write requests arm a `write_file` tool that saves text/scripts to disk and serves them via `GET /files/{name}` (append support across turns).
7. **System Guardrails** — Monitors free RAM and limits concurrent requests to protect the system.
8. **Directory-Based Automatic Indexing** — Configures directories to scan for files; automatically extracts text, generates embeddings, and indexes documents into the RAG store. Supports live file watching, PDF processing via upstream multimodal endpoints, and per-directory collections.

---

## Prerequisites

1. **Rust toolchain** — Install via [rustup](https://rustup.rs/): `curl --proto '=https' --tlsv1.2 -sSf https://rustup.rs | sh`
2. **llama.cpp** — A-PROX owns and launches `llama-server` itself from the `[llama_server]` config block (spawned on `127.0.0.1:8080`, no `--load-mode mlock`). You no longer need to launch it manually first — but an externally-running instance at the same URL is also detected and reused.
3. **Optional: SearXNG** — For live web search, A-PROX manages a local SearXNG instance on port 8888. It works without it (returns placeholder results).
4. **Optional: ComfyUI (image generation)** — For `image_generate` to work, a Qwen-Image 2.1 GGUF ComfyUI install is configured in `[comfy_ui]` (spawned lazily on the first image job, port 8188).

---

## Quick Start

### 1. Build

```bash
# Download the ONNX embedding model (~67 MB)
./scripts/download_model.sh

# Build the release binary
cargo build --release
```

If you don't have a GPU or don't want to wait for ONNX, the build works without the model file — A-PROX falls back to heuristic embeddings at runtime.

### 2. Start the Server

The simplest way:

```bash
./run.sh
```

Or run the binary directly:

```bash
./target/release/a-prox
```

A-PROX reads `config/default.toml` automatically. If the file doesn't exist, it uses sensible defaults.

### 3. Access the Health Check

Verify it's running (no API key needed):

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "A-PROX",
  "version": "0.1.0",
  "hardware": {
    "total_ram_gb": "125.7",
    "available_ram_gb": "97.7",
    "global_cpu_usage_pct": "2.0%",
    "gpu_mode": "Zero-VRAM (Pure CPU Middleware)"
  },
  "concurrency": {
    "active_inferences": 0,
    "queued_requests": 0,
    "max_slots": 1
  },
  "upstream": {
    "endpoint": "http://127.0.0.1:8080",
    "model_alias": "qwen3.6-35b-moe"
  }
}
```

---

## Starting with Custom Settings (CLI Flags)

You don't need to edit the config file to change anything. Use CLI flags:

```bash
# Change the listening port
./target/release/a-prox --port 9000

# Point to a different llama.cpp instance
./target/release/a-prox --upstream http://192.168.1.50:8080

# Use a custom config file
./target/release/a-prox --config /path/to/my-config.toml

# Combine multiple flags
./target/release/a-prox \
  --port 9000 \
  --host 0.0.0.0 \
  --upstream http://127.0.0.1:8080
```

### Available Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--config` | Load a custom TOML config file | `--config /path/to/custom.toml` |
| `--port` | Change the listening port | `--port 9000` |
| `--host` | Bind to a specific network interface | `--host 192.168.1.100` |
| `--upstream` | Change the llama.cpp server URL | `--upstream http://127.0.0.1:8080` |
| `--searxng-port` | Change the port for managed SearXNG | `--searxng-port 9999` |

> Note: there is no `--api-key` flag and no enforced client key — the key comparison
> (when a key *is* provided) uses `upstream.api_key`. See the Auth notes in `AGENTS.md`.

---

## Using the API

The chat endpoint does **not** currently enforce a client key, but it is good
practice to send one anyway (the API accepts either method below; any provided key
is only compared on `/monitor/*` routes, against `upstream.api_key`):

**Method 1: `X-API-Key` header (recommended)**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: a-prox-s3cur3k3y" \
  -d '{
    "model": "qwen3.6-35b-moe",
    "messages": [{"role": "user", "content": "Hello, who am I talking to?"}]
  }'
```

**Method 2: `Authorization: Bearer` header**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer a-prox-s3cur3k3y" \
  -d '{
    "model": "qwen3.6-35b-moe",
    "messages": [{"role": "user", "content": "Hello, who am I talking to?"}]
  }'
```

### Streaming Responses

Add `"stream": true` to your request body and use the `-N` (no-buffering) flag with curl:

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: a-prox-s3cur3k3y" \
  -d '{
    "model": "qwen3.6-35b-moe",
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a story."}]
  }'
```

### Health Check (no API key required)

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics       # same as /health, included for compatibility
```

---

## Built-in SearXNG

A-PROX can automatically start a local SearXNG instance when launched. This enables web search without running a separate service.

### How it works

When you start A-PROX, it checks if SearXNG is installed. If not, it:
1. Creates a Python virtual environment in `searxng/.venv/`
2. Installs `searxng` via pip
3. Generates a `settings.yml` with search engines enabled
4. Starts SearXNG as a background process on port 8888
5. Waits up to 30 seconds for SearXNG to become healthy

On subsequent launches, it skips installation and starts SearXNG immediately.

### Requirements

- **Python 3.10+** (check with `python3 --version`)
- **pip** (check with `python3 -m pip --version`)

### SearXNG Configuration

```toml
[searxng]
enabled = true                    # Enable managed SearXNG (default: true)
port = 8888                       # Port for managed SearXNG (must match search.searxng_url port)
install_dir = "~/.local/share/a-prox-searxng"  # Where SearXNG is installed
listen_port = 8888                # Internal SearXNG listen port
search_engines = "google,bing,duckduckgo,wikipedia,github"  # Comma-separated engine list
max_results = 5                   # Max results per search query
```

### Disabling Built-in SearXNG

If you already have SearXNG running externally:

```toml
[searxng]
enabled = false
```

Or skip the install entirely by removing the `searxng/` directory.

### Using Web Search

Once SearXNG is running (either built-in or external), the LLM can use web search via:

**Direct message trigger:**
```json
{"role": "user", "content": "Search the web for Rust async best practices 2026"}
```

**Via tool call:**
```json
{
  "role": "assistant",
  "content": "I'll search for that.\n<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"Rust async best practices 2026\"}}</tool_call>",
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "web_search",
        "arguments": "{\"query\": \"Rust async best practices 2026\"}"
      }
    }
  ]
}
```

### Health Check (no API key required)

---

## Configuration

Edit `config/default.toml` or create your own TOML file. All settings have sensible defaults and can be overridden via CLI flags.

```toml
[server]
host = "0.0.0.0"        # Network interface to bind
port = 8000              # Port A-PROX listens on
# NOTE: there is no server.api_key field. The sample value below is stale;
# client-key comparison (when a key is provided) uses upstream.api_key.

[upstream]
base_url = "http://127.0.0.1:8080"  # Where llama.cpp runs (may be A-PROX-owned)
api_key = "change-me"               # API key sent TO llama.cpp AND key compared
#                                    # against when a client supplies a key
model_alias = "qwen3.6-35b-moe"     # Display name for the model
timeout_seconds = 180               # Max wait time for llama.cpp responses

[guardrails]
max_concurrent_inferences = 1       # Max simultaneous generations (protects DDR4)
queue_depth = 16                    # Max queued requests (excess get HTTP 429)
rate_limit_per_minute = 120         # Requests per client per minute
min_free_ram_gb = 16.0              # Reject bulk operations if RAM drops below this

[context]
max_context_tokens = 65536          # Max tokens in conversation (prunes older messages)
reserve_completion_tokens = 8192    # Reserve tokens for the LLM's response
sliding_window_turns = 7            # Keep this many recent turns verbatim

[embeddings]
model_path = "models/bge-small-en-v1.5-int8.onnx"  # Path to ONNX embedding model
cpu_threads = 4                             # Threads for ONNX inference
dimension = 384                             # Embedding vector size (matches the model)

[db]
path = "data/a_prox.db"     # SQLite database location (persists RAG documents)
mmap_size_mb = 16384        # Memory-mapped I/O size in MB (uses ~16 GB RAM)
cache_size_mb = 1024        # SQLite page cache size in MB

[search]
searxng_url = "http://127.0.0.1:8888"  # Local SearXNG instance (leave empty or use fallback)
timeout_seconds = 4                     # Web search timeout
max_results = 5                         # Max search results per query
cache_ttl_seconds = 86400               # Web cache validity (24 hours)

[ingestion]
directories = []                        # Directories to auto-index at startup
file_extensions = ["txt", "md", "json", "yaml", "yml", "toml", "html", "htm", "log", "csv", "xml"]
excluded_patterns = ["*.lock", "*.swp", "*.tmp", ".DS_Store"]
collection = "auto-indexed"             # Default collection for indexed files
max_file_size_kb = 512                  # Skip files larger than this
watch_interval_secs = 30                # Periodic re-scan interval
pdf_enabled = true                      # Send PDFs to upstream for text extraction
pdf_upstream_model = "qwen3.6-35b-moe"  # Vision model for PDF processing
```

**Key settings to customize:**
- **`upstream.api_key`** — Must match the key the llama.cpp server is launched with (and what image-gen's `[llama_server].api_key` uses). A-PROX sends it as `Authorization: Bearer` upstream.
- **`upstream.base_url`** — Must point to the llama.cpp server (`http://127.0.0.1:8080` for the A-PROX-owned instance).
- **`db.path`** — Where RAG documents are stored. Change if you want a different location.
- **`search.searxng_url`** — Set to `http://127.0.0.1:8888` if you have SearXNG; leave as-is for mock fallback.

---

## RAG (Retrieval-Augmented Generation)

A-PROX includes a full RAG pipeline that lets your LLM answer questions using documents stored in a local SQLite vector database. All embeddings and vector search run on CPU with zero GPU memory — the ONNX embedding model processes text locally.

### Architecture Overview

```
Text Files ──→ Chunking ──→ ONNX Embeddings ──→ SQLite Vector Store
                                                      │
User Query ──→ ONNX Embeddings ──→ Hybrid Search ────┘
                              (Vector + FTS5 + RRF)
                                                      │
                                                      ▼
                                            Retrieved Chunks Injected into Context
                                                      │
                                                      ▼
                                            Forwarded to llama.cpp for Answering
```

The system uses **hybrid search** combining three techniques:
1. **Vector similarity** — cosine distance on 384-dimensional BGE embeddings
2. **Full-text search** — SQLite FTS5 with porter stemmer
3. **Reciprocal Rank Fusion (RRF)** — merges both result rankings with k=60 constant

This gives the best of both approaches: semantic understanding from vectors, precise keyword matching from FTS, and robust ranking from RRF fusion.

---

### How RAG is Triggered

There are multiple ways to trigger RAG search or ingestion:

#### 1. Model Targeting
Send requests with `model` set to one of these aliases:

| Model Alias | Behavior |
|-------------|----------|
| `a-prox-rag`, `a-prox-knowledge`, `a-prox-docs` | Route to RAG augmented search |
| `a-prox-agent`, `a-prox-tools` | Route to agentic tool loop (uses `rag_search` tool) |
| `a-prox-direct`, `a-prox-pass`, `a-prox-fast` | Fast pass-through (no RAG) |

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: a-prox-s3cur3k3y" \
  -d '{
    "model": "a-prox-rag",
    "messages": [{"role": "user", "content": "What does my notes say about Rust async?"}]
  }'
```

#### 2. Slash Commands
Prefix your message with `/rag`, `/knowledge`, or `/docs`:

```
/rag How do I configure the vector store?
/knowledge What is in my knowledge base?
/docs Tell me about the architecture
```

#### 3. Tool Command Flags

Prefix your message with any of the per-tool `/flags` to **force the agentic
loop** with that tool armed. Each flag is configurable under `[tool_commands]`
and is disabled when set to `""`. Message text after the flag is passed to the
model as the tool's request.

| Flag | Tool armed | Behavior |
|------|-----------|----------|
| `/tools [names...]` | all internal tools | Generic agentic loop; add tool-name args (e.g. `/tools search fetch`) to arm an exact subset |
| `/search` | `web_search` | Loop restricted to local web search |
| `/fetch` | `web_fetch` | Loop restricted to page fetching/scraping |
| `/ragsearch` | `rag_search` | Loop restricted to RAG store search |
| `/ingest` | `rag_ingest` | Loop restricted to RAG store ingestion |
| `/time` | `system_time` | Loop restricted to current time/date |
| `/image` | `image_generate` | Loop restricted to image generation — t2i by default, **i2i automatically when a photo is attached** |
| `/file` | `write_file` | Loop restricted to saving a text file (`/files/{name}`) |

`/tools` accepts leading tool-name args to restrict the loop to exactly those
tools (aliases like `search`, `fetch`, `rag`, `time`, `image`, `file` work; the
remaining text becomes the request):

```
/tools search fetch what are the current GPU prices?
/tools rag time compare notes on embeddings
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model": "qwen3.6-35b-moe",
       "messages": [{"role": "user", "content": "/image a red circle on white"}]}'
```

Config example (each key is a string; empty string disables that flag):

```toml
[tool_commands]
# Agentic-loop / flag commands
agentic = "/tools"          # /tools search fetch ...  arms a subset by args
web_search = "/search"
web_fetch = "/fetch"
rag_search = "/ragsearch"
rag_ingest = "/ingest"
system_time = "/time"
image_generate = "/image"
write_file = "/file"
# Routing-only commands (empty string disables)
bypass = "/bypass"          # → FastPassThrough
direct = "/direct"          # → FastPassThrough
pass = "/pass"              # → FastPassThrough
rag = "/rag"                # → RAG search
knowledge = "/knowledge"    # → RAG search
docs = "/docs"              # → RAG search
```

#### 4. Natural Language Triggers
The router detects intent from message content:

| Intent | Trigger Phrases |
|--------|-----------------|
| RAG Search | `"search docs"`, `"in knowledge base"`, `"from my files"`, starts with `/rag` |
| RAG Ingestion | `"ingest into rag"`, `"save to knowledge base"`, `"index this"`, `"store in rag"` |

```json
{"role": "user", "content": "Search docs for embedding configuration"}
```

```json
{"role": "user", "content": "Ingest into rag: The vector store uses sqlite-vec with cosine distance"}
```

#### 5. Agentic Tool Loop
When routed to `AgenticToolLoop`, the LLM can autonomously decide to call:

| Tool | Purpose |
|------|---------|
| `rag_search` | Search the RAG store for relevant chunks |
| `rag_ingest` | Ingest new text into the RAG store |

Example: User asks `"What's the current price of Bitcoin?"` → Router sends to AgenticToolLoop → LLM calls `web_search` for price → `rag_search` for historical context → combines results in final answer.

---

### Directory-Based Automatic Indexing

A-PROX can automatically scan directories on your filesystem, extract text from files, and index them into the RAG store. This happens at startup and continues watching for changes in the background.

#### Configuration

```toml
[ingestion]
# Directories to scan for indexing
directories = [
    "/home/user/notes",
    "/home/user/projects/docs",
    "~/my-knowledge-base"
]

# File extensions to index (txt, md, json, yaml, toml, html, csv, xml, log)
file_extensions = ["txt", "md", "json", "yaml", "yml", "toml", "html", "htm", "log", "csv", "xml"]

# Patterns to exclude (glob-style)
excluded_patterns = ["*.lock", "*.swp", "*.tmp", ".DS_Store"]

# Default collection name for all indexed files
collection = "auto-indexed"

# Maximum file size in KB (files larger than this are skipped)
max_file_size_kb = 512

# How often to re-scan directories (seconds) — passive, non-aggressive
watch_interval_secs = 30

# Enable PDF processing via upstream llama.cpp multimodal endpoint
pdf_enabled = true

# Model to send PDFs to for text extraction (must support vision)
pdf_upstream_model = "qwen3.6-35b-moe"
```

#### How It Works

1. **Startup** — A-PROX scans all configured directories recursively (up to 20 levels deep).
2. **Text Extraction** — Each file type is handled differently:
   - **Text files** (`txt`, `md`, `log`, `csv`, `xml`) — Raw content read
   - **Config files** (`json`, `yaml`, `toml`) — Serialized to string format
   - **HTML files** — Cleaned using readability extraction (strips nav, scripts, ads)
   - **PDF files** — Sent to upstream llama.cpp multimodal endpoint for text extraction, falls back to raw read on failure
3. **Chunking** — Each file is split into ~512-token chunks at paragraph/sentence boundaries with 64-token overlap.
4. **Embedding** — Each chunk gets a 384-dimensional vector embedding via ONNX BGE-small on CPU.
5. **Storage** — Chunks are stored in SQLite with their embeddings and FTS5 indices.
6. **Collection Naming** — Each subdirectory becomes a separate collection (e.g., `notes/` → `notes` collection). PDFs go to the default collection.
7. **Deduplication** — Files are hashed with SHA-256. Only changed files are re-indexed.
8. **Live Watching** — The `notify` crate watches directories for file changes. When a file is modified, it's re-indexed within a 500ms debounce window. Periodic full scans run at the configured interval.

#### Example Directory Structure

```
/home/user/notes/
├── rust/
│   ├── async_patterns.md      → Collection: "rust"
│   └── error_handling.md      → Collection: "rust"
├── architecture/
│   ├── system_design.md       → Collection: "architecture"
│   └── api_docs.json          → Collection: "architecture"
└── reports/
    └── quarterly_review.pdf   → Collection: "auto-indexed" (PDFs use default collection)
```

#### PDF Processing

PDFs are sent to your upstream llama.cpp server as multimodal requests. The model receives the PDF as a base64-encoded image and extracts text. This requires:

1. `pdf_enabled = true` in config
2. A vision-capable model at your upstream endpoint
3. Sufficient upstream response time (PDFs are larger requests)

If the upstream fails, A-PROX falls back to reading the raw PDF bytes as text (unstructured but usable).

---

### Manual Ingestion (Per-Request)

You can ingest individual documents through the API without configuring directories.

#### Via Message Content
Send a message containing an ingestion trigger phrase:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: a-prox-s3cur3k3y" \
  -d '{
    "model": "qwen3.6-35b-moe",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with RAG access."},
      {"role": "user", "content": "Ingest into rag: The quick brown fox jumps over the lazy dog. This is a sample document for testing RAG."}
    ]
  }'
```

#### Via Agentic Tool Loop
Request a message that triggers the tool loop, and the LLM will call `rag_ingest`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: a-prox-s3cur3k3y" \
  -d '{
    "model": "qwen3.6-35b-moe",
    "messages": [
      {"role": "user", "content": "Save this to the knowledge base: A-PROX uses hybrid search combining vector similarity with FTS5 full-text search."}
    ]
  }'
```

---

### RAG API Endpoints

#### Reindex All Directories
```bash
curl -X POST http://localhost:8000/ingestion/reindex \
  -H "X-API-Key: a-prox-s3cur3k3y"
```

Response:
```json
{
  "status": "success",
  "files_indexed": 12,
  "files_skipped": 3,
  "files_errors": 0,
  "chunks_total": 47
}
```

#### Get Ingestion Status
```bash
curl http://localhost:8000/ingestion/status \
  -H "X-API-Key: a-prox-s3cur3k3y"
```

Response:
```json
{
  "status": "success",
  "files_indexed": 12,
  "files_skipped": 3,
  "files_errors": 0,
  "chunks_total": 47,
  "indexed_files_count": 12,
  "indexed_files": [
    {
      "path": "/home/user/notes/rust/async_patterns.md",
      "collection": "rust",
      "chunks": 5,
      "last_indexed": 1726234567
    }
  ]
}
```

---

### How Retrieved Context Is Injected

When a RAG search returns results, the retrieved chunks are formatted and injected into the message history before forwarding to llama.cpp:

```
### Relevant Retrieved Context:

[Citation 1 | Source: notes/async_patterns.md | Score: 0.847]
async/await allows writing asynchronous code that looks synchronous...

[Citation 2 | Source: notes/error_handling.md | Score: 0.723]
Rust's Result<T, E> type enables explicit error handling without exceptions...

### End Retrieved Context

<original user query>
```

The LLM sees these citations as part of the conversation context and references them in its response. Citations include source file, relevance score, and the chunk content.

---

### Database Schema

The RAG system stores data in a single SQLite database at the path configured in `[db]`. The database contains:

**`document_chunks`** — Chunk metadata
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment row ID |
| collection | TEXT | Collection name (directory basename or default) |
| source_uri | TEXT | Original file path |
| chunk_index | INTEGER | Position within source file |
| content | TEXT | The actual text chunk |
| token_count | INTEGER | Approximate token count |
| created_at | INTEGER | Unix timestamp |

**`document_chunks_fts`** — Full-text search index
- Virtual table using FTS5 with porter unicode61 stemmer
- Enables keyword matching on chunk content

**`vec_chunks`** — Vector embeddings
- Virtual table using sqlite-vec (vec0) with cosine distance
- 384-dimensional float vectors (BGE-small model output)

**`ingestion_sources`** — Ingestion tracking
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| source_path | TEXT UNIQUE | File path |
| content_hash | TEXT | SHA-256 hash of file contents |
| last_indexed | INTEGER | Unix timestamp of last indexing |
| chunks_count | INTEGER | Number of chunks created |
| collection | TEXT | Collection name |

**`request_logs`** — Request audit trail
- Tracks prompt hashes, model, tokens, latency for monitoring

---

### Tuning RAG Performance

The RAG system runs entirely on CPU with minimal memory footprint:

- **ONNX Model**: `bge-small-en-v1.5-int8` (~67 MB) runs in ~50-100ms per embedding on modern CPUs
- **sqlite-vec**: Uses SIMD-accelerated cosine distance for vector search
- **RRF Fusion**: k=60 constant balances vector vs. lexical importance
- **Chunking**: 512-token chunks with 64-token overlap preserves context continuity

You can tune chunking in `src/rag/mod.rs`:
```rust
// Default: 512 token max chunk, 64 token overlap
Self { chunker: TextChunker::new(512, 64) }
```

Smaller chunks = more precise retrieval but more vectors. Larger chunks = fewer vectors but potentially less precise. The 512/64 balance works well for most documentation and note-taking use cases.

---

### Troubleshooting RAG

**"No results found" for my query**
- Check that documents are actually indexed: `curl http://localhost:8000/ingestion/status`
- Verify the query matches the content semantically (try different phrasing)
- Check that the `[db]` directory has write permissions

**Slow ingestion**
- The ONNX embedding model runs on CPU; large directories with many files will take time
- Check CPU usage during ingestion; A-PROX uses 4 threads for embedding by default
- Reduce `max_file_size_kb` to skip oversized files
- Reduce `watch_interval_secs` to scan less frequently

**PDFs not extracting text**
- Verify `pdf_enabled = true` in config
- Check that your upstream llama.cpp model supports vision (multimodal)
- Check upstream logs for errors
- Falls back to raw PDF byte reading if upstream fails

**Stale indexed content**
- Run `POST /ingestion/reindex` to force a full re-index
- Check that file modification times are updating correctly
- Verify the notify watcher is active (check startup logs for "Set up file watcher")

---

## Image Generation

A-PROX can generate images through its `image_generate` tool, backed by a managed
ComfyUI (Qwen-Image 2.1 GGUF) backend. Full design/wiring details live in
`imagegen_implementation_plan.md`; this section covers usage and config.

### How it works

1. **Detection** — A request counts as image-requested when the latest user turn
   contains t2i verb phrases (`paint`, `draw a picture of`, `make a logo/image`, …)
   or a t2i phrase **plus an attached image part** (→ image-to-image). It is routed
   into the normal agentic tool loop with `image_generate` armed. The `/image`
   flag does the same explicitly (`is_image_request` bypassed): t2i unless a photo
   is attached, in which case the i2i pipeline (reference dims) runs.
2. **Phase A (rewrite)** — llama.cpp (vision) rewrites the request into a
   workflow-specific prompt per `prompts/t-iprompt.txt` / `prompts/i-iprompt.txt`,
   extended with a `HARNESS_DIRECTIVE` (JSON schema for the tool call). If the model
   emits bare JSON instead of a structured tool call, `try_parse_image_generate_json`
   recovers it.
3. **Phase B (generate)** — A-PROX ensures ComfyUI is up (port `8188`), SIGTERMs
   llama.cpp (≤30s, SIGKILL fallback), uploads the reference image (i2i), injects
   `workflows/t2i.json` / `i2i.json` (per-job random KSampler seed), submits and polls the job
   (≤ `generation_timeout_s`), downloads the PNG via `/view`, saves it to the serve
   dir, then **restarts llama.cpp** (≤120s health wait). The concurrency permit is
   held for the whole job (chat requests queue during the llama downtime).
4. **Phase C (return)** — A synthetic user message (text + base64 data-URL of the
   PNG) is appended so llama.cpp (vision) can stream a caption. One
   `delta.image_url` SSE event is emitted before the text (non-streaming: top-level
   `image_url` on the response JSON). The PNG is also served at
   `GET /images/{name}`.

Resolution math (`src/imagegen/ratio.rs`): target 2 MP from `wh_ratio`, both sides
rounded to a multiple of 16, capped at 4096; `ratio_follow="<image1>"` uses the
reference image's own dimensions.

### Wire contract for image results

- **Streaming:** the choice delta carries `delta.image_url` (a data URL or served
  URL) followed by normal text deltas. **Never send `delta.content` as a JSON
  array** — CLAN casts it to String and throws.
- **Tool result** (given to the model for captioning):
  `{"status":"ok","image_url":...,"image_path":...,"prompt":...}`.
- During a generation job llama.cpp is down, so `/health` and `/v1/models` may
  error until it restarts; this is expected.

### Config

```toml
[llama_server]            # llama.cpp that A-PROX owns (stopped during image jobs)
enabled = true
executable = "/home/…/llama.cpp/build/bin/llama-server"
host = "127.0.0.1"
port = 8080
api_key = "…"             # must match upstream.api_key and [comfy_ui].api_key

[comfy_ui]                # managed ComfyUI backend (paused models auto-unload ~0.4 GB)
enabled = true
host = "127.0.0.1"
port = 8188
workdir = "/path/to/ComfyUI"
python = "/path/to/ComfyUI/.venv/bin/python"
main_py = "main.py"
api_key = "change-me"
listen_timeout_secs = 30
shutdown_timeout_secs = 10

[image_generation]
enabled = true
workflows_dir = "workflows"
prompts_dir = "prompts"
serve_dir = "data/generated_images"
public_base_url = ""            # if set, image_url points here instead of http://127.0.0.1:{port}
poll_interval_ms = 1500
generation_timeout_s = 180
default_negative_prompt = "lowres, jpeg artifacts, bad anatomy, watermark, signature, extra fingers, cropped, worst quality"
```

An empty `public_base_url` makes served `image_url`s use
`http://127.0.0.1:{server.port}`, which is what CLAN on the same machine or LAN
needs. Point it at your public host for remote clients.

---

## File Generation (write_file)

A-PROX can also have the model **write text files** (scripts, notes, markdown,
CSV, …) and serve them to clients, using the same agentic-loop machinery as images
but without any ComfyUI/llama-stop overhead.

### How it works

1. **Detection** — A request counts as a file-write when the latest user turn
   matches `FILE_KEYWORDS` in `src/filegen/mod.rs` or a writing-verb plus a
   file cue ("write a short python script to a file…"). It is routed into the
   agentic loop with `write_file` armed.
2. **Execution** — The model calls `write_file` with `filename` + `content`
   (+ optional `mode`). A-PROX cleans the filename, rejects dangerous extensions
   (`deny_exts`: `exe`, `sh`, `bat`, `html`, `php`, …), caps `content` at
   `max_content_chars` (and tells the model to split via `mode="append"`), then
   stores the file at `serve_dir` under `{id}_{cleaned-name}`.
3. **Append** — Call the tool again with the **same filename** and
   `mode="append"` to keep writing the same logical file across turns.
4. **Return** — One `delta.file_url` SSE event `{"url","name","mime"}` is emitted
   before the caption text (non-streaming: top-level `file_url`). The file is
   served at `GET /files/{name}`.

### Wire contract for file results

- **Streaming:** `delta.file_url` `{"url","name","mime"}` as the first delta,
  then normal text deltas (may appear alongside `delta.image_url`).
- **Tool result** (given to the model): `{"status":"ok","file_url":...,"file_name":...,"size":...,"mime":...}`.
- Successful `write_file` bodies are redacted from the history stored in the loop
  context (`sanitize_tool_calls_for_history`), so long append-chains don't bloat
  the context window.

### Config

```toml
[file_generation]
enabled = true
serve_dir = "data/generated_files"
public_base_url = ""            # if set, file_url points here instead of http://127.0.0.1:{port}
max_content_chars = 24000       # per-call content cap; larger → model splits with mode=append
deny_exts = ["exe", "sh", "bat", "com", "cmd", "dll", "so", "sys", "html", "htm", "php", "jar"]
```

---

## Using with LLM Frontends

Point any OpenAI-compatible frontend (SillyTavern, OpenWebUI, etc.) to:

```
API Base URL: http://localhost:8000/v1
API Key: a-prox-s3cur3k3y
Model: qwen3.6-35b-moe
```

For remote access (e.g., from your phone), use your server's IP instead of `localhost`:

```
API Base URL: http://192.168.1.100:8000/v1
```

Then start A-PROX:

```bash
./target/release/a-prox --port 8000
```
