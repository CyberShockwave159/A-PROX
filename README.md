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
 │ llama.cpp  │  │ ComfyUI Qwen-IMG │  ← booted per image job, stopped after
 └────────────┘  └──────────────────┘
```

- **Port 8000** — A-PROX gateway. Clients connect here.
- **Port 8080** — llama.cpp server. Only reachable on `localhost` (behind A-PROX). A-PROX launches it from `[llama_server]` (paused during image jobs and restarted after).
- **Port 8188** — Managed ComfyUI backend used for `image_generate`; not started at boot — it is booted on demand for each image job (after llama.cpp is paused) and fully stopped once the job finishes, releasing its VRAM back to llama.cpp.
- **API Key** — Not enforced on `/v1/*`, `/ingestion/*`, `/monitor/*`; when a key *is* provided to `/monitor/api` or `/monitor/stream` it is compared against `upstream.api_key`.

---

## Hardware Optimization

A-PROX was built for a machine where the GPU is fully occupied:
- **GPU (RTX 3080 Ti 12GB):** 0 MB used by A-PROX. All proxy logic runs on CPU.
- **CPU (Ryzen 9 5900X):** llama.cpp uses 6 threads; A-PROX uses the remaining 18.
- **RAM (128 GB):** 16 GB for SQLite memory-mapped I/O, 1 GB page cache, rest free.
- **Concurrency limit:** Max 1 active generation to prevent memory-bandwidth bottlenecks.

---

## Client Capability Reporting

`GET /health` advertises what this build can do, so a client can enable the
matching features without a separate probe:

```json
{
  "status": "healthy",
  "service": "A-PROX",
  "version": "1.9.0",
  "capabilities": ["rag", "image", "file"],
  "hardware": { "...": "..." },
  "concurrency": { "...": "..." },
  "upstream": { "endpoint": "...", "model_alias": "..." }
}
```

`rag` is always present. `image` and `file` mirror `[image_generation].enabled`
and `[file_generation].enabled`. A client that gets no `capabilities` key (any
older build, or a plain llama.cpp server) should assume no A-PROX features.

---

## Direct RAG Store Access

The RAG store is usable as a **memory backend** for a client that wants recall
without routing a generation through the LLM. These three endpoints are pure
store operations: no model call, no inference permit, no acknowledgement turn.

> Why not the `/ingest` flag or the `RAGIngestion` route? Both forward upstream
> afterwards for a model acknowledgement, so every ingested turn costs a
> generation — and `RAGIngestion` hardcodes the shared `default` collection, so
> a client cannot scope memories per conversation.

### `POST /rag/ingest`

```json
{ "collection": "clan_<characterId>_<threadId>", "source_uri": "clan/c/t/m1", "content": "User: hi
Alice: hello" }
```

```json
{ "status": "ok", "chunks": 2, "collection": "...", "source_uri": "..." }
```

Idempotent by `source_uri`: chunks previously stored for the same
`(collection, source_uri)` pair are deleted first, so re-ingesting a rewritten
document replaces it instead of accumulating duplicates. `source_uri` is
optional; a timestamped one is generated when omitted.

### `POST /rag/query`

```json
{ "query": "what did we agree on", "collection": "clan_<characterId>_<threadId>", "top_k": 3, "min_score": 0.35 }
```

```json
{ "status": "ok", "results": [ { "chunk_id": 7, "collection": "...", "source_uri": "...", "chunk_index": 0, "content": "...", "score": 0.81 } ] }
```

All three filters are optional. Omitting `collection` searches **every**
collection, which is rarely what a client with per-conversation memories wants.
`top_k` defaults to 5 and is clamped to `[1, 50]`; `min_score` is clamped to
`[0, 1]`.

### `GET /rag/collections/{name}/count`

```json
{ "status": "ok", "collection": "clan_abc_t1", "chunks": 12, "chunks_total": 340 }
```

Lets a client distinguish an empty collection (needing a backfill) from a
populated one.

---

## Request Fields

A few body fields are A-PROX-specific. They are consumed and stripped before
the payload is forwarded upstream, so they never reach llama.cpp.

### `model` — routing aliases

`a-prox-rag`, `a-prox-knowledge`, `a-prox-docs`, `a-prox-agent`, `a-prox-tools`,
`a-prox-direct`, `a-prox-pass`, `a-prox-fast` select a *strategy*, not a model.
A-PROX rewrites them back to `[upstream].model_alias` before forwarding. (If
you set an alias and A-PROX did not rewrite it, that is a bug — a strict
OpenAI-compatible backend will reject the request.)

### `rag` — retrieval tuning

```json
"rag": { "collection": "clan_abc_t1", "top_k": 3, "min_score": 0.35 }
```

Consumed by the `RAGAugmented` route, which otherwise searches all collections
and takes 5 hits. Absent → the historical behaviour, unchanged.

### `roleplay` — per-message tool restriction

Tag a message with `"roleplay": true` and A-PROX restricts the tools that message
can reach to `rag_search`, `rag_ingest` and `image_generate`. Intended for
immersive roleplay clients, where a turn must never break character to run a web
search, read the clock, or write a file.

```json
{"role": "user", "content": "I smile and say hello", "roleplay": true}
```

Enforcement is in `src/router/mod.rs` (narrowing a forced tool list) and
`src/server/routes.rs` (filtering the armed tool schemas), so it covers the
forced-flag, client-supplied-tools and all-internal paths alike. A `/flag` naming
only a forbidden tool (`/search`, `/fetch`, `/time`, `/file`) downgrades to a
plain pass-through rather than arming it.

### Per-turn generation budget

`guardrails.max_generation_tokens` (default `4096`) is sent as `max_tokens` on
every agentic-loop turn — but **only when the client didn't set its own**.
`0` omits the field and restores the upstream default.

This exists because A-PROX otherwise inherits the upstream's cap, and
llama.cpp's is 2048. That is ample for a turn that calls a tool immediately, but
an image turn has to *plan* first: Phase A rewrites the prompt, and only then
does the model call `image_generate`. A model that reasons at length gets cut
off at 2048, the tool call is never emitted, and the request completes
"successfully" with no image and no error — the failure is invisible from the
client. Raise the value for such models; lower it to bound the cost of a turn
that fails to converge.

### `image_only` — artifact only, no text

```json
"image_only": true
```

Return the image and stop. Skips two upstream turns that exist only to produce
*text about* the picture: the vision turn that hands the generated image back to
the model so it can stream a caption, and the final synthesis turn (plus its
non-streaming fallback when the caption comes back empty).

The `delta.image_url` event is emitted by A-PROX itself, so the artifact does not
depend on either turn. Measured on a 35B MoE model, this removes ~85s from a
~200s image request.

Use it when the client discards the reply text — e.g. attaching a generated
picture to an existing message it is going to keep verbatim. Leave it unset when
you want the caption (the default, and what a conversational "draw me a cat"
expects).

Only short-circuits when the image was actually produced; a failed generation
falls through so the model can report why, exactly as before.

### `image_style` — visual style

```json
"image_style": "anime"
```

Selects an entry from `[image_generation.styles]`. The style contributes a
`prompt_suffix` appended to the final prompt and a `negative_prompt` that
**replaces** `image_generation.default_negative_prompt`. Both are applied in
`WorkflowTemplate::apply`, i.e. **after** the prompt enhancer has rewritten the
prompt — which is the point: a style stated only in the prompt text is at the
mercy of a rewriter. Unknown keys fall back to the neutral `default` style.

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
4. **Optional: ComfyUI (image generation)** — For `image_generate` to work, a Qwen-Image 2.1 GGUF ComfyUI install is configured in `[comfy_ui]`. When enabled, A-PROX boots it on demand for each image job — after pausing llama.cpp — and stops it again once the job finishes, so it never holds VRAM while idle. Both backends' stdout/stderr are piped to the A-PROX console so you can watch model-load progress from one terminal.

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
  "version": "1.9.0",
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

A client can also treat this store as a **memory backend**, writing and reading
conversational turns directly via `/rag/ingest` and `/rag/query` (see [Direct
RAG Store Access](#direct-rag-store-access)) and recalling them through the
`a-prox-rag` route. Scoping memories to a `collection` per conversation is what
keeps one session's memories out of another's retrievals.

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

Scoping: a `collection` is the unit of isolation. `query_rag` searches *all*
collections when no filter is given, so a client that stores memories for
several conversations must pass its own `collection` — either through
`/rag/query` or through the `rag` request object on a chat request.

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
3. **Phase B (generate)** — A-PROX SIGTERMs llama.cpp (≤30s, SIGKILL fallback),
   then boots ComfyUI for this job only (port `8188`), uploads the reference image
   (i2i), injects `workflows/t2i.json` / `i2i.json` (per-job random KSampler seed),
   submits and polls the job (≤ `generation_timeout_s`), downloads the PNG via
   `/view`, saves it to the serve dir, then **fully stops ComfyUI** (releasing its
   VRAM) and **restarts llama.cpp** (≤ `health_timeout_s`, default 600s — long
   enough for cold boots). The concurrency permit is
   held for the whole job (chat requests queue during the llama downtime).
4. **Phase C (return)** — A synthetic user message (text + the generated image
   served from A-PROX's *own loopback* `/images/{name}` endpoint, passed as an
   `image_url` part) is appended so llama.cpp (vision) can stream a caption —
   the multi-MB PNG bytes never enter the LLM context (only llama's bounded
   vision tokens do). One `delta.image_url` SSE event is emitted before the text
   (non-streaming: top-level `image_url` on the response JSON); that client-facing
   payload is a base64 data-URL when `inline_data_url` is enabled, otherwise the
   served `http(s)://…/images/…` URL. The PNG is also served at `GET /images/{name}`.

Only llama.cpp is started eagerly at A-PROX boot (see `state.rs`); ComfyUI is
booted on demand per image job and never stays up between requests.
Both backends' stdout/stderr are piped to the A-PROX console, so llama.cpp token
counts and ComfyUI workflow progress appear in the same terminal you launched
A-PROX from.

Resolution math (`src/imagegen/ratio.rs`): target 2 MP from `wh_ratio`, both sides
rounded to a multiple of 16, capped at 4096; `ratio_follow="<image1>"` uses the
reference image's own dimensions.

### Wire contract for image results

- **Streaming:** the choice delta carries `delta.image_url` (a data URL or served
  URL) followed by normal text deltas. **Never send `delta.content` as a JSON
  array** — CLAN casts it to String and throws.
- **Tool result** (given to the model; always a small served URL, never base64 —
  the base64 we generate is client-only):
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
api_key = "…"             # must match upstream.api_key
health_timeout_s = 600    # cold boots (HDD model load) can take minutes

[comfy_ui]                # managed ComfyUI backend, booted per image job
enabled = true
url = "http://127.0.0.1:8188"
workdir = "/path/to/ComfyUI"            # ComfyUI install directory (the venv lives here)
python = "/path/to/ComfyUI/bin/python"  # run through the venv's interpreter
args = ["main.py", "--enable-manager"]
health_timeout_s = 600

[image_generation]
enabled = true
mp = 2.0
multiple = 16
max_side = 4096
unet = "qwen-image-Uncensored-2.1-Q4_K_M.gguf"
clip = "qwen3vl_8b_int8_convrot.safetensors"
vae = "qwen_image_2.1_vae_bf16.safetensors"
t2i_workflow = "workflows/t2i.json"
i2i_workflow = "workflows/i2i.json"
t2i_prompt_file = "prompts/t-iprompt.txt"
i2i_prompt_file = "prompts/i-iprompt.txt"
serve_dir = "data/generated_images"
public_base_url = ""            # if set, image_url points here (highest priority)
inline_data_url = false         # true → emit image_url as a base64 data-URL (needs client-side decoding)
poll_interval_ms = 2000
generation_timeout_s = 180
default_negative_prompt = "bad anatomy, bad composition, bad lighting, distorted face, extra limbs, low quality, out of focus, overexposed, plastic, poor symmetry, signature, watermark, ugly, censored"

# Named visual styles, selected per request with the `image_style` field.
# Applied to the prompt and negative prompt when the ComfyUI graph is built —
# i.e. AFTER the prompt enhancer rewrote the prompt, so a model rewriter cannot
# dilute the style. A style's `negative_prompt` REPLACES the
# `default_negative_prompt` above, which is realism-leaning and actively fights
# the non-photographic styles.
[image_generation.styles.default]
prompt_suffix = ""
negative_prompt = ""

[image_generation.styles.anime]
prompt_suffix = "anime key visual, cel-shaded, clean line art, flat colour blocking, expressive eyes"
negative_prompt = "photorealistic, realistic skin pores, 3d render, cgi, photographic, airbrushed, soft gradients"

[image_generation.styles.semi-realistic]
prompt_suffix = "semi-realistic digital illustration, soft painterly shading, subsurface skin scattering, detailed fabric texture"
negative_prompt = "flat cel shading, chibi, plastic skin, harsh 3d render, fully photographic, anime line art"

[image_generation.styles.photo-realistic]
prompt_suffix = "photorealistic photograph, natural skin texture, shallow depth of field, physically accurate lighting, 35mm lens"
negative_prompt = "illustration, anime, cel-shaded, painting, cgi, 3d render, plastic skin, oversaturated"
```

`default` is the neutral fallback used when a request omits `image_style` or
names one that isn't configured; it adds nothing and suppresses nothing. See
[Request Fields](#request-fields) for the client side.

### Character consistency across images

The i2i workflow feeds the attached image to `TextEncodeQwenImage21` as
`images.image_1` while the KSampler runs at `denoise: 1` with a latent from the
same encoder. That is reference *conditioning*, not a latent init — the model
receives the subject as conditioning tokens, which is why a character keeps a
recognisable face across unrelated scenes. It is the mechanism behind
"image-to-image automatically when a photo is attached".

Two consequences for clients:

- The reference **must** be PNG, JPEG or WebP. A-PROX sniffs magic bytes
  (`guess_format`) and silently falls through to text-to-image for anything
  else — a request carrying an AVIF reference produces a plausible image that
  ignored the reference, with no error.
- The reference must arrive as an `image_url` part on the **latest user
  message**, and only array content is inspected — a plain string is ignored.

There are two independent URL representations — the LLM always sees the small
served URL; the client sees either that served URL or an inline data-URL:

- **Client-facing** `image_url`/`file_url` (SSE `delta.image_url` /
  `delta.file_url`, non-streaming outer JSON): resolved in this order —
  1. `inline_data_url: true` → a base64 `data:` URI (works anywhere the client
     can decode the payload, even when `/images`/`/files` isn't routable; the
     client must decode with `Image.memory`, not `Image.network`);
  2. config `public_base_url` (explicit override; point it at your public host
     for remote clients);
  3. `X-Forwarded-Proto` + `X-Forwarded-Host` (when A-PROX sits behind a TLS
     reverse proxy);
  4. the request `Host` header — the address the client actually dialed
     (loopback, LAN IP, public IP, or proxy domain);
  5. fallback `http://127.0.0.1:{server.port}` for direct loopback use.
- **LLM-facing** (tool results + the Phase C captioning message): ALWAYS the
  small served URL, never base64 — so the generated PNG's multi-MB bytes never
  enter the KV cache. The caption image is served to llama.cpp from A-PROX's own
  loopback `/images/{name}` endpoint.

> **Operational note on `inline_data_url`:** base64 data-URLs demand client-side
> byte decoding (`Image.memory`), and the CLAN Flutter clients (Android APK, Linux
> desktop, PWA) render served-URL images far more reliably — the inline path has
> proven flaky on Android in particular. Keep `inline_data_url = false` (default)
> and rely on `public_base_url` / the request `Host` header instead; the client
> is expected to always display the artifact URL alongside the inline image so a
> tap-through link remains available even if the render or download fails.

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
public_base_url = ""            # if set, file_url points here (highest priority)
inline_data_url = false         # true → emit file_url as a base64 data-URL (needs client-side decoding)
max_content_chars = 24000       # per-call content cap; larger → model splits with mode=append
deny_exts = ["exe", "sh", "bat", "com", "cmd", "dll", "so", "sys", "html", "htm", "php", "jar"]
```

File URLs resolve through the same `image_url`/`file_url` order above (config →
forwarded headers → request `Host` → loopback); see the image-generation
configuration table for an inline base64 data-URL option.

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
