# A-PROX Agent Instructions

## Port Map
- **8000** → A-PROX (public).
- **8080** → llama.cpp upstream (localhost only). **Owned by A-PROX** since image-gen (spawned without `--load-mode mlock` from `[llama_server]` config). Must run with `--api-key` matching `upstream.api_key` + `[llama_server].api_key`; A-PROX sends that key as `Authorization: Bearer` on every upstream call (e.g. `routes.rs:63`). Both managed subprocesses (llama + ComfyUI) are spawned with `stdout`/`stderr` inherited, so their logs appear on the A-PROX console.
- **8188** → Managed ComfyUI backend (`[comfy_ui]`, image generation). **Not** started at boot; booted per image job (after llama.cpp stops) and fully stopped once the job finishes (VRAM returned to llama.cpp).
- **8888** → Managed SearXNG (optional, enabled by default).
- **`GET /images/{name}`** on the A-PROX server → serves generated PNGs from `[image_generation].serve_dir`.
- **`GET /files/{name}`** on the A-PROX server → serves generated text files from `[file_generation].serve_dir` (`data/generated_files`).

## File Write (`write_file` tool)
Arrived after the image pipeline; a file request (latest user turn matched by `FILE_KEYWORDS` or a verb+file-cue signal in `src/filegen/mod.rs::is_file_request`) routes into the same `AgenticToolLoop` with the `write_file` tool armed (`tools/registry.rs::write_file_definition`):
- `prepare_file_request` merges `FILE_DIRECTIVE` into the **first/only** system message (same single-system-message rule as images — never insert a second leading `system`).
- Execution: `dispatch_tool_execution` in routes.rs routes **write_file → image_generate → registry** at all three dispatch sites (non-stream loop, streaming loop, finalize). `maybe_execute_write_file` cleans the filename (`clean_filename`), checks the deny list (`is_denied_extension`; config `[file_generation] deny_exts`, defaults in code), caps `content` (`max_content_chars`, warns and instructs to split), then saves via `FileStore`. Id = `file_<epoch_ms>_<counter>`, stored/served name = `{id}_{clean}`.
- `mode=append` **reuses** the active file id when the filename matches (`FileGenContext.active`); `mode=overwrite` (default) always starts a fresh id. See `maybe_execute_write_file`.
- Wire contract mirrors images: ONE `delta.file_url` SSE event `{"url","name","mime"}` emitted before any text (`file_url_sse_payload`), non-streaming `file_url` on the outer JSON — always alongside `image_url` (both may appear: image first, then file). **No llama stop/restart and no permit held** for files (nothing blocks the backend).
- Context hygiene: after a successful `write_file` the `content` argument is redacted from the assistant `tool_calls` stored in history (`sanitize_tool_calls_for_history`), so chunked bodies don't bloat context; failed calls keep content so the model can retry. `try_parse_write_file_json` is the bare-JSON fallback (checks `filename`+`content` keys).
- Config: `[file_generation]` `enabled`, `serve_dir`, `public_base_url`, `inline_data_url`, `max_content_chars`, `deny_exts`.
- Artifact URL resolution (`artifact_base_url` in routes.rs, solved once per request in `chat_completions`): config `public_base_url`; else `X-Forwarded-Proto`+`X-Forwarded-Host` (first values, proxy TLS); else request `Host` header (the address the client actually dialed); else `http://127.0.0.1:{server.port}`. The base travels on `ImageGenContext`/`FileGenContext` → `GenerateRequest.public_base` → `image_served_url`/`file_served_url`. This makes `image_url`/`file_url` reachable for LAN/WAN/reverse-proxy clients.
- **LLM context never carries base64 — data-URLs are client-only.** `GeneratedImage.public_url` / `GeneratedFile.public_url` are ALWAYS small served URLs; tool-result JSONs (`result_json`/`file_result_json`) and the caption message use those, so a multi-MB base64 blob never enters the KV cache (a 2.5 MB PNG → ~2.7M tokens would exceed a 262144-token context). The optional `data_url: Option<String>` field (set only when `[image_generation]/[file_generation].inline_data_url = true`) is emitted to the CLIENT only, via the `delta.image_url`/`delta.file_url` SSE events and the non-streaming `image_url`/`file_url` fields (`image_client_url`/`file_client_url`). The Phase C caption part instead points llama.cpp at A-PROX's own served image (`http://127.0.0.1:{server.port}/images/{id}.png`, built in `inject_generated_image_as_user_msg`), so llama fetches it as a real vision image with bounded tokens. `inline_data_url` requires client-side base64 decoding (`Image.memory`), not `Image.network`.

## Image Generation (read before touching image code)
Canonical reference: `imagegen_implementation_plan.md`. Pipeline (Phase A/B/C) in `src/server/routes.rs` + `src/imagegen/orchestrator.rs`:
- A request whose latest user turn has `image_url` parts (→ i2i) or t2i verb keywords routes into the existing `AgenticToolLoop` with one extra tool `image_generate` armed (defined in `tools/registry.rs::image_generate_definition`, dispatched in routes.rs — see `maybe_execute_image_generate`, NOT the registry arm, which returns an error string).
- Phase A: llama.cpp (vision) rewrites the request per workflow system prompt (`prompts/t-iprompt.txt` / `prompts/i-iprompt.txt` + `HARNESS_DIRECTIVE`). `try_parse_image_generate_json` is the bare-JSON fallback when the model skips a structured tool call.
- Phase B: `ImageGenService::generate` — SIGTERM llama (≤30s, SIGKILL fallback) → boot ComfyUI only for this job (`ensure_running`) → upload ref image (i2i) → inject `workflows/t2i.json`/`i2i.json` (nodes: 1 unet, 2 clip, 5 KSampler random seed ≥0 — ComfyUI rejects -1, 8 TextEncodeQwenImage21, 26 EmptyLatentImage dims; i2i: 11 LoadImage, 32 ResizeImageMaskNode dims) → submit+poll `/history/<id>` (≤180s) → `/view` → save PNG → **fully stop ComfyUI** (SIGTERM/SIGKILL, VRAM released) → **restart llama.cpp unconditionally** (≤ `health_timeout_s`, default 600s for cold boots). Only llama.cpp boots at A-PROX startup (`state.rs`); ComfyUI is never left up between image jobs. The concurrency permit is held the whole time (max 1 image job, chat queued during downtime).
- Phase C: synthetic user message (text + the generated image served from A-PROX's own loopback endpoint as an `image_url` part — see the KV-safety bullet in File Write) → llama streams caption; ONE `delta.image_url` SSE event emitted before text (`image_url_sse_payload`); non-streaming: `image_url` on outer JSON. The **client-facing** `image_url`/`file_url` fields carry the inline base64 data-URL when `[image_generation]/[file_generation].inline_data_url = true` (`image_client_url`/`file_client_url`), else the per-request served URL — the two are independent and never mix.
- Resolution math (`src/imagegen/ratio.rs`): target 2 MP from `wh_ratio`, multiple-of-16, cap 4096; `ratio_follow=<image1>` uses the reference image's own dims; `ResolutionSelector` (node 10) is left dormant.
- **Never emit `content` as a JSON array** in SSE deltas — CLAN `sse_client.dart:362` casts `delta['content']` to String and throws.
- `/health` + `/v1/models` are NOT gated during Phase B (they may error on a down llama); documented behavior.

## Auth (read this before touching auth code)
Key enforcement is **not wired up** in the current code, despite README/API docs claiming otherwise:
- `src/server/middleware.rs` defines an `ApiKey` extractor (401 when no `X-API-Key` / `Authorization: Bearer` header) but **no handler uses it**, and `build_router` (`src/server/mod.rs`) adds no auth layer.
- `/v1/chat/completions` and `/ingestion/*` accept requests with **no key validation at all**.
- Only `/monitor/api` and `/monitor/stream` check a key — and only *if one is provided* (optional auth), compared against `state.config.upstream.api_key` (default `change-me`, also the llama.cpp key). `/monitor/stats` ignores keys entirely.
- There is no `server.api_key` config field and **no `--api-key` CLI flag** (README's CLI table is stale). Changing the "client key" means changing `upstream.api_key`.
- To actually enforce auth, wire the existing `ApiKey` extractor into handlers or add a tower middleware layer in `build_router`.

## Architecture
A-PROX sits between clients and llama.cpp. It routes requests three ways: fast pass-through, RAG-augmented search, or agentic tool loop (max 5 turns). All embeddings, vector search, and scraping run on CPU — zero GPU memory.

Key invariants:
- Context pruner never prunes `system` role messages; always preserves latest `user` message (`src/context/trimmer.rs`).
- RRF (Reciprocal Rank Fusion) scoring uses `k=60` constant in `src/db/vector_store.rs`.
- ONNX fallback: if `models/bge-small-en-v1.5-int8.onnx` is missing, `CpuEmbedder` uses deterministic hash-based heuristic embeddings. Also loads `models/tokenizer.json` for the `FastTokenizer`.
- `sqlite-vec` extension auto-initializes once globally via `rusqlite::ffi::sqlite3_auto_extension` (`vector_store.rs`).
- Max 1 concurrent inference by default (config `guardrails.max_concurrent_inferences`); guardrails module = semaphore + RAM watchdog (`min_free_ram_gb`).

## Model Routing (`src/router/mod.rs`)
Priority order in `classify_request`:
1. Explicit model alias → route (`a-prox-direct|pass|fast` → passthrough; `a-prox-rag|knowledge|docs` → RAG; `a-prox-agent|tools` → agentic loop).
2. `x-proxy-bypass: true` header → FastPassThrough (skips intent).
3. Routing-only slash-command prefixes: `/bypass`/`/direct`/`/pass` → passthrough; `/rag`/`/knowledge`/`/docs` → RAG. All six are config-driven under `[tool_commands]` (`bypass`, `direct`, `pass`, `rag`, `knowledge`, `docs`; defaults as shown, empty string disables). RAG-family commands still require `detect_intent == RAG`.
4. Per-tool slash-command flags (`match_tool_command`, config `[tool_commands]`): `/tools` → AgenticToolLoop with all internal tools; `/search`, `/fetch`, `/ragsearch`, `/ingest`, `/time`, `/image`, `/file` → `AgenticToolForced` (loop restricted to that one tool). `/tools` accepts leading tool-name args (`/tools search fetch`) to arm an exact subset (aliases: `search`, `fetch`, `rag`, `time`, `image`, `file`, …); non-tool text stays as the query. Image/file flags also arm their pipelines (`prepare_forced_image_request`/`prepare_forced_file_request`). `/image` auto-detects an attached `image_url` part → i2i (reference dims), else t2i. All flag strings are config-driven and disable when set to `""`; longest flag wins.
5. Image requests (`is_image_request`) and file-write requests (`is_file_request`) force the agentic loop — both checks run after slash commands but before the intent classifier.
6. Intent classifier (cosine similarity vs example prompts, threshold `0.70` per category) → RAG search / ingest / agentic.
7. Non-empty client-supplied `tools` array → AgenticToolLoop.
8. Default → FastPassThrough.

## Commands
```bash
# Download dependencies (ONNX model + SearXNG)
./scripts/download_model.sh
./scripts/download_searxng.sh

# Build (release profile + target-cpu=znver3 via .cargo/config.toml)
cargo build --release

# Run tests
cargo test

# Start server (rebuilds + downloads deps, then runs with config/default.toml)
./run.sh

# Run directly with CLI overrides — ONLY these flags exist:
# --config --port --host --upstream --searxng-port   (no --api-key!)
./target/release/a-prox --config config/custom.toml --port 9000 --upstream http://127.0.0.1:8080
```

## Config
Default loads `config/default.toml`. Missing/unreadable config file → `AppConfig::default()` (graceful, `src/config.rs`). `--searxng-port` also sets `searxng.listen_port`.

Notable defaults from `config/default.toml`:
- `max_context_tokens = 65536`, `reserve_completion_tokens = 8192`, `sliding_window_turns = 7`
- `max_file_size_kb = 512`, `watch_interval_secs = 30`
- Intent threshold `0.70` across all categories
- `pdf_enabled = true` — PDFs sent upstream to multimodal endpoint
- `db.mmap_size_mb = 16384` — SQLite memory-map uses ~16 GB RAM
- `guardrails.enable_agentic_tools = true` (false disables agentic routing)
- `file_generation.enabled = true` — exposes the `write_file` tool on detected file requests (see File Write section)
- `[tool_commands]` — per-tool `/flag` strings (defaults: `/tools`, `/search`, `/fetch`, `/ragsearch`, `/ingest`, `/time`, `/image`, `/file`) **plus** routing-only commands (`/bypass`, `/direct`, `/pass`, `/rag`, `/knowledge`, `/docs`); a leading configured flag forces `AgenticToolForced`, an empty string disables the flag. `/tools` arms an exact subset with leading tool-name args.

(Note: the `[context]` sample values in `README.md` — 32768/4096/10 — are stale; the committed `default.toml` uses 65536/8192/7.)

## API Endpoints
All routes in `src/server/mod.rs`; auth column reflects the *current* code (see Auth section):

| Route | Method | Auth (current) | Purpose |
|-------|--------|----------------|---------|
| `/health`, `/metrics` | GET | None | Health check |
| `/monitor`, `/monitor/api`, `/monitor/stats` | GET | None | Dashboard UI + stats API |
| `/monitor/stream` | GET | None | SSE broadcast for live monitor |
| `/v1/models` | GET | None | Proxies upstream `/v1/models` |
| `/v1/chat/completions` | POST | **None enforced** | Main chat endpoint (incl. image generation) |
| `/v1/chat/completions/async` | POST | **None enforced** | Submit async request (returns request_id) |
| `/v1/chat/completions/{id}/stream` | GET | **None enforced** | Resume SSE stream for async request |
| `/v1/chat/completions/{id}/result` | GET | **None enforced** | Fetch final cached result |
| `/v1/chat/completions/{id}/status` | GET | **None enforced** | Poll async request status |
| `/v1/chat/completions/{id}` | DELETE | **None enforced** | Cancel async request |
| `/ingestion/reindex` | POST | **None enforced** | Force full directory re-index |
| `/ingestion/status` | GET | **None enforced** | Get indexing stats |
| `/images/{name}` | GET | None | Serve stored generated PNGs |
| `/files/{name}` | GET | None | Serve stored generated text files |

## Key files
- `src/server/routes.rs` — main request handling, agentic loop (`execute_agentic_loop`), image-gen pipeline (Phase A/B/C helpers: `prepare_image_request`, `maybe_execute_image_generate`, `inject_generated_image_as_user_msg`, `image_url_sse_payload`), file-write pipeline (`prepare_file_request`, `maybe_execute_write_file`, `dispatch_tool_execution`, `sanitize_tool_calls_for_history`, `file_url_sse_payload`, `resolve_file_public_url`), artifact URL resolution (`artifact_base_url`, `file_served_url`, `file_data_url`), monitor handlers
- `src/imagegen/` — `orchestrator.rs` (ImageGenService: ComfyUI submit/poll/fetch + llama lifecycle), `ratio.rs` (WxH math), `mod.rs` (intent keywords, prompt loading, bare-JSON parse)
- `src/filegen/mod.rs` — write_file detection (`is_file_request`, keyword + verb/cue signals), `FILE_DIRECTIVE`, `clean_filename`, `is_denied_extension`, bare-JSON fallback (+ inline tests)
- `src/files/` — `store.rs` (FileStore on `[file_generation].serve_dir`: save/append/resolve/read, MIME map) + `GeneratedFile`
- `src/comfy_ui/` — `manager.rs` (ComfyUIManager: ensure-running, submit, poll, view, upload-image, stop; spawns via venv `python` + `workdir`/`args`, stdout/stderr inherited), `workflow.rs` (WorkflowTemplate: node/input injection)
- `src/llama_server/` — `manager.rs` (LlamaServerManager: spawn no-mlock, /health wait ≤ `health_timeout_s`, SIGTERM/SIGKILL stop, stdout/stderr inherited)
- `src/images/` — `store.rs` (ImageStore on `[image_generation].serve_dir`) + serving via `serve_image`
- `src/server/middleware.rs` — `ApiKey` extractor (defined, currently unused)
- `src/server/mod.rs` — `build_router`: route table + server startup
- `src/context/trimmer.rs` — context pruning (sliding window + system preservation)
- `src/context/tokenizer.rs` — `FastTokenizer` with `count_tokens()`
- `src/db/vector_store.rs` — SQLite + sqlite-vec + FTS5 hybrid search (RRF)
- `src/embeddings/onnx_cpu.rs` — ONNX CPU embeddings, mean pooling + L2 norm, heuristic fallback
- `src/guardrails/mod.rs` — concurrency semaphore + RAM watchdog
- `src/tools/registry.rs` — built-in tools: `web_search`, `web_fetch`, `rag_search`, `rag_ingest`, `system_time`; armed per-request schemas: `image_generate_definition`, `write_file_definition`
- `src/monitor/mod.rs` — monitor state, `compute_statistics()`, history propagation
- `src/monitor/dashboard.html` — embedded dashboard UI (`include_str!`)
- `src/rag/mod.rs` — RAG pipeline, chunking: `TextChunker::new(512, 64)`
- `src/router/mod.rs` + `src/router/intent.rs` — routing decision + cosine-similarity intent classifier
- `src/search/` — SearXNG JSON client (`readability`/`scraper` for HTML extraction), mock fallback results
- `src/searxng/` — managed SearXNG install/spawn lifecycle (`manager.rs` uses `install_dir` from config)
- `src/ingestion/` — directory indexing, file watching, PDF processing

## Testing notes
- Main suite is `tests/unit_tests.rs`; the only inline `#[cfg(test)]` modules under `src/` are `src/router/intent.rs`, `src/imagegen/{ratio.rs,tests.rs}`, `src/filegen/mod.rs`, `src/files/store.rs`, and `src/server/routes.rs` (`filegen_tests` for `sanitize_tool_calls_for_history`, `base_url_tests` for `artifact_base_url`/`file_served_url`/`file_data_url`).
- `test_onnx_cpu_embeddings_and_vector_store` creates a temp SQLite DB under `data/` (`data/test_*.db`) and removes it; the test falls back to heuristic embeddings when the ONNX model is missing.
- The agentic loop sends non-streaming requests for intermediate turns, then forwards the final result in the requested streaming mode.
- When modifying the agentic loops in `routes.rs` (two: non-streaming `execute_agentic_loop` and the streaming variant), clone `state.context_mgr` before spawning async blocks to avoid E0521 borrow-escape errors: `let context_mgr = state.context_mgr.clone();` and use the clone inside the `tokio::spawn` closure.

## Common pitfalls
- **Token tracking in monitor**: `tokens_sent` uses `context_mgr.calculate_total_tokens(&pruned_messages)`; `tokens_received` uses `context_mgr.tokenizer().count_tokens(content)`. Never use `chars().count() / 4` — it's inaccurate.
- **Parsing tool results in agentic loop**: llama.cpp response content may be a plain string. Use `.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("")` — don't assume content is a JSON object (see `routes.rs` pattern).
- **SQLite DB location**: `data/a_prox.db` — persists RAG documents. Destroying it loses all indexed documents.
- **SearXNG install location**: `download_searxng.sh` and the manager install to `[searxng] install_dir` (default `~/.local/share/a-prox-searxng`), **not** the repo `searxng/` dir — though `run.sh` probes repo-local `searxng/.venv` first. Requires Python 3.10+ (`python3 -m venv` + pip). Disable via `[searxng] enabled = false`.
- **Don't trust README auth/CLI claims**: there is no enforced API key and no `--api-key` flag (see Auth section).
- **`image_generate` is intercepted in routes.rs, not the registry**: the `registry.rs::execute_tool` arm returns an explanatory error string; real dispatch happens in `maybe_execute_image_generate` (needs permit + `ImageGenContext`). Same for `write_file` → `maybe_execute_write_file` (no permit needed).

## Async Request Queue (A-PROX `async` endpoints)
- **Database**: `async_requests` table in `data/a_prox.db` (schema added via `src/db/schema.rs` INIT_SQL). Fields: `id` (client-provided UUID), `payload` (full request JSON), `status` (`queued`/`processing`/`completed`/`failed`/`cancelled`), `result` (final response JSON), `created_at`/`updated_at`/`expires_at` (Unix seconds), `route_decision`, `tokens_received`, `error`. Indexes on `status` and `expires_at`.
- **Worker**: `AsyncQueueWorker` in `src/async_queue/worker.rs` — background task with semaphore-limited concurrency (`config.async.max_concurrent`, default 1). Pulls `queued` requests, processes via existing routing logic (`execute_agentic_loop`, `forward_to_upstream`, etc.), stores result, updates monitor. Cleanup task runs every `config.async.cleanup_interval_minutes` (default 5m) to delete expired requests.
- **Endpoints** (in `src/async_queue/endpoints.rs`):
  - `POST /v1/chat/completions/async` — idempotent submit (client provides or server generates UUID), returns `{request_id, status: "queued"}`.
  - `GET /v1/chat/completions/{id}/status` — poll status.
  - `GET /v1/chat/completions/{id}/result` — fetch final result (400 if not terminal).
  - `GET /v1/chat/completions/{id}/stream` — SSE stream: if `completed`, replays stored result; if `processing`/`queued`, waits and emits keepalives until complete.
  - `DELETE /v1/chat/completions/{id}` — cancel (only non-terminal).
- **Config** (`[async]` in `config/default.toml`): `enabled` (default true), `cache_ttl_hours` (default 1), `max_concurrent` (default 1), `cleanup_interval_minutes` (default 5).
- **CLAN-AI integration**: CLAN-AI submits via `POST /async`, stores `PendingRequest` locally, resumes via `GET /:id/stream` on app lifecycle `resumed`/`inactive`. Thread edit/delete during background processing cancels old request, submits new at queue end.