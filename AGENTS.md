# A-PROX Agent Instructions

## Port Map
- **8000** → A-PROX (public).
- **8080** → llama.cpp upstream (localhost only). Must be started with `--api-key` matching `upstream.api_key`; A-PROX sends that key as `Authorization: Bearer` on every upstream call (e.g. `routes.rs:63`).
- **8888** → Managed SearXNG (optional, enabled by default).

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
3. Slash-command prefixes in latest user message: `/bypass`, `/direct`, `/pass` → passthrough; `/rag`, `/knowledge`, `/docs` → RAG.
4. Intent classifier (cosine similarity vs example prompts, threshold `0.70` per category) → RAG search / ingest / agentic.
5. Non-empty client-supplied `tools` array → AgenticToolLoop.
6. Default → FastPassThrough.

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

(Note: the `[context]` sample values in `README.md` — 32768/4096/10 — are stale; the committed `default.toml` uses 65536/8192/7.)

## API Endpoints
All routes in `src/server/mod.rs`; auth column reflects the *current* code (see Auth section):

| Route | Method | Auth (current) | Purpose |
|-------|--------|----------------|---------|
| `/health`, `/metrics` | GET | None | Health check |
| `/monitor`, `/monitor/api`, `/monitor/stats` | GET | None | Dashboard UI + stats API |
| `/monitor/stream` | GET | None | SSE broadcast for live monitor |
| `/v1/models` | GET | None | Proxies upstream `/v1/models` |
| `/v1/chat/completions` | POST | **None enforced** | Main chat endpoint |
| `/ingestion/reindex` | POST | **None enforced** | Force full directory re-index |
| `/ingestion/status` | GET | **None enforced** | Get indexing stats |

## Key files
- `src/server/routes.rs` — main request handling, agentic loop (`execute_agentic_loop`), monitor handlers
- `src/server/middleware.rs` — `ApiKey` extractor (defined, currently unused)
- `src/server/mod.rs` — `build_router`: route table + server startup
- `src/context/trimmer.rs` — context pruning (sliding window + system preservation)
- `src/context/tokenizer.rs` — `FastTokenizer` with `count_tokens()`
- `src/db/vector_store.rs` — SQLite + sqlite-vec + FTS5 hybrid search (RRF)
- `src/embeddings/onnx_cpu.rs` — ONNX CPU embeddings, mean pooling + L2 norm, heuristic fallback
- `src/guardrails/mod.rs` — concurrency semaphore + RAM watchdog
- `src/tools/registry.rs` — built-in tools: `web_search`, `web_fetch`, `rag_search`, `rag_ingest`, `system_time`
- `src/monitor/mod.rs` — monitor state, `compute_statistics()`, history propagation
- `src/monitor/dashboard.html` — embedded dashboard UI (`include_str!`)
- `src/rag/mod.rs` — RAG pipeline, chunking: `TextChunker::new(512, 64)`
- `src/router/mod.rs` + `src/router/intent.rs` — routing decision + cosine-similarity intent classifier
- `src/search/` — SearXNG JSON client (`readability`/`scraper` for HTML extraction), mock fallback results
- `src/searxng/` — managed SearXNG install/spawn lifecycle (`manager.rs` uses `install_dir` from config)
- `src/ingestion/` — directory indexing, file watching, PDF processing

## Testing notes
- Main suite is `tests/unit_tests.rs`; the only inline `#[cfg(test)]` module under `src/` is `src/router/intent.rs`.
- `test_onnx_cpu_embeddings_and_vector_store` creates a temp SQLite DB under `data/` (`data/test_*.db`) and removes it; the test falls back to heuristic embeddings when the ONNX model is missing.
- The agentic loop sends non-streaming requests for intermediate turns, then forwards the final result in the requested streaming mode.
- When modifying the agentic loops in `routes.rs` (two: non-streaming `execute_agentic_loop` ~line 587 and the streaming variant ~line 832), clone `state.context_mgr` before spawning async blocks to avoid E0521 borrow-escape errors: `let context_mgr = state.context_mgr.clone();` and use the clone inside the `tokio::spawn` closure.

## Common pitfalls
- **Token tracking in monitor**: `tokens_sent` uses `context_mgr.calculate_total_tokens(&pruned_messages)`; `tokens_received` uses `context_mgr.tokenizer().count_tokens(content)`. Never use `chars().count() / 4` — it's inaccurate.
- **Parsing tool results in agentic loop**: llama.cpp response content may be a plain string. Use `.and_then(|m| m.get("content")).and_then(|v| v.as_str()).unwrap_or("")` — don't assume content is a JSON object (see `routes.rs:651` pattern).
- **SQLite DB location**: `data/a_prox.db` — persists RAG documents. Destroying it loses all indexed documents.
- **SearXNG install location**: `download_searxng.sh` and the manager install to `[searxng] install_dir` (default `~/.local/share/a-prox-searxng`), **not** the repo `searxng/` dir — though `run.sh` probes repo-local `searxng/.venv` first. Requires Python 3.10+ (`python3 -m venv` + pip). Disable via `[searxng] enabled = false`.
- **Don't trust README auth/CLI claims**: there is no enforced API key and no `--api-key` flag (see Auth section).