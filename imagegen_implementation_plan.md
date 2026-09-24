# Implementation Plan: Dual-Workflow Image Generation in A-PROX

Status: **approved** (plan mode) — implementation in progress.
> This file is the canonical reference for the image-generation feature. Re-read it
> after any context compaction before touching image-gen code.

---

## 1. Objective

Enable CLAN-AI2 users to request images from chat. A-PROX:
1. Detects an image-request intent (like tool calls) and routes into the existing `AgenticToolLoop`.
2. Uses llama.cpp (running, with vision) to rewrite the user request into a fully detailed image prompt, governed by a per-workflow system prompt (`t-iprompt.txt` for text-to-image, `i-iprompt.txt` for image-edit).
3. Stops llama.cpp, runs a ComfyUI Qwen-Image workflow (GGUF, CPU-friendly, no GPU reserved for embeddings/scraping), restarts llama.cpp with vision, and returns the generated image + llama.cpp's message stream to the client.

Two distinct ComfyUI workflows:
- **text-to-image (t2i)** — pure text prompt → new image.
- **image-to-image / edit (i2i)** — user uploads a reference image → edited/new version.

Each workflow has its own system prompt, applied when llama.cpp writes the image prompt.

## 2. Locked Decisions (do not reopen)

| # | Decision | Consequence |
|---|----------|-------------|
| 1 | A-PROX owns llama.cpp as a child process, launched **without** `--load-mode mlock`. | RAM/swap behavior acceptable; see Risks. |
| 2 | Reuse `AgenticToolLoop` + one `image_generate` tool. No new `RouteDecision` variant. | New `IntentCategory::ImageGeneration` routes into `AgenticToolLoop`. |
| 3 | ComfyUI stays running/loaded after a job (models auto-unload to ~0.4 GB idle). No process shutdown between requests. | Fallback only if testing shows models don't unload: fully stop ComfyUI. |
| 4 | Client (CLAN-AI2) proper image field is user's work. Wire contract = **`delta.image_url` only** (see §4). | A-PROX never sends `content` as a JSON array (client cast would throw). |
| 5 | ~10 min total latency accepted. Permit-held requests queue; `/health` + `/v1/models` 503 during llama downtime (documented). | Do not mark request failed during llama downtime. |
| 6 | A-PROX appends a short **harness directive** after the verbatim system prompts: "Emit your JSON as the arguments of a call to `image_generate`." | Keeps user's prompt files verbatim; bare-JSON fallback still parsed if model disobeys. |
| 7 | Negative prompt comes from config `[image_generation].default_negative_prompt` (override), not the workflow template string. | |
| 8 | GGUF = `qwen-image-Uncensored-2.1-Q4_K_M.gguf` (as in the user's workflows). | |

## 3. Architecture

```
client ──POST /v1/chat/completions (stream)──▶ A-PROX
                                              │
          Router.intent.ImageGeneration        │
          (message has image_url parts?)       │
           YES→ i2i workflow + i-iprompt       │
           NO → t2i workflow + t-iprompt       │
                                              ▼
   AgenticToolLoop (max 5 turns, permit held from routes.rs:252)
     turn 1: llama.cpp (vision) rewrites prompt per system prompt +
             harness directive  → tool call image_generate{rewritten_prompt,
             wh_ratio, ratio_follow, negative_prompt}
             (fallback: bare single-line JSON parsed from content)
     tool:   image_generate → ImageGenService::generate
              │  1. ensure ComfyUI up (:8188)
              │  2. SIGTERM llama.cpp (libc::kill), wait ≤30s, SIGKILL fallback
              │  3. i2i only: POST /upload/image, set LoadImage node filename
              │  4. inject workflow template (prompt, negative, random seed ≥ 0 (KSampler rejects -1),
              │     WxH from ratio math / ratio_follow dims)
              │  5. POST /prompt; poll /history/{id} (2s, ≤180s)
              │  6. GET /view → PNG → save data/generated_images/gen_<id>.png
              │  7. relaunch llama.cpp (no mlock), wait /health (≤ `health_timeout_s`, default 600s)
              ▼
   final turn: A-PROX injects synthetic user message (text + base64 data URL
               of generated image) → llama.cpp (vision) streams caption
   stream out: one `delta.image_url` SSE event, then text, then [DONE]
```

### Workflow selection
The choice (t2i vs i2i) is made in `routes.rs` **before** the loop's first LLM turn, by checking whether the latest user message has `image_url` content parts. The model never chooses the workflow.

## 4. Wire Contract (CLAN-AI2)

Exactly **one** SSE event, emitted after the role chunk and before streamed text:

```json
data: {"choices":[{"delta":{"image_url":{"url":"http://<host>:8000/images/gen_<id>.png"}}}]}
```

Rules:
- Never emit `content` as a JSON array — CLAN `sse_client.dart:362` does `delta?['content'] as String? ?? ''`, which throws on an array.
- URL host = `[image_generation].public_base_url` config, else incoming `Host` header.
- Non-streaming path: same field on the outer response JSON.
- Files served by A-PROX route `GET /images/{name}` (Map-read + `Content-Type` from extension).

### Client-side work delta (user's, for reference)
On `StreamChunk` (`lib/core/network/sse_client.dart:43`):
- add `final String? imageUrl;`
- read `delta['image_url']` in `_processDataBlock` (`:355`); can be string or `{url}`.
- **thread through all ~10 `filterReasoning` re-creates** (`:170-315`) or it drops when reasoning-strip is on.
Consume in `stream_mutation_mixin.dart` (`:99`) mirroring the `pendingReasoningBuffer` pattern → `ChatMessage.imageUrl` (`chat_message.dart:52`) via `copyWith`/`toMap`/`fromMap` → `local_storage.dart` migration (`image_url TEXT`, copy the `image_path` ALTER pattern `:202-206`) → `message_bubble.dart` assistant branch (`Image.network`).

## 5. Config (`config/default.toml`)

```toml
[llama_server]
enabled = true
executable = "/home/jstanton/llama.cpp/build/bin/llama-server"
model_path = "/run/media/jstanton/MNT B (HDD 2T)/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
mmproj_path = "/run/media/jstanton/MNT B (HDD 2T)/mmproj-Qwen3.6-35B-A3B-BF16.gguf"
host = "127.0.0.1"
port = 8080
api_key = "<upstream.api_key>"     # keep in sync: A-PROX sends this key upstream
ctx_size = 0
no_mmproj_offload = true
flash_attn = "on"
n_cpu_moe = 34
cache_type_k = "q4_0"
cache_type_v = "q4_0"
reasoning_preserve = true
kv_unified = true
threads = 6
batch_size = 4096
ubatch_size = 1024
image_min_tokens = 1024
image_max_tokens = 2048
cors_origins = "http://0.0.0.0:8082"
load_mode = "none"                 # was --load-mode mlock; dropped per decision #1
health_timeout_s = 600             # cold boots (HDD model load) can take 5-8 min
stop_grace_s = 30

[comfy_ui]
enabled = true
url = "http://127.0.0.1:8188"
workdir = "/run/media/jstanton/MNT G (SSD 1T)/ComfyUI"
python = "/run/media/jstanton/MNT G (SSD 1T)/ComfyUI/bin/python"
args = ["main.py", "--enable-manager"]
health_timeout_s = 600             # eager boot at A-PROX startup, like llama.cpp
poll_interval_ms = 2000
generation_timeout_s = 180

[image_generation]
enabled = true
mp = 2.0                     # target megapixels
multiple = 16
max_side = 4096
unet = "qwen-image-Uncensored-2.1-Q4_K_M.gguf"
clip = "qwen3vl_8b_int8_convrot.safetensors"
vae = "qwen_image_2.1_vae_bf16.safetensors"
t2i_workflow = "workflows/t2i.json"
i2i_workflow = "workflows/i2i.json"
t2i_prompt_file = "prompts/t-iprompt.txt"
i2i_prompt_file = "prompts/i-iprompt.txt"
default_negative_prompt = "lowres, blurry, ..."
serve_dir = "data/generated_images"
public_base_url = ""         # fallback: request Host header

[intent.categories.image_generation]
threshold = 0.70
# [[intent.categories.image_generation.examples]] ... (add)
```

### CLI
No new CLI flags. All image config via TOML (`main.rs:9-24` unchanged).

## 6. New Files / Modules (A-PROX)

```
src/llama_server/{mod.rs,manager.rs}    # LlamaServerManager: spawn (no mlock), /health wait, SIGTERM, RAII restart
src/comfy_ui/{mod.rs,manager.rs,workflow.rs}  # ComfyUIManager: ensure-running, submit/poll/fetch, /upload/image; WorkflowTemplate injection
src/imagegen/{mod.rs,orchestrator.rs,ratio.rs} # ImageGenService; compute WxH; workflow selection helpers
src/images/{mod.rs,store.rs}            # ImageStore: persist + serve GET /images/{name}
workflows/t2i.json                      # copy of "Custom Qwen Image 2.1 (GGUF) Text-to-Image.json"
workflows/i2i.json                      # copy of "Custom Qwen Image 2.1 (GGUF) Image-to-Image.json"
prompts/t-iprompt.txt                   # copy of repo-root t-iprompt.txt
prompts/i-iprompt.txt                   # copy of repo-root i-iprompt.txt
```

New Cargo deps (`Cargo.toml:8-53`):
- `libc = "0.2"`              (SIGTERM; `Child::kill()` is SIGKILL)
- `image = { version = "0.25", default-features = false, features = ["png", "jpeg", "webp"] }`  (read input dims for `ratio_follow`)
- `reqwest`: add feature `"multipart"` (ComfyUI `/upload/image`)

Registered in:
- `AppConfig` (`config.rs:4-20`): `llama_server`, `comfy_ui`, `image_generation`
- `AppState` (`state.rs:17-32`): `llama_server_manager`, `comfy_ui_manager`, `image_service` — constructed around the `searxng_manager` site (`state.rs:88-92`); thread `image_service` into `ToolRegistry::new` (`state.rs:69-72`)
- `main.rs`: no change
- `server/mod.rs::build_router`: add `GET /images/{name}`

## 7. Integration Points (exact, current code)

| Step | Site | Detail |
|------|------|--------|
| Intent category | `intent.rs:13-19` + `label()` `:22-28` | `IntentCategory::ImageGeneration` |
| Keyword lists | `intent.rs:216-268` | "generate an image", "create a picture of", "draw/render ...", etc. |
| Config category | `config.rs:156-204`, defaults `207-249`, `default_intent_threshold` `251-253` | `[intent.categories.image_generation]` |
| Centroid wiring | `intent.rs:96-121` + `route_by_intent` `router/mod.rs:128-133` | map to `RouteIntent::AgenticTool` (reuse loop) |
| Tool schema | `registry.rs:149` (inside `json!([...])`) | add `image_generate`; auto-propagates to `routes.rs:377`/`647` |
| Tool dispatch | `registry.rs:156` | `match call.name` → arm calling `image_service` (constructor gains the service) |
| Arm tool in loop | non-stream `routes.rs:742-743`, finalize `:826`, stream `:1006-1008` | branch `tool_call.name == "image_generate"`; orchestrate BEFORE pushing tool result; result String = `{status, image_url, image_path, prompt}` |
| Bare-JSON fallback | `routes.rs:722` (no-tool-call branch) | if tool armed & `raw_content` is single-line JSON w/ `rewritten_prompt` → treat as the tool action |
| System prompt selection | pre-loop near `routes.rs:252` | pick t2i vs i2i from latest user message's `image_url` parts; inject system prompt + harness directive |
| Final vision user msg | before `final_payload["messages"]` at `:778` (non-stream) / `:1040` (stream) | `ChatMessage { role:"user", content: Some(json!([{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}])) }` |
| Image chunk emission | after `role_chunk` (`routes.rs:964`), before flushing `buffered` (`:1169`); also in forced non-stream path `:1115-1163` | single SSE event with `delta.image_url`; non-stream: field on outer JSON |
| Serve route | `server/mod.rs::build_router` | `GET /images/{name}` |
| Monitor | `routes.rs` `monitor_api` | route label/tool reflects `image_generate`; add llama-downtime + generate-time metrics |

Loop facts to respect:
- `max_turns = 5` (`routes.rs:651`), loop at `:661` (non-stream) / `:878` (stream). Expected image sequence uses 2–3 turns.
- `execute_tool` returns `String` (`registry.rs:153-156`); errors are in-band strings, pushed as `tool` role messages.
- Permit acquired `routes.rs:252`, held for whole loop (`:655`/`:868`).
- Upstream timeout = global client timeout `config.upstream.timeout_seconds` (180s) — Phase-A prompt write fits.
- Token accounting already prices `image_url` parts at fixed 576 tokens (`trimmer.rs:62-75`) — the ~13MB base64 body does not skew `calculate_total_tokens`.
- When modifying the two loops, `let context_mgr = state.context_mgr.clone();` before `tokio::spawn` (AGENTS.md).

## 8. ComfyUI Workflow Templates (verified 2026-09-22 live)

Both templates validate against live `object_info` (ComfyUI 0.37.0. Wrong inputs = `vae` and `device` exist only as OPTIONAL — OK).

### t2i node map
| Node | Class | Notes |
|------|-------|-------|
| 1 | `UnetLoaderGGUF` | `unet_name = qwen-image-Uncensored-2.1-Q4_K_M.gguf` |
| 2 | `CLIPLoader` | `clip_name = qwen3vl_8b_int8_convrot.safetensors`, `type = qwen_image`, `device = default` |
| 5 | `KSampler` | steps 20, cfg 1.5, euler, sgm_uniform, denoise 1, **seed → random ≥ 0 (KSampler rejects -1)** |
| 6 | `VAELoader` | `qwen_image_2.1_vae_bf16.safetensors` |
| 7 | `VAEDecode` | |
| 8 | `TextEncodeQwenImage21` | prompt, negative_prompt, resolution=0, clip, vae → [positive, negative, latent] |
| 9 | `SaveImageAdvanced` | `filename_prefix = gen_<id>`, png 8-bit sRGB |
| 10 | `ResolutionSelector` | aspect_ratio/megapixels/multiple — **LEFT DORMANT** (see §9) |
| 26 | `EmptyLatentImage` | width/height (direct ints), batch 1 |

### i2i node map (deltas)
| Node | Class | Notes |
|------|-------|-------|
| 11 | `LoadImage` | `image = <uploaded filename>` (A-PROX writes via `/upload/image`) |
| 32 | `ResizeImageMaskNode` | resize_type "scale dimensions", width/height (direct ints), crop center, scale_method area |
| 8 | `TextEncodeQwenImage21` | also `images.image_1 = [32,0]` (reference latent) |
| 5 | `KSampler` | `latent_image = [8,2]` (ReferenceImage latent from node 8) |

### Assets (confirmed on disk)
- `/run/media/.../ComfyUI/models/diffusion_models/qwen-image-Uncensored-2.1-Q4_K_M.gguf`
- `models/text_encoders/qwen3vl_8b_int8_convrot.safetensors`
- `models/vae/qwen_image_2.1_vae_bf16.safetensors`
- GGUF loader reads from `models/diffusion_models/` (ComfyUI-GGUF).

## 9. Resolution (ratio) Math — `imagegen/ratio.rs`

A-PROX computes WxH itself; `ResolutionSelector` (node 10) is left in the graph but disconnected (orphan nodes are fine; not referenced → not executed).

- `wh_ratio` = `"W:H"` (e.g. `"3:2"`):
  - `area = mp * 1_000_000`; `w = sqrt(area * W / H)`; `h = w * H / W`.
- `ratio_follow = "<image1>"` (i2i): use the uploaded image's actual pixel dims.
- Final: round both to nearest `multiple` (16); cap at `max_side` (4096).
- Feed as direct INTs into node 26 (`EmptyLatentImage`) for t2i; node 32 `resize_type.width/height` for i2i.

The LLM may emit ratios outside `ResolutionSelector`'s 8 options (e.g. `1:2`, `2:1`, `18:39`) — irrelevant now since WxH is computed directly.

## 10. ComfyUI Interaction Details

- **ensure-running**: `GET {url}/system_stats` (or `/object_info`) → if down, spawn `python main.py --enable-manager` from `workdir`, poll health ≤ `health_timeout_s`.
- **upload**: `POST {url}/upload/image` multipart field `image` → returns `{name}`; set node 11 `image`.
- **submit**: `POST {url}/prompt` body `{prompt: {nodes…}, client_id}` → `{prompt_id, number}`.
- **poll**: `GET {url}/history/{prompt_id}` every `poll_interval_ms` ≤ `generation_timeout_s`; read outputs of node 9 → `[{filename, subfolder, type:"output"}]`.
- **fetch**: `GET {url}/view?filename=…&subfolder=…&type=output` → PNG bytes → `ImageStore::save`.
- **timeouts/client**: use a dedicated reqwest client (not `state.http_client`, which has 180s global) — no per-request timeout on ComfyUI calls; each poll call individually bounded.
- On failure at any step: RAII guard / `Drop` triggers llama restart.

## 11. llama_server Manager

Mirror `src/searxng/manager.rs` pattern (struct behind `Mutex<Option<Child>>`, `start`/`is_running`/`stop`, Drop → stop):
- `start`: assemble args from `[llama_server]` EXACTLY as today's manual command minus `--load-mode mlock`; spawn with stdin null, stdout/stderr inherited (forwarded to the A-PROX console); poll `GET {host}:{port}/health` ≤ `health_timeout_s`; premature exit → Err.
- `stop`: SIGTERM via `libc::kill(pid, SIGTERM)`, wait ≤ `stop_grace_s` (100ms poll), SIGKILL fallback (`Child::kill`), clear slot.
- `is_up`: health probe.
- Only one manager instance; constructed in `AppState::new`; started iff `llama_server.enabled` (default true once A-PROX owns it — user stops manual launching).

## 12. Pipeline Phases

**Phase A — prompt fleshing (llama up, vision):**
- Inject chosen system prompt (verbatim file contents, config-path referenced with `include_str!` fallback) **+ harness directive** appended.
- Model → `image_generate` tool call (or bare JSON fallback at `routes.rs:722`).
- Args parsed by `ToolRegistry::execute_tool` arm; validate `rewritten_prompt` non-empty; re-prompt once if too sparse.

**Phase B — orchestration (`ImageGenService::generate`):**
1. ensure ComfyUI up
2. SIGTERM llama; wait ≤30s
3. i2i: `/upload/image` → node 11 filename
4. bake template: `prompt`, `negative_prompt` (config), random seed ≥ 0 (KSampler rejects -1), WxH (§9), `filename_prefix = gen_<id>`
5. submit + poll history (≤180s) + fetch PNG
6. store → `GET /images/gen_<id>.png`
7. relaunch llama; wait /health (≤ `health_timeout_s`, default 600s)
8. return `{status:"ok", image_url, image_path, prompt}`
- Performed BEFORE `execute_tool` returns so the tool result carries `image_url`; RAII guarantees llama restart on error.

**Phase C — vision final pass:**
- Inject synthetic user message (text + `data:image/png;base64,…`) before final payload.
- Stream final synthesis; emit the single `delta.image_url` event (§4), then text; `[DONE]`.
- Non-streaming path: image_url in outer JSON.

## 13. Tests

- `imagegen/ratio.rs` unit tests: ratio math, /16 rounding, 4096 cap, `ratio_follow` dims.
- ComfyUI workflow-template validator: diff every `class_type` + input key against live `object_info` (dev script; guards future ComfyUI/GGUF drift).
- Orchestrator test with mock ComfyUI (axum test server for `/history`, `/view`, `/upload/image`).
- Manual E2E against real ComfyUI + llama (see §14).

## 14. Manual E2E Acceptance Checklist

1. `cargo test` passes.
2. `cargo build --release` builds.
3. Start A-PROX alone (llama + ComfyUI unmanaged) → managers spawn both; `/health` OK.
4. `POST /v1/chat/completions` stream: "Generate an image of a cyberpunk cat at sunset" → observe: llama up → llama stops → ComfyUI job runs → llama restarts → text + `delta.image_url` event → `[DONE]`. Image reachable at `GET /images/gen_<id>.png`.
5. i2i: send request with an attached image → i2i workflow used; output respects reference composition.
6. `/monitor` shows route label and `image_generate` tool call; timings recorded.
7. During Phase B, `/health` returns 503, then recovers.

## 15. Risks (accepted)

- **~10 min chat backend down** per image request (permit held). `/health` + `/v1/models` 503 during llama downtime → document in README.
- **No-mlock llama** may swap/evict pages on RAM pressure → accepted (decision #1).
- **Large base64 body** (~13 MB) in the final vision pass → tokenizer prices at fixed 576 (`trimmer.rs:62-75`), accounting stays clean.
- **Workflow drift** on ComfyUI/GGUF updates → §13 validator.
- **`execute_tool` errors are in-band strings** → orchestration errors surface as tool-result text; loop continues (acceptable UX: assistant explains failure).

## 16. Docs to Update

- `AGENTS.md`: port map (ComfyUI 8188, images endpoint), `[llama_server]` ownership (manual launch stopped), mlock removal, `cors_origins` kept, image routing notes, new pitfalls.
- `README.md`: new config sections, `GET /images` endpoint, wire contract, 503-during-generation note.

## 17. Source Assets (read-only, do not regenerate)

- `implementation_plan.md` — STALE (prior, completed think-block work). Leave as-is.
- `Custom Qwen Image 2.1 (GGUF) Text-to-Image.json` / `... Image-to-Image.json` — authoring source → copy to `workflows/`.
- `t-iprompt.txt` / `i-iprompt.txt` — authoring source → copy to `prompts/`.
- `/home/jstanton/CLAN-AI2` — client; user implements the `delta.image_url` field (§4).