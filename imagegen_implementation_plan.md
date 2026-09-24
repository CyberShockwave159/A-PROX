Plan
1. Per-request base URL resolution (src/server/routes.rs)
New helper:
fn artifact_base_url(cfg_override: &str, headers: &HeaderMap, server_port: u16, inline: bool) -> String
Resolution order for the emitted artifact URL:
- inline_data_url: true → return a base64 data: URI (built by the caller — see below), supersedes everything.
- else config public_base_url (existing override, highest priority) → trim trailing /.
- else X-Forwarded-Proto (first of comma-list, default http) + X-Forwarded-Host (first value) — covers TLS reverse proxies.
- else Host header — the address the client actually dialed (LAN IP, public IP, proxy domain).
- else fallback http://127.0.0.1:{server.port} (preserves today's loopback behavior for direct local use).
- Reject empty / whitespace-only values.
2. Plumb the base through existing contexts (routes.rs)
- Add public_base: String to ImageGenContext (line 678) and FileGenContext (line 692).
- Compute the resolved base once in chat_completions (headers already in scope, line 239); pass it into prepare_image_request, prepare_forced_image_request, prepare_file_request, prepare_forced_file_request (one added &str param each).
- Contexts already travel into both the non-streaming loop and the streaming tokio::spawn task, so all three tool-dispatch sites (routes.rs:1382/1428/1474 and 1766/1809/1878) get it for free.
3. Images (orchestrator.rs + routes.rs)
- Add public_base: String to GenerateRequest (orchestrator.rs:40).
- maybe_execute_image_generate fills it from image_ctx.public_base (routes.rs:972).
- generate sets public_url = data-URL when img_cfg.inline_data_url (reuse to_data_url(&bytes), already pub in src/imagegen/mod.rs:168), else format!("{base}/images/{id}.png", ...).
- GeneratedImage.public_url is consumed by every emission site unchanged: SSE delta.image_url (3 sites), non-streaming image_url (routes.rs:1630), tool-result JSON (routes.rs:1004) — data-URL mode works uniformly.
4. Files (routes.rs)
- maybe_execute_write_file builds public_url = data:{mime};base64,{B64::encode(content_bytes)} when file_generation.inline_data_url, else format!("{base}/files/{servable_name}") using fctx.public_base. Same config-priority logic. SSE delta.file_url + non-streaming file_url consume it unchanged.
5. Config (src/config.rs + config/default.toml)
- ImageGenerationConfig.inline_data_url: bool = false
- FileGenerationConfig.inline_data_url: bool = false
- Wire through Default impls + config/default.toml with short comments.
6. Tests
- Unit tests for artifact_base_url: forwarded proto+host, X-Forwarded-Host precedence over Host, plain Host, missing headers → loopback fallback, whitespace rejection, config override wins. (Inline in routes.rs under an existing test module or a small new one.)
- Extend the existing inline data-URL helpers tests to cover data: prefix emission for image/file artifacts.
7. Docs
- AGENTS.md, README.md, imagegen_implementation_plan.md: document the new resolution order, the two new config flags, and the note that data-URL mode requires the CLAN client to decode base64 (Image.memory) instead of Image.network — plus the infra requirement that /images/* + /files/* must be reachable (firewall / NAT / reverse proxy) for URL mode.
Notes / tradeoffs
- No wire-contract change in URL mode; data-URL mode keeps the same event shape but embeds a large base64 body (~1.33× PNG size, potentially multi-MB) — opt-in, with a client-side decode requirement.
- serve_image/serve_file endpoints are unchanged and already host-agnostic.
- A reverse proxy must also route /images/* and /files/* to A-PROX for URL mode behind a domain.