# Cached Messages / Background Request Completion Implementation Plan

## Overview
Allow CLAN-AI clients to submit requests that A-PROX manages asynchronously. If the user navigates away, A-PROX continues processing and caches the result for up to 1 hour. When the client returns, it fetches the cached response.

## Architecture

### A-PROX Server (Rust/Axum)
**New Endpoints:**
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/chat/completions/async` | Submit request, return `{request_id, status: "queued"}` immediately |
| `GET` | `/v1/chat/completions/{request_id}/stream` | SSE stream for the request (new or reconnection) |
| `GET` | `/v1/chat/completions/{request_id}/result` | Get final cached result (non-streaming JSON) |
| `GET` | `/v1/chat/completions/{request_id}/status` | Poll status: `queued` \| `processing` \| `completed` \| `failed` |
| `DELETE` | `/v1/chat/completions/{request_id}` | Cancel & cleanup |

**Server-side Components:**
- Request queue with SQLite persistence (survives restart)
- Background worker that processes queued requests against llama.cpp
- Response cache with 1-hour TTL (configurable)
- Idempotent request submission (client provides UUID, can retry safely)

### CLAN-AI Client (Flutter/Dart)
**New Models & Persistence:**
- `PendingRequest` model: `requestId`, `threadId`, `assistantMessageId`, `payload`, `createdAt`, `status` (`pending`/`streaming`/`completed`/`failed`)
- SQLite table `pending_requests` (survives app restart/crash)

**Modified Flow:**
1. `ChatViewModel.sendMessage()` → `StreamMutationMixin.doStreamResponse()`
2. Check DB for existing `PendingRequest` for this `assistantMessageId`
3. If exists → resume via `GET /{id}/stream`
4. If not → submit async via `POST /async`, store `PendingRequest`, then stream
5. On completion → mark `PendingRequest` completed, cache result locally
6. On cancellation → delete `PendingRequest`

**App Lifecycle & Reconnection:**
- `WidgetsBindingObserver.didChangeAppLifecycleState` in `_ClanAiAppState`
- On `resumed`/`inactive` → scan `PendingRequest` table for incomplete requests
- For each: re-establish stream via `GET /{id}/stream`
- UI: show "Reconnecting..." indicator during resume

**Settings:**
- Toggle: "Background request completion" (default: on)
- Config: Cache TTL (default: 1 hour)

**Conflict Resolution:**
- If user edits/deletes the active thread while a background request is pending:
  - Cancel the background request via `DELETE /{id}`
  - Submit new request (goes to end of queue if queue exists)

---

## Implementation Phases

### Phase 1: A-PROX Server - Core Infrastructure
1. **Database schema** - Add `async_requests` table with: `id` (TEXT PRIMARY KEY), `payload` (TEXT), `status` (TEXT), `result` (TEXT), `created_at` (INTEGER), `updated_at` (INTEGER), `expires_at` (INTEGER), `route_decision` (TEXT), `tokens_received` (INTEGER), `error` (TEXT)
2. **Request queue worker** - Background task that pulls `queued` requests, processes via existing `execute_agentic_loop` / `forward_to_upstream`, stores result
3. **Async endpoints** - Implement the 5 new endpoints in `routes.rs`
4. **Config** - Add `async_cache_ttl_hours` (default 1), `async_max_concurrent` (default 1)
5. **Tests** - Unit tests for queue worker, integration tests for async endpoints

### Phase 2: CLAN-AI Client - Persistence Layer
1. **Database migration** - Add `pending_requests` table to `LocalDatabase` (schema v16)
2. **Model** - `PendingRequest` in `lib/data/models/pending_request.dart`
3. **Repository** - CRUD in `LocalDatabase` + methods in `ChatRepository`
4. **Tests** - Migration tests, CRUD tests

### Phase 3: CLAN-AI Client - Async API
1. **LlamaApiService** - Add `submitAsyncCompletion`, `streamAsyncCompletion`, `fetchAsyncResult`
2. **HttpClient** - Add methods for new endpoints
3. **Tests** - Network tests for new endpoints

### Phase 4: CLAN-AI Client - Streaming Integration
1. **StreamMutationMixin** - Modify `doStreamResponse` to use async flow
2. **Resume logic** - Handle existing `PendingRequest` on stream start
3. **Artifact handling** - Ensure downloaded artifacts re-attach on resume
4. **Tests** - Mixin tests for async flow, resume scenarios

### Phase 5: CLAN-AI Client - Lifecycle & Reconnection
1. **App lifecycle** - `didChangeAppLifecycleState` in `_ClanAiAppState`
2. **Background scan** - Periodic scan + immediate scan on resume
3. **Reconnection UI** - "Reconnecting..." indicator, progress
4. **Conflict handling** - Cancel background request on thread edit/delete
5. **Tests** - Integration tests for lifecycle scenarios

### Phase 6: Settings & Polish
1. **Settings UI** - Toggle + TTL config in settings
2. **Error handling** - 404 on expired ID → resubmit, network failures → backoff
3. **Documentation** - Update README, AGENTS.md
4. **E2E tests** - Full background completion flow

---

## Data Structures

### A-PROX `async_requests` Table (SQLite)
```sql
CREATE TABLE async_requests (
    id TEXT PRIMARY KEY,                    -- Client-provided UUID
    payload TEXT NOT NULL,                   -- Full request JSON
    status TEXT NOT NULL,                    -- queued, processing, completed, failed, cancelled
    result TEXT,                             -- Full response JSON (non-streaming) or final SSE events
    created_at INTEGER NOT NULL,             -- Unix epoch seconds
    updated_at INTEGER NOT NULL,             -- Unix epoch seconds
    expires_at INTEGER NOT NULL,             -- Unix epoch seconds (created_at + TTL)
    route_decision TEXT,                     -- FastPassThrough, RAGAugmented, AgenticToolLoop, etc.
    tokens_received INTEGER DEFAULT 0,
    error TEXT
);
CREATE INDEX idx_async_requests_status ON async_requests(status);
CREATE INDEX idx_async_requests_expires ON async_requests(expires_at);
```

### CLAN-AI `pending_requests` Table (SQLite)
```sql
CREATE TABLE pending_requests (
    request_id TEXT PRIMARY KEY,             -- Matches A-PROX request ID
    thread_id TEXT NOT NULL,                 -- Local thread ID
    assistant_message_id TEXT NOT NULL,      -- Local message ID
    payload TEXT NOT NULL,                   -- Full request JSON (for resubmit if needed)
    status TEXT NOT NULL,                    -- pending, streaming, completed, failed
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    server_base_url TEXT NOT NULL,           -- Server this request was sent to
    error TEXT
);
CREATE INDEX idx_pending_requests_status ON pending_requests(status);
CREATE INDEX idx_pending_requests_thread ON pending_requests(thread_id);
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Request ID | Client-generated UUID v4 | Idempotent retry, works offline, no round-trip |
| Auth | Same Bearer token + requestId in path | Consistent with existing API |
| Local cache | Final message only | Artifacts already in attachment store |
| Queue processing | SQLite in A-PROX | No external deps; Redis later if needed |
| Max concurrent | Per-user limit (configurable) | Avoids slot exhaustion on llama.cpp |
| Stale context | Include history hash in request | Server validates on resume; rejects if mismatch |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| A-PROX restart loses queue | SQLite persistence survives restart |
| Mobile OS kills app | Persist state to SQLite; resume on cold start |
| Slot queueing on llama.cpp | Client-side queue + server-side queue; expose queue position |
| Stale context on resume | Include history hash; server rejects with 409 if mismatch |
| Large responses | Cache only final message; artifacts in attachment store |
| Concurrent requests | Queue locally; process sequentially per thread |

---

## Testing Strategy

### A-PROX
- Unit: `AsyncRequestQueue` enqueue/dequeue, TTL cleanup, status transitions
- Integration: Submit async → poll status → stream result → fetch result
- Load: 50 concurrent async requests, verify ordering, no leaks

### CLAN-AI
- Unit: `PendingRequest` CRUD, migration v15→v16
- Integration: Send message → background → resume → verify artifacts
- Lifecycle: App background → foreground → resume stream
- Conflict: Edit thread during background → cancel old, submit new
- Web PWA: Service worker doesn't interfere with reconnection

---

## Configuration

### A-PROX (`config.toml`)
```toml
[async]
enabled = true
cache_ttl_hours = 1
max_concurrent = 1          # Concurrent background workers
cleanup_interval_minutes = 5
```

### CLAN-AI (Settings)
```dart
class Settings {
  bool backgroundCompletionEnabled = true;
  int cacheTtlHours = 1;
}
```

---

## Files to Modify

### A-PROX
- `src/db/schema.rs` - New async_requests table
- `src/db/mod.rs` - Export new schema
- `src/async_queue/` - New module: worker, endpoints, models
- `src/server/routes.rs` - Add 5 new endpoints
- `src/config.rs` - Add AsyncConfig
- `src/state.rs` - Add AsyncQueue to AppState
- `src/main.rs` - Start queue worker
- `tests/async_endpoints_test.rs` - Integration tests

### CLAN-AI
- `lib/data/datasources/local_storage.dart` - Schema v16, pending_requests table
- `lib/data/models/pending_request.dart` - New model
- `lib/data/repositories/chat_repository.dart` - Pending request CRUD
- `lib/data/datasources/llama_api_service.dart` - Async methods
- `lib/core/network/http_client.dart` - New endpoint methods
- `lib/ui/shared/mixins/stream_mutation_mixin.dart` - Async flow + resume
- `lib/main.dart` - Lifecycle observer + background scan
- `lib/ui/features/settings/` - Toggle + TTL config
- `test/` - All new tests

---

## Timeline Estimate

| Phase | Effort | Parallelizable |
|-------|--------|----------------|
| 1: A-PROX Core | 2-3 days | No (prerequisite) |
| 2: Client Persistence | 1 day | After Phase 1 |
| 3: Client Async API | 1 day | After Phase 1 |
| 4: Streaming Integration | 1-2 days | After Phases 2-3 |
| 5: Lifecycle & Reconnection | 1-2 days | After Phase 4 |
| 6: Settings & Polish | 0.5 days | After Phase 5 |
| **Total** | **6-10 days** | |

---

## Success Criteria

1. User sends message → navigates away (app backgrounded/killed) → returns → sees completed response
2. Image generation via A-PROX `/image` flag works in background
3. Artifacts (images/files) correctly downloaded and displayed on resume
4. Thread edit/delete during background generation cancels old request, submits new
5. All existing tests pass + new tests cover async flow
6. No regression in sync (non-async) request path