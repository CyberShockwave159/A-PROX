# Roleplay Overhaul — A-PROX memory handoff + scene image generation

Cross-repo implementation plan. Mirrored in both `CLAN-AI2/` (Flutter client) and
`A-PROX/` (Rust proxy) so it can be referenced while implementing.

Status: **implemented.** 715 Dart tests and 108 Rust tests pass; `flutter analyze`
is clean. The notes below record the design *as built*, including the decisions
that changed during implementation.

---

## 1. Two structural findings that shape the design

### 1.1 A-PROX's RAG is a document store, not conversational memory

A-PROX has **no memory feature** and **no direct ingest/query HTTP endpoints**
(`src/server/mod.rs:34-45` registers only `/health`, `/monitor/*`, `/v1/models`,
`/v1/chat/completions`, `/ingestion/reindex`, `/ingestion/status`,
`/images/{name}`, `/files/{name}`). For A-PROX to "replicate" CLAN-AI's memory the
client must *push* every roleplay turn in.

The existing `RAGIngestion` route (`src/server/routes.rs:348-392`) is unusable for
this:

- It hardcodes collection `"default"` (`:369`) — no per-character scoping.
- It forwards upstream afterwards (`:390`) for a full LLM acknowledgement
  round-trip, so every ingested turn costs a generation.
- Its guard falls through to the model when the payload is <3 words (`:355`).

`ingest_document` (`src/rag/mod.rs:31`) and `query_rag` (`src/rag/mod.rs:65`) are
both **sync** and directly callable, so new endpoints are cheap.

### 1.2 The step-1 image prompt gets rewritten a second time

The `/image` request routes to `RouteDecision::AgenticToolForced`
(`src/router/mod.rs:91-93`), which injects `prompts/i-iprompt.txt` — the "Edit
Prompt Enhancer" (`src/server/routes.rs:415`). The model then calls
`image_generate` with its own `rewritten_prompt`
(`src/tools/registry.rs:181-184`, consumed at `src/server/routes.rs:992-1002`).

**Step 1's output is a draft for the enhancer, not the final prompt.** Anything
carried in prompt text is at the mercy of a rewriter we don't control. Therefore
**style must be enforced at the workflow level** (`WorkflowTemplate::apply`,
`src/comfy_ui/workflow.rs:47`) *after* the rewrite.

Related: `negative_prompt` is an optional *model-supplied* argument
(`src/server/routes.rs:1015-1021`) that gets **appended** to
`image_generation.default_negative_prompt` (`:1033-1041`). That default is
realism-leaning (`config/default.toml:99` — `"plastic", "overexposed",
"bad composition"`) and actively fights an anime theme.

---

## 2. Locked decisions

| Concern | Decision |
|---|---|
| Memory write path | New A-PROX `/rag/ingest`, `/rag/query`, `/rag/collections/{n}/count` — no LLM call |
| Memory read path | `model: "a-prox-rag"` alias + `"rag": {collection, top_k, min_score}` object |
| Tool restriction | Per-message `"roleplay": true`; A-PROX intersects armed tools to `rag_search`/`rag_ingest`/`image_generate` |
| Memory scoping | `clan_<charId>_<threadId>`; visual sheet in `clan_<charId>_visual` |
| Ingestion | Fire-and-forget per completed turn + one-time backfill of existing threads |
| topK / minScore | Sent in the `rag` object; existing sliders stay meaningful in both modes |
| Toggle | Global `ServerConfig.serverSideRagEnabled`; local `clan_ai_vectors.db` preserved untouched |
| Client-RAG UI when off | Hide the RAG chip and "Manage Memories"; keep Settings Clear buttons and character export |
| Identity reference | Locked portrait → `avatarData` → offer to auto-generate |
| Refine | Long-press/tap an existing image → pick a variant → used as the next reference |
| Appearance sheet | New `characters.appearance` field, mirrored into RAG, LLM-drafted once with user review |
| Sheet contents | **Appearance only** — wardrobe deliberately excluded (it should change with the scene) |
| Style theme | `anime` / `semi-realistic` / `photo-realistic` / unset; enforced **both** in prompt text and via an A-PROX `image_style` field |
| Theme default | Unset — no directive sent until the user picks |
| Theme detection | "Detect from avatar" button; user confirms before it sticks |
| Theme surfaces | `CharacterEditDialog` (canonical) + inline in the portrait-offer and generate-image flows |
| Image caption | Discarded; original text kept verbatim in a new sibling variant |
| Image progress | Streaming placeholder in the bubble; blocks sends (reuses regeneration UX) |
| Roleplay upload | Attach button hidden entirely |
| Capability probe | `/health` already returns `"service": "A-PROX"`; add a `capabilities` array |

---

## 3. Phase 0 — A-PROX server (`src/`)

### 0.1 `/health` capabilities

`src/server/routes.rs:47-66` already emits `"service": "A-PROX"`. Add:

```json
"capabilities": ["rag", "image", "file"]
```

derived from live config (`image_generation.enabled`, `file_generation.enabled`;
`rag` always on). Also fix the stale hardcoded `"version": "0.1.0"` →
`env!("CARGO_PKG_VERSION")`.

### 0.2 New RAG routes

Register in `src/server/mod.rs` alongside the existing routes. All call the
existing **sync** engine, so no concurrency permit is required.

| Route | Body | Returns |
|---|---|---|
| `POST /rag/ingest` | `{collection, source_uri?, content}` | `{status, chunks}` |
| `POST /rag/query` | `{query, collection?, top_k?, min_score?}` | `{results:[{content, source_uri, collection, score}]}` |
| `GET /rag/collections/{name}/count` | — | `{collection, chunks}` |

- `ingest` → `rag_engine.ingest_document(collection, source_uri, content)`
  (`src/rag/mod.rs:31`).
- `query` → `rag_engine.query_rag(query, collection, top_k)` (`src/rag/mod.rs:65`),
  then filter on `SearchResult.score` (`src/db/vector_store.rs`). Default
  `top_k = 5` matches the existing `RAGAugmented` behaviour.
- `count` → a `SELECT COUNT(*)` against `document_chunks` filtered by collection.

### 0.3 Model-alias restoration (pre-existing bug)

`forward_to_upstream` (`src/server/routes.rs:612-628`) forwards the payload
verbatim, so a client sending `model: "a-prox-rag"` leaks that alias to
llama.cpp / any strict OpenAI backend. Rewrite aliases to
`config.upstream.model_alias` (`src/config.rs:68`) before forwarding:

`a-prox-rag|knowledge|docs|agent|tools|direct|pass|fast` → `upstream.model_alias`

### 0.4 Honor the `rag` request object

In the `RouteDecision::RAGAugmented` branch (`src/server/routes.rs:331-347`),
replace the hardcoded `query_rag(&query, None, 5)` with values parsed from a new
top-level `"rag"` object, and strip that key from the payload before forwarding.

```json
"rag": { "collection": "clan_abc_xyz", "top_k": 3, "min_score": 0.35 }
```

Absent → current behaviour, unchanged.

**This is load-bearing, not cosmetic:** `query_rag` searches *all* collections
when `collection` is `None` (`src/rag/mod.rs:65-72`). Per-thread scoping is what
stops the user's indexed notes from bleeding into roleplay.

### 0.5 Per-message `roleplay` flag → tool restriction

- Add `pub roleplay: Option<bool>` to `context::ChatMessage` with
  `#[serde(default, skip_serializing_if = "Option::is_none")]`.
- In `RequestRouter::classify_request` (`src/router/mod.rs:44`), detect the flag
  and intersect the armed tool set with the allow-list.
- Also guard the unforced `AgenticToolLoop` paths at
  `src/server/routes.rs:407-409` and `1377-1409`.

Allow-list: `rag_search`, `rag_ingest`, `image_generate`.

`RAGAugmented` / `RAGIngestion` / `FastPassThrough` arm no tools and are
unaffected. **This is the hard guarantee the user asked for: `web_search`,
`web_fetch`, `system_time` and `write_file` become unreachable from a roleplay
turn.**

### 0.6 `image_style` request field (theme enforcement)

Add a `[image_generation.styles]` config block. Threading:

`payload` → `execute_agentic_loop` (which already receives `payload`,
`src/server/routes.rs:436`) → `ImageGenContext.style` → `GenerateRequest.style`
→ applied inside `ImageGenService::generate` (`src/imagegen/orchestrator.rs:109`),
which **already holds `img_cfg`** — so no new plumbing is needed.

Applied in `WorkflowTemplate::apply` *after* the enhancer has rewritten the
prompt.

```toml
[image_generation.styles.default]
prompt_suffix = "photorealistic photograph, natural skin texture, shallow depth of field, physically accurate lighting, 35mm lens"
negative_prompt = "bad anatomy, distorted face, extra limbs, low quality, out of focus, overexposed, signature, watermark"

[image_generation.styles.anime]
prompt_suffix = "anime key visual, cel-shaded, clean line art, flat colour blocking, expressive eyes"
negative_prompt = "photorealistic, 3d render, cgi, skin pores, photographic, flat shading"

[image_generation.styles.semi-realistic]
prompt_suffix = "semi-realistic digital illustration, soft painterly shading, subsurface skin scattering, detailed fabric texture"
negative_prompt = "flat cel shading, chibi, plastic skin, harsh 3d render, photographic"
```

The style negative prompt **replaces** `default_negative_prompt`
(`config/default.toml:99`) rather than appending — that default is
realism-leaning and fights the anime theme.

Unknown style key → fall back to `default`.

### 0.7 Docs

- `README.md`: new endpoint reference, the `rag` / `roleplay` / `image_style`
  request fields, the `[image_generation.styles]` config block, and the updated
  list of RAG trigger mechanisms.
- `AGENTS.md`: update the routing-priority list in the Model Routing section.

---

## 4. Phase 1 — CLAN-AI plumbing (`lib/`)

| Change | File |
|---|---|
| `ragIngest` / `ragQuery` / `ragCollectionCount` constants | `lib/core/constants/api_endpoints.dart` |
| `PingResult.isAprox` + `Set<String> capabilities`, parsed from `/health` (the 15s health poll feeds it free) | `lib/core/utils/latency_meter.dart:14-27` |
| **Non-serialized** `capabilities` + `bool get isAprox` — every VM method already receives `ServerProfile? connection`, so capability data flows in with **zero new wiring** | `lib/data/models/server_profile.dart` |
| `toOpenAiPayload` gains `modelOverride`, `rag`, `extraBody` (merged last, so future A-PROX fields need no signature churn) | `lib/domain/models/generation_params.dart:69-97` |
| `_streamOpenAi` gains the same three args; `roleplayMode` adds `"roleplay": true` to each message object; model becomes `modelOverride ?? selectedModel ?? 'default'` | `lib/data/datasources/llama_api_service.dart:113-163` |
| Forward the new args; add `rag`, `ingestRagTurn`, `queryRag`, `ragCollectionCount`, `completeOnce` | `lib/data/repositories/chat_repository.dart:140-158` |

**Explicit array-content builder** (new, in `llama_api_service.dart`):
`_serializeOpenAiContent` only covers DB-backed user messages, but A-PROX's
`extract_image_part` inspects **only** array content
(`src/imagegen/mod.rs:138-146`). The synthetic image request must therefore be
built as `[{type:text},{type:image_url}]` by hand.

**New `lib/data/datasources/aprox_rag_client.dart`** — thin wrapper over
`ApiHttpClient`; `collectionFor(characterId, threadId) => 'clan_${characterId}_$threadId'`.
**All failures swallowed and logged** — memory is best-effort and must never
break a reply.

---

## 5. Phase 2 — Schema v15 → v16

`lib/data/datasources/local_storage.dart:52` → `version: 16`, with a guarded
`ALTER TABLE` block following the existing `PRAGMA table_info` pattern
(`:227-234`), plus **both** `CREATE TABLE characters` blocks (`:120`, `:299`).

| Column | Type | Purpose |
|---|---|---|
| `appearance` | `TEXT` | Canonical visual description (the "appearance sheet") |
| `identity_portrait_data` | `BLOB` | User-approved reference portrait, ≤512px JPEG |
| `visual_theme` | `TEXT` | `anime` \| `semi-realistic` \| `photo-realistic` \| NULL |

BLOB mirrors the existing `avatar_data` pattern (`:308`) rather than introducing
file-path management. Also add `CharacterProfile` fields, `toMap`/`fromMap`,
`CharacterRepository` read/write, export/import round-trip, and a new
`VisualTheme` enum with `label`, `wireValue` and `isSet`.

**No `messages` migration is needed** — scene images reuse the existing
`image_path` column and the existing variant bookkeeping.

---

## 6. Phase 3 — Roleplay memory handoff

### 6.1 Setting

- `ServerConfig.serverSideRagEnabled` (default `false`, so existing users keep
  client RAG).
- `SwitchListTile` at the top of the existing roleplay-only RAG Memory section
  (`lib/ui/features/settings/views/settings_screen.dart:538-642`). Grey out the
  per-character counts when on; **keep** Clear / Clear All so local data stays
  recoverable.
- The two existing sliders
  (`lib/ui/features/settings/views/parameter_tuning_sheet.dart:250-276`) now feed
  the `rag` object in both modes. This also **fixes a live bug**: no roleplay
  call site currently passes `customParams`, so `ragTopK`/`ragMinScore` always
  resolve to `3` / `0.0`.

### 6.2 Gate the client RAG path

In `lib/ui/features/roleplay/view_models/roleplay_view_model.dart`, sourced from
the `serverConfig` already passed to every method (no new plumbing):

- Skip `RoleplayContextBuilder` retrieval: `:318-332`, `:416-429`, `:653-665`.
- Skip the `ragMemoryCount` / `ragMemoryContents` snapshots: `:343-346`,
  `:439-442`, `:524-533`.
- Skip `_embedMessageAsync`: `:271`, `:712-729`.
- Skip the embedding deletions: `:390-399`, `:624-627`, `:745-749`.

Fix two latent bugs while in there:

- `editUserPrompt` (`:417-429`) and `deleteMessage` (`:653-665`) omit `threadIds`,
  leaking sibling-branch memories into context.
- `_embedMessageAsync`'s `isFirstMessage` param (`:770`) is declared but unused,
  contradicting its own doc comment.

### 6.3 Server-side writes

- `_ingestTurnToServer()` — posts a synthesized document per turn from the
  existing `onComplete` hook slot:
  `[Character: X | Thread: Y]\nUser: …\nX: …`, with
  `source_uri: clan/<charId>/<threadId>/<messageId>`.
- `_backfillThreadToServer()` — runs once per session when
  `ragCollectionCount(collection) == 0`; batches the existing history in 5-turn
  batches.
- **Image variants are excluded** — captions and image prompts must never enter
  memory.

### 6.4 Request shape

When server RAG is on:

```dart
modelOverride: 'a-prox-rag',
rag: {'collection': ..., 'top_k': params.ragTopK, 'min_score': params.ragMinScore},
roleplayMode: true,
```

`selectedModel` in `ServerConfig` is never mutated.

### 6.5 UI cleanup

- Hide the RAG-memory chip (`lib/ui/features/chat/views/message_bubble.dart:706-737`).
- Hide "Manage Memories" (`lib/ui/features/roleplay/views/roleplay_drawer.dart:470-478`).
- Keep "Export Character + Memories" so the local store stays inspectable.

> **Prompt-shape note:** A-PROX injects retrieved context by *prepending* a
> `### Relevant Retrieved Context: …` block to the **last user message**
> (`src/server/routes.rs:2327-2341`), whereas CLAN-AI injects into the **system
> prompt**. Retrieval still works; the retrieved text carries more recency
> weight. Worth one live eyeball with a real character.

---

## 7. Phase 4 — Character consistency

### 7.1 Reference resolution

`_resolveIdentityReference()` in the roleplay VM:

1. `character.identityPortraitData` if set → use it.
2. else `character.avatarData` if non-null → **transcode to ≤512px JPEG** → use it.
3. else → no reference; show the portrait-offer dialog.

**Transcode is mandatory.** `guess_format` (`src/imagegen/mod.rs:154-165`) sniffs
magic bytes and accepts only PNG/JPEG/WebP. AVIF or anything else silently falls
through to t2i — a reference that "does nothing" with no error.

### 7.2 Portrait-offer dialog

Shown only when A-PROX is detected and the character has no portrait. One-shot
t2i `/image` with a portrait-sheet directive and **no** reference image.
Phrased per the selected theme, e.g. for anime:

> *A centred head-and-shoulders character reference portrait of {name}, anime key
> visual, cel-shaded, front-facing, neutral expression, plain neutral
> background.*

Deliberately not a scene prompt — a bust shot conditions identity far better.
Renders with **Approve** / **Regenerate** / **Discard**. Approve writes the
downscaled JPEG to `identity_portrait_data`, after which the dialog never
reappears for that character.

### 7.3 Appearance sheet

- Source of truth: `characters.appearance`, editable in `CharacterEditDialog`.
- On every `CharacterRepository` save with a non-empty value → `POST /rag/ingest`
  into `clan_<charId>_visual` with `source_uri: clan/<charId>/visual` (stable, so
  it replaces rather than accumulates).
- Retrieved via `POST /rag/query` at step 1 and injected as a hard constraint.
- **LLM-drafted once**, on first image request, when `appearance` is empty and
  `personality.length > 200` (the ST card `description` blob —
  `lib/core/utils/silly_tavern_card_parser.dart:114-121`): a one-shot extraction
  strips the card to physical appearance only, shown in an editable review sheet
  with **Save** / **Regenerate** / **Skip**. Nothing is written until the user
  saves.

### 7.4 Theme

`VisualTheme` is sent **both** ways, as decided:

1. In the step-1 prompt, so the enhancer elaborates in the right register.
2. As the `image_style` request field — authoritative, applied post-enhancer
   (Phase 0.6).

UI: a dropdown under the avatar in `CharacterEditDialog` with a
*"match your avatar's style — mismatches reduce identity consistency"* hint,
plus a **Detect from avatar** button that runs one cheap vision call
("which style is this image: anime, semi-realistic, or photo-realistic?") and
preselects without saving. Fails gracefully with a snackbar if the upstream model
isn't multimodal (A-PROX already does vision calls for PDFs, so the pattern is
supported, but the model must be vision-capable).

The same dropdown appears inline in the portrait-offer and generate-image
dialogs. `CharacterEditDialog` remains the single source of truth.

### 7.5 Image request shape

The final user message must be an **array**:

```json
{"role": "user", "content": [
  {"type": "text", "text": "<step-1 draft, themed>"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]}
```

`rewrite_latest_user_text` (`src/server/routes.rs:883-900`) replaces the text
part in place and leaves `image_url` intact, so the `/image` flag still routes and
`extract_reference_image` still finds the reference.

The prompt must be phrased as a **new picture of this subject in a new scene** —
`prompts/i-iprompt.txt` is an *editing* prompt ("An input image is ALWAYS present
— this is always an image-editing task, never text-to-image from nothing") and
otherwise tries to preserve the portrait's own pose and background. Its guidance
("Preservation locks content, never edit strength"; hold unmentioned attributes
at input fidelity) is exactly what we want, so we deliberately do not override
it.

These messages are **not** marked `roleplay`, so only `image_generate` is armed.

### 7.6 Refine from an existing image

Long-press (or tap) a generated image → sheet listing all variants of that
message → pick one → it becomes the reference for the next generation, producing
a new variant.

Consequences worth naming explicitly:

- The "not with the previously generated image" rule now applies **only to the
  Generate Image button**. Re-clicking that button always starts from the locked
  portrait.
- Chains are possible (refine the refined image), each hop accumulating drift from
  the portrait. The appearance sheet is re-injected each hop to counteract it.
  Nothing caps chain depth.

### 7.7 Message bubble

- `onGenerateImage` param + one icon after Regenerate
  (`lib/ui/features/chat/views/message_bubble.dart:827-839`), same 28×28
  constraints / `context.clanTextMuted`, gated
  `!isUser && status != streaming && onGenerateImage != null`.
- `lib/ui/features/roleplay/views/roleplay_screen.dart:239` passes it only when
  `capabilities.contains('image')`; `chat_screen.dart:121` never does, so
  assistant mode is structurally excluded.
- The bubble has no `isRoleplay` field — capability presence *is* the gate, the
  same pattern the file already uses for `characterName` / `onEditAssistant`.

### 7.8 `generateImageForMessage`

1. **Guard** — refuse if `!capabilities.contains('image')`, `isGenerating`, or
   `messages[i].role != assistant`.
2. **Step 1 — write the draft prompt.** `completeOnce` with a dedicated minimal
   system prompt (avoids the roleplay identity guard at
   `lib/core/utils/roleplay_prompt_formatter.dart:85-87` fighting the
   instruction). Body: the last 3 exchanges (≤6 messages, ending at the target
   assistant message) + the retrieved appearance sheet + the theme + a final user
   turn *"Write the image prompt for the scene depicted above."*
   **The final turn is prefixed `/bypass `** so A-PROX takes `FastPassThrough`
   (`src/router/mod.rs:85-87`, checked *before* `is_image_request` at `:97`) —
   otherwise a draft like "paint a picture…" trips the image classifier and burns
   a generation.
3. **Step 2 — generate.** New `doCreateVariantMessage` mixin helper modeled on
   `doRegenerateMessage`
   (`lib/ui/shared/mixins/stream_mutation_mixin.dart:530-582`) but seeding
   `content: original.content`, streamed with a new
   `textMode: StreamTextMode.discard` that skips the buffer append (`:114`) and
   the text flush (`:68-72`) while leaving reasoning, artifact capture and
   metrics intact. The mixin's existing `delta.image_url` handling (`:147-161`)
   downloads the bytes and persists `imagePath`; A-PROX's caption is dropped.
4. **Result** — if no image arrives, revert to the original message and snackbar.
   No empty variant is left behind.
5. **Re-clicking** runs the identical flow and produces another sibling variant.
   The previously generated image is never sent, because `_serializeOpenAiContent`
   (`llama_api_service.dart:174-202`) only serializes images for
   `MessageRole.user`. ✅
6. **RAG interaction** — image variants skip embedding/ingestion entirely, and
   the original message's embeddings are left alone (unlike `regenerateMessage`,
   which deletes the old embedding at `:390-399`).

### 7.9 Composer

- Wrap the attach `IconButton` in `if (!widget.isRoleplay)`
  (`lib/ui/features/chat/views/prompt_input_bar.dart:141-154`).
- Drop `imagePath` from the roleplay `onSend` closure
  (`lib/ui/features/roleplay/views/roleplay_screen.dart:343`).
- Ignore any `imagePath` passed to `RoleplayViewModel.sendMessage`.

---

## 8. Phase 5 — Tests (all hermetic)

### A-PROX (Rust)

- `roleplay` restriction branch in `classify_request`.
- Alias → `upstream.model_alias` rewrite.
- Style suffix / negative prompt applied in `WorkflowTemplate::apply`.
- `rag` object parsing (collection / top_k / min_score, and the absent fallback).

### CLAN-AI

- v15→v16 migration guard.
- `appearance` / `visual_theme` / `identity_portrait_data` round-trips.
- Reference priority chain (3 cases).
- Reference sent as **array content, not a string**.
- Non-JPEG/PNG/WebP source is transcoded, not sent raw.
- No reference + no avatar → offer dialog, not a silent t2i.
- `/bypass` present on the step-1 final turn.
- `StreamTextMode.discard` suppresses text but still captures the artifact and
  persists `imagePath`.
- `modelOverride` wins over `selectedModel`.
- `roleplay: true` on every message in roleplay mode, absent otherwise.
- Appearance ingested on save with a stable `source_uri`.
- Image variants excluded from RAG ingestion.
- Attach button visible in assistant mode, absent in roleplay.
- Icon present/absent per `onGenerateImage`.

**`test/helpers/fake_chat_repository.dart` must gain the new `ingestRagTurn` /
`queryRag` / `completeOnce` / `ragCollectionCount` members.** It `implements` the
concrete repo, so this is a hard compile requirement.

---

## 9. Phase 6 — Docs

- `README.md` — roleplay memory backends, scene image generation, the A-PROX
  endpoint contract.
- `ARCHITECTURE.md` — the image-gen variant path; the three request fields
  (`modelOverride` / `rag` / `roleplay`); "server RAG and client RAG are mutually
  exclusive".
- `AGENTS.md` — never write embeddings *and* ingest; image variants never enter
  memory; assistant images are never re-sent.
- `CHANGELOG.md`.

---

## 9a. What changed during implementation

- **`image` package added** (`pubspec.yaml`). Reference normalisation needs real
  image decode/re-encode: an avatar in a format A-PROX ignores must be
  transcoded or the reference silently does nothing. Pure Dart, so it works on
  web/PWA too and is testable without a platform channel.
- **`IdentityReferenceResolver` degrades rather than fails.** A corrupt or
  undecodable reference is passed through unchanged, downgrading the request to
  text-to-image instead of erroring — a scene without a consistent face beats no
  scene. `isSupportedFormat` lets the caller detect and warn.
- **`RequestOptions.referenceImage` instead of array content in the model.** A
  synthetic `ChatMessage` cannot carry typed content parts (its `content` is a
  `String`, and the array form only comes from a stored attachment), so the
  reference rides on the request options and `LlamaApiService` attaches it to
  the final user message at serialisation time. One place, always the right
  message.
- **The system prompt is tagged too.** The plan said "every message"; the first
  implementation tagged only user/assistant turns. A request whose only message
  is the system prompt would then be unmarked, so the system message is tagged
  too.
- **Variant revert uses the database, not memory.** `doRevertVariant` re-reads the
  original from SQLite rather than trusting the in-memory copy, and re-checks
  that the placeholder is still the visible variant before overwriting.
- **`_parseDate` hardening in `CharacterProfile.fromMap`.** A new round-trip test
  surfaced that a row with missing `created_at`/`updated_at` threw, which would
  take down the whole character list. Now tolerated.
- **Two latent roleplay bugs fixed in passing**: `editUserPrompt` and
  `deleteMessage` were not scoping RAG search to the thread lineage (leaking
  sibling-branch memories), and `_embedMessageAsync`'s `isFirstMessage` was
  declared but unused.
- **`ChatRepository` is now also provided via `Provider`** in `main.dart` so the
  auxiliary character-image calls (style detection) can reach `completeOnce`
  without going through a view model.
- **First live test found a real bug in `completeOnce`.** It inherited
  `serverConfig.defaultParams.reasoning`, so on a reasoning model the image
  prompt draft burned its whole 512-token budget thinking and returned empty
  `content` — the request succeeded and the feature silently produced nothing.
  Fixed by forcing `reasoning: false` (a thinking preamble is waste for a
  one-paragraph artefact) and by falling back to `reasoning_content` if `content`
  is empty. Lesson: `streamChatCompletions` pins `reasoning` from the config, so
  the asymmetry was easy to miss; every new entry point has to make the same
  decision explicitly.

## 9b. Post-implementation optimization: skipping the caption turns

The first live run revealed that the `/image` request made **five** upstream
llama.cpp passes, two of which exist only to produce text this client throws
away:

| Pass | Prompt tokens | Wall time | Needed? |
|---|---|---|---|
| 1 — scene prompt draft | 1771 | 15.4s | yes |
| 2 — Phase A, enhancer → `image_generate` | 8536 | 38.5s | yes |
| 3 — image job (llama stopped, ComfyUI, restart) | — | 148.8s | yes |
| 4 — Phase C, vision caption of the generated image | 9091 | 38.5s | **no** |
| 5 — synthesis fallback (caption came back empty) | 1992 | 46.6s | **no** |

A-PROX gained an opt-in `image_only` request field: on `image_generate` success
it emits `delta.image_url` + `finish_reason` + `[DONE]` and returns, skipping
passes 4 and 5. The artifact event is emitted by A-PROX, not the model, so the
image does not depend on the vision turn at all. Gated on the image actually
existing, so a failure still falls through and gets reported. The client sets it
from `RequestOptions.sceneImage` because it streams with
`StreamTextMode.discard`.

Savings: ~85s of a ~200s request. Passes 1 and 2 remain — the draft and the
enhancer both shape the final prompt, so neither is removable without changing
the output.

## 9c. Second live run: the silent no-image failure

The first `image_only` run still failed with "server returned no image". The log
showed the pipeline stopping dead after Phase A:

```
0.48.658  slot launch_slot_: id 2 | task 518
1.58.312  eval time = 40561.63 ms / 2048 tokens     ← 2048, not 512
1.58.313  release: id 2 | task 518                  ← and nothing after
```

No `image_generate` call, no ComfyUI, no artifact — the request completed
"successfully" with no image and no error.

**Why 2048.** A-PROX never sets `max_tokens` on agentic-loop turns, so the
upstream default governed, and llama.cpp's is 2048. (The existing
`llama_server.image_max_tokens = 2048` is a *different* knob — llama.cpp's
vision-context budget, passed as `--image-max-tokens`.) An image turn must *plan*
before it can call the tool, so a model that reasons at length gets cut off
mid-thought and the tool call never happens. The failure is invisible from the
client: HTTP 200, valid SSE, just no `delta.image_url`.

**Fix.** New `[guardrails] max_generation_tokens` (default 4096), applied by
`apply_generation_budget` to every agentic-loop turn — as a *default* only, so a
client that sets its own `max_tokens` still wins. `0` omits the field and restores
the upstream default.

**Second contributor.** The step-1 draft was still capped at 512 tokens, and this
model narrates a "Here's a thinking process: 1. Analyze User Input…" preamble
into `content` (because reasoning is forced off). At 512 the preamble consumed the
whole budget, so A-PROX received the instruction truncated mid-sentence. Raised
to 1024, and `completeOnce` no longer overrides a caller's explicit `maxTokens`.

## 10. Residual risks

1. **Avatar style mismatch is user-controlled, not solved.** The theme makes the
   failure *legible* rather than mysterious, but an anime avatar with a
   photo-realistic theme will still condition poorly. "Detect from avatar" is the
   mitigation.
1b. **Verified end-to-end against a live A-PROX + ComfyUI** (Qwen3.6-35B-A3B
   Uncensored, Qwen-Image 2.1 on a 3080 Ti). Three live runs fixed three real
   bugs that hermetic tests could not catch: `completeOnce` inheriting the
   reasoning flag, the silent no-image failure from llama.cpp's 2048 cap, and the
   two redundant post-image turns. Character consistency held, and the image
   attached to a new variant without altering the original text.

3. **The 60s receive timeout is the real ceiling on the draft, not the token
   budget.** After reasoning was enabled, the draft took **62.4s** and was
   aborted by `ApiHttpClient.post`'s 60s receive budget — *after the server had
   already answered*, with `n_gen 2923` and `truncated = 0`. The client reported
   "the model returned an empty image prompt", which points nowhere near the
   cause. Raising the token budget alone would have made this worse, not better.

   `post` now takes a per-call `timeout`, threaded through `completeOnce`. The
   draft uses a 10-minute ceiling, sized from the budget: 16384 tokens at a
   typical 50 tok/s is ~5.5 min plus prompt evaluation. It is a ceiling, not a
   cost — the request returns as soon as the model finishes (~2.9k tokens, ~60s
   observed).

   **Lesson:** any auxiliary call with a large `maxTokens` ceiling needs its own
   `timeout`. The shared budget is tuned for chat turns, and a timeout that fires
   after a successful response is indistinguishable from an empty model reply.

2. ~~**The step-1 draft still hits its cap.**~~ **Fixed.** The cause was forcing
   reasoning *off* for the draft: with no scratchpad, a reasoning model narrated
   a "Here's a thinking process: …" preamble into `content`, which then had to
   share the budget with the prompt and was cut off mid-sentence. `completeOnce`
   now takes per-call `reasoning` / `maxTokens`; the draft runs with reasoning
   **on** at 4096 tokens (`SceneImagePrompts.draftMaxTokens`) so the deliberation
   lands in `reasoning_content` and `content` holds just the prompt. The cheap
   auxiliary calls keep reasoning off at 256 tokens. Costs extra wall time on the
   draft, which was accepted as the better product.
2. **Only one reference slot.** `images.image_1` is the only image input in
   `workflows/i2i.json`. Feeding portrait + last scene image would need
   `images.image_2` support in the installed ComfyUI `TextEncodeQwenImage21` node
   (unverified) plus workflow changes. Deliberately left off.
3. **Concurrency.** A-PROX allows one active inference by default. Step 1 must
   complete before step 2 (it does, sequentially) and per-turn ingestion fires
   only after the main stream completes.
4. **Model-alias leakage is a pre-existing A-PROX bug.** CLAN-AI must not ship
   `modelOverride: 'a-prox-rag'` against an unpatched server — hence the
   capability gate.
5. **Prompt pollution on the RAG path is gone** (model alias, no `/rag` prefix),
   but step 1 carries a `/bypass` prefix the user never sees, since it is a
   wire-only one-shot request.
6. **Step-1 output is not the final prompt** — it is a draft for A-PROX's
   enhancer. This is why the theme is enforced at the workflow level too.

---

## 11. Recommended first implementation step

Make **step 1 a live manual test before any UI work**: attach a real character
avatar to a `/image` request with a scene prompt and inspect the output. If
reference conditioning does not hold identity on this setup, layers 2 and 3 will
not save it and the design needs rethinking.


## Variant navigation + image delivery fix (2026-09-25)

Two independent defects made a generated scene image look like it never arrived.

### 1. Variant group metadata was inconsistent (fixed)
`doRegenerateMessage` / `doCreateImageVariant` gave every sibling the new
`siblingIds` but never updated `totalVariants`, and `doRevertVariant` restored
the original with the deleted placeholder still listed. `doSwitchVariant` also
substituted `currentMsg` for any unresolvable sibling id
(`orElse: () => currentMsg`), producing a duplicate with the same `variantIndex`
so the switch silently no-opped behind a live-looking arrow.

Evidence from `~/Documents/clan_ai.db`: a real image variant had
`total_variants = 4` with a 2-entry `sibling_ids` that excluded itself, so it
was unreachable from the displayed message.

Fix: `_syncVariantGroup` recomputes `siblingIds` **and** `totalVariants` for
every member from the authoritative id set; `variantGroupFields` gives the same
values to a not-yet-persisted message; `doSwitchVariant` drops unresolved ids.
The production DB still contains groups written by the old code; they will heal
on the next regeneration in that group.

### 2. `filterReasoning` dropped artifact-only chunks (the actual "no image")
A-PROX `image_only` mode sends the `delta.image_url` event and **nothing else**
— no vision caption, no synthesis turn. `SseClient.filterReasoning` only ever
yields in response to text or reasoning, attaching artifacts to those yields via
`_copyArtifacts`. A chunk carrying an artifact but no text therefore produced no
yield and was discarded.

This was invisible while A-PROX also streamed a caption: the artifact rode along
on a text-bearing chunk. Turning on `image_only` removed the text and silently
broke delivery — the server generated and served the image (verified: `GET
/images/gen_*.png` -> 200, `image/png`) while the client reported "the server
returned no image" and reverted the variant.

Fix: pass artifact-only chunks straight through. Regression tests cover the
image-only shape, a 100-frame keepalive run (a real image request idles in
ComfyUI for minutes behind `: keepalive` comments), and no double `isDone`.

### 3. Diagnostics
`downloadImageArtifact` swallowed every exception and returned null, which the
roleplay flow reported as "the server returned no image" — pointing at the server
when the fault was local. It now logs and rethrows, and the stream-finalizer
timeout logs too. A client restart is required for these changes to take effect.
