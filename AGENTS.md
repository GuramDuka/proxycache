# proxycache

OpenAI-compatible proxy for `llama.cpp` KV cache slot management with disk save/restore.

## Environment

**WSL2** — use `uv` from Windows for package management. Do NOT install Python packages via pip/apt inside WSL2.

**Python ≥3.10** required. Dependencies: fastapi, httpx, uvicorn.

## Commands

```bash
uv sync                                        # install deps (pyproject.toml)
python proxycache.py                           # run (env vars from config.py)
uvicorn app:app --host 0.0.0.0 --port 8081     # or via uvicorn directly
python test_smoke.py                           # smoke tests (no framework, uses unittest.mock)
./build-cache-agent.sh                         # build Go cache-agent binary
```

**No linter, no typechecker, no test framework.** Smoke tests use `unittest.mock` + `asyncio.run()`.

## Architecture

| File | Role |
|------|------|
| `proxycache.py` | 13-line uvicorn entry point — **not** where main logic lives |
| `app.py` | FastAPI app, routes, `StreamReader` class, streaming pipeline, `POST /v1/chat/completions` |
| `backend_manager.py` | Singleton: backend registry, `LlamaClient`/`CacheAgentClient` instances, model-to-backend mapping, refresh cooldowns, liveness checker |
| `config.py` | All config via env vars (no .env file) |
| `hashing.py` | Text → word-block hashing, LCP matching, meta key generation |
| `llama_client.py` | httpx client to llama.cpp; slot save/restore, router mode slot discovery, auth headers |
| `slot_manager.py` | Per-model slot pools, ring buffer eviction, KV cache skip logic, acquire/release |
| `kv_meta_manager.py` | Singleton: meta file read/write/list/delete/reconcile, restore candidate scoring |
| `cache_agent_client.py` | HTTP client for remote cache file deletion |
| `cache-agent/` | Go cache agent (lightweight HTTP server for remote cache deletion) |
| `kv_meta/` | Per-cache `.meta.json` files (gitignored) |

## Gotchas

- **llama.cpp prerequisite**: MUST start with `--slot-save-path <dir>`. Cache save/restore fails silently without it.
- **Config**: all env vars only — `config.py` has defaults. No `.env` file support.
- **Cache key**: `sha256(canonical_name + '\n' + ','.join(token_ids))` — model name is included in the key.
- **Backend keys**: stable `host:port` strings (e.g. `"10.0.0.1:8000"`), NOT integer indices. Integer indices change across restarts. Sanitized to `host-port` for filesystem dirs.
- **Slot pinning** is duplicated 3 ways in every request body: root (`slot_id`, `id_slot`, `_slot_id`), `options` dict, and query params.
- **Save happens after response** completes (both stream and non-stream), never before.
- **Streaming**: background `reader` task races socket reads against disconnect event → `asyncio.Queue`. Heartbeat checks `is_disconnected()` every 0.5s. `stream()`'s `finally` calls `_cleanup()` which saves the slot only if `_stream_complete` is True (stream finished normally, not cancelled mid-stream), then releases it.
- **Slot acquire timeout**: 60s hardcoded (`ACQUIRE_TIMEOUT` in app.py). Returns 503 if all slots busy. Retry loop sleeps 5s × attempt number, up to 6 retries.
- **Slot timeout**: `SLOT_TIMEOUT` (default 30s) wraps `/slots/{id}?action=save|restore`. Separate from `REQUEST_TIMEOUT` (600s).
- **KV cache skip**: `acquire_for_request` checks `_slot_kv_state` before restoring. If slot's tracked KV cache blocks have LCP ratio >= `KV_CACHE_SKIP_THRESHOLD` (default 0.9), restore is skipped — llama.cpp appends to existing cache. Only safe on single-slot backends.
- **Ring buffer eviction**: `SlotManager` evicts expired entries (age-first) then LRU when `_total_bytes > CACHE_MAX_SIZE_GB`. Only triggers on saves.
- **Slot refresh cooldown**: 300s per (model, backend) pair on success, 30s on failure. On-demand discovery via `GET /slots` (non-router) or `GET /models` + child `/slots` (router mode). Falls back to 1 slot if discovery fails.
- **Meta reconciliation**: on startup, orphaned/corrupted `.meta.json` files are deleted via `kv_meta.reconcile()`.
- **Model resolution**: exact match → case-insensitive substring → "any" (all models). Using a more generic name distributes requests across all backends that serve matching models.
- **Cache save skip**: if restore ratio >= `CACHE_SAVE_RATIO_THRESHOLD` (default 0.9) and no recompute happened, save is skipped to avoid duplicate cache entries.
- **Recompute detection**: if `cached_tokens < prompt_tokens * 0.92`, the KV cache restore was partial/useless — recompute penalty is incremented.
- **Liveness checker**: pings backends every 5s, triggers discovery on state change.
- **API key**: `BACKEND_API_KEY` env var or per-backend `api_key` in `BACKENDS` JSON. Per-backend takes priority. Forwarded via `x-api-key` header on all requests.
- `.gitignore` covers `kv_meta/`, `venv/`, `__pycache__/`, `run-proxycache.ps1`, `uv.lock`, `cache-agent.exe`.
