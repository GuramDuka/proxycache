# proxycache — Change Report & Context

> Created: 2026-06-08
> Purpose: Document all changes made to make proxycache-proxy work with llama.cpp builds that require an API key and support on-demand model loading.
> Status: All changes verified and tested.

---

## Table of Contents

1. [Problem: 401 Unauthorized on all backend endpoints](#1-problem-401-unauthorized-on-all-backend-endpoints)
2. [Problem: 400 Bad Request on /apply-template and /tokenize](#2-problem-400-bad-request-on-apply-template-and-tokenize)
3. [Problem: 501 Not Implemented on /slots?action=save](#3-problem-501-not-implemented-on-slotsslotidactionsave)
4. [Problem: discover_models returned empty list](#4-problem-discover_models-returned-empty-list)
5. [Problem: "all slots busy" for unloaded models](#5-problem-all-slots-busy-for-unloaded-models)
6. [Backend API Key Configuration](#6-backend-api-key-configuration)
7. [Request Flow Diagram](#7-request-flow-diagram)
8. [File Structure (Current State)](#8-file-structure-current-state)
9. [Run Commands](#9-run-commands)
10. [Potential Future Issues](#10-potential-future-issues)
11. [Verified Scenarios](#11-verified-scenarios)

---

## 1. Problem: 401 Unauthorized on all backend endpoints

### What happened
The llama.cpp build (with router mode) requires an `x-api-key` header for **all** requests:
- `POST /apply-template` — returned 401
- `POST /tokenize` — returned 401
- `GET /slots` — returned 401
- `POST /slots/{id}?action=save` — returned 401

### Solution
**File:** `llama_client.py`

Added `api_key` parameter to `LlamaClient.__init__()` and a new `_auth_headers()` method:

```python
class LlamaClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self._api_key = api_key

    def _auth_headers(
        self, headers: Dict[str, str] | None = None
    ) -> Dict[str, str] | None:
        """Merge provided headers with the client's API key. Per-request headers take priority."""
        if not self._api_key:
            return headers
        merged = {"x-api-key": self._api_key}
        if headers:
            merged.update(headers)
        return merged
```

### Why in the constructor (not per-call)?
- **Background calls** (`refresh_slot_counts`, `_get_child_slots`) don't have access to the HTTP request body
- If the key were only passed to `chat_completions`, background requests to `/slots` and `/models` would fail with 401
- Putting the key at the client level covers **all** HTTP calls automatically

### Updated methods (all now pass `_auth_headers()`)
| Method | Endpoint | Auth? |
|--------|----------|-------|
| `apply_chat_template` | `POST /apply-template` | ✅ |
| `tokenize` | `POST /tokenize` | ✅ |
| `chat_completions` | `POST /v1/chat/completions` | ✅ |
| `save_slot` | `POST /slots/{id}?action=save` | ✅ |
| `restore_slot` | `POST /slots/{id}?action=restore` | ✅ |
| `get_slot_status` | `GET /slots/{id}` | ❌ (no key yet) |
| `discover_models` | `GET /models` + `GET /v1/models` | ✅ |
| `_get_child_slots` | `GET http://127.0.0.1:{port}/slots` | ✅ |

> ⚠️ **Important:** `get_slot_status` does not yet pass auth. If the backend starts requiring the key there too, add `self._auth_headers()` to that method.

---

## 2. Problem: 400 Bad Request on /apply-template and /tokenize

### What happened
llama.cpp requires a `"model"` field in the request body for both `/apply-template` and `/tokenize`. Without it, returns 400.

### Solution
**File:** `llama_client.py`

Both methods now accept `model_name` and include it in the request body:

```python
async def apply_chat_template(self, messages, model_name=None, headers=None):
    body = {"messages": messages}
    if model_name:
        body["model"] = model_name
    # ...

async def tokenize(self, text, add_special=False, model_name=None, headers=None):
    body = {"content": text, "add_special": add_special}
    if model_name:
        body["model"] = model_name
    # ...
```

`model_name` is passed from `app.py` via `BackendManager.get_model_n_ctx()`.

---

## 3. Problem: 501 Not Implemented on /slots?action=save

### What happened
Some llama.cpp builds return `501 Not Implemented` on `POST /slots/{id}?action=save`.
Previously, the code only handled `500 Internal Server Error`.

### Solution
**File:** `llama_client.py` → `save_slot()`

```python
if resp.status_code >= 500:
    log.warning("Save slot returned %d: slot=%d, file=%s", ...)
    return False, 0
```

Now handles **all** 5xx codes (500, 501, 502, 503, 504...).

### Why this is safe
- `save_slot` returns `(False, 0)` — the slot cache is not saved to disk
- Cache is not critical: on restart, meta files are recreated via `reconcile_meta()`
- A warning log is emitted so the issue can be tracked

---

## 4. Problem: discover_models returned empty list

### What happened
llama.cpp router mode returns all models with `status.value == "unloaded"` (all 137 models).
Previously, `discover_models` filtered to only `status.value == "loaded"` → empty list → "no models discovered".

### Solution
**File:** `llama_client.py` → `discover_models()`

```python
# Before:
if status_obj.get("value") != "loaded":
    continue

# After: all models are returned; n_ctx is only extracted from loaded models
if status.get("value") == "loaded":
    # extract n_ctx from args / loaded_info
else:
    n_ctx = DEFAULT_N_CTX  # 16384
```

### Why this works
- llama.cpp supports **on-demand loading**: when the first request arrives for an unloaded model, the backend loads it automatically
- `BackendManager.refresh_slot_counts()` sets 1 slot for unloaded models (see below)

---

## 5. Problem: "all slots busy" for unloaded models

### What happened
`refresh_slot_counts()` queried `/slots` for each discovered model.
For unloaded models, `/slots` returned an empty list → 0 slots → "all slots busy".

### Solution
**File:** `backend_manager.py` → `refresh_slot_counts()`

```python
if slots and isinstance(slots, list):
    n_slots = len(slots)
else:
    # Model discovered but not loaded — default to 1 slot
    # Backend will load it on demand
    if backend_key not in slot_counts:
        slot_counts[backend_key] = {}
    slot_counts[backend_key][canonical_name] = 1
    log.info(
        "Model '%s' on backend '%s' has 1 slot (unloaded, will load on demand)",
        canonical_name,
        backend_key,
    )
    self._refresh_state[refresh_key] = (now, True)
    refreshed_any = True
```

### Why 1 slot
- Sufficient for the first request to an unloaded model
- The backend will load the model and create slots automatically
- On the next `refresh_slot_counts`, the real slot count will be pulled

---

## 6. Backend API Key Configuration

### What happened
The API key was hardcoded or had to be passed through the BACKENDS JSON.

### Solution
**File:** `config.py`

```python
# Default API key for all backends (can be overridden per-backend in BACKENDS config)
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")
```

**File:** `backend_manager.py`

```python
api_key = be.get("api_key") or BACKEND_API_KEY
client = LlamaClient(url, api_key=api_key)
```

### Priority order
1. `api_key` inside each object in the `BACKENDS` JSON (per-backend override)
2. `BACKEND_API_KEY` env var (global for all backends)
3. `""` (empty string, if nothing is set)

### Usage examples
```bash
# Option 1: global key
BACKEND_API_KEY=oryucyShimkeurlucbajAxnamcaiHad7 proxycache.py

# Option 2: key in BACKENDS JSON
BACKENDS='[{"url":"http://127.0.0.1:8000","api_key":"oryucyShimkeurlucbajAxnamcaiHad7"}]' proxycache.py

# Option 3: both (BACKENDS api_key overrides global)
BACKEND_API_KEY=default proxycache.py
# BACKENDS='[{"url":"http://127.0.0.1:8000","api_key":"override"}]'
```

---

## 7. Request Flow Diagram

```
Client → proxycache:8081 → BackendManager → LlamaClient → llama.cpp:8000
                                                              ↑
                                                         x-api-key: oryucy...
```

### Full request path
1. `app.py` receives request from client
2. `BackendManager` finds the client for the backend (with api_key)
3. `LlamaClient` adds `x-api-key` via `_auth_headers()`
4. Request goes to llama.cpp with authentication
5. Response flows back

### Background calls (also authenticated)
- `refresh_slot_counts()` → `client.get_slots_info()` → `self._auth_headers()`
- `_get_child_slots()` → `self.client.get(url, headers=self._auth_headers())`
- `discover_models()` → `self.client.get("/models", headers=self._auth_headers())`

---

## 8. File Structure (Current State)

| File | Role | Changes |
|------|------|---------|
| `config.py` | Configuration | +`BACKEND_API_KEY` env var |
| `llama_client.py` | HTTP client | +`api_key`, `_auth_headers()`, discover all models |
| `backend_manager.py` | Backend management | +`api_key` from config, default 1 slot |
| `app.py` | FastAPI + routes | Passes `api_key` and `model_name` |
| `proxycache.py` | Entry point | No changes |
| `slot_manager.py` | Slot management | No changes |
| `hashing.py` | Prompt hashing | No changes |

---

## 9. Run Commands

```bash
# Start with API key
env PORT=8081 \
    BACKEND_API_KEY=oryucyShimkeurlucbajAxnamcaiHad7 \
    BACKENDS='[{"url":"http://127.0.0.1:8000"}]' \
    python proxycache.py

# Test chat completion
curl -X POST http://127.0.0.1:8081/v1/chat/completions \
  -H "x-api-key: oryucyShimkeurlucbajAxnamcaiHad7" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen-Heretic-MTP-BPW5.75-192K","messages":[{"role":"user","content":"Hi"}]}'

# Check slots
curl http://127.0.0.1:8081/slots
```

---

## 10. Potential Future Issues

### If the backend starts requiring the key on GET /slots/{id}
Add `self._auth_headers()` to `get_slot_status()`:
```python
async def get_slot_status(self, slot_id: int) -> Optional[dict]:
    resp = await self.client.get(f"/slots/{slot_id}", headers=self._auth_headers())
```

### If the backend changes the /models response format
`discover_models()` parses `data[].status.args` to extract `--port`, `-ctx`, `-c`, `--ctx-size`.
If the format changes, update the parsing logic.

### If the backend stops supporting on-demand loading
Change `refresh_slot_counts()` — instead of defaulting to 1 slot, wait for the model to load.

### If different keys are needed for different endpoints
Currently `_api_key` is used for all requests. If different keys are needed for `/slots` vs `/chat/completions`, add a second parameter to the constructor.

---

## 11. Verified Scenarios

| Scenario | Status |
|----------|--------|
| Streaming request to Darwin-BPW3.75-128K | ✅ |
| Non-streaming request to Darwin-BPW3.75-128K | ✅ |
| Streaming request to Qwen-Heretic-MTP-BPW5.75-192K | ✅ |
| Non-streaming request to Qwen-Heretic-MTP-BPW5.75-192K | ✅ |
| discover_models for unloaded models | ✅ |
| refresh_slot_counts for unloaded models | ✅ |
| save_slot with 501 Not Implemented | ✅ (graceful) |
