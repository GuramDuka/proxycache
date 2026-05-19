<img width="1000"  alt="image_" src="https://github.com/user-attachments/assets/0d966dde-f1d8-432f-bad0-aa79a5ccf396" />

# proxycache

An OpenAI-compatible proxy for `llama.cpp` that manages KV cache slots with smart slot selection, disk save/restore, and automatic cache cleanup. It is also compatible with `llama-swap` for multi-model routing.

## What this service is

This service is a proxy in front of `llama.cpp` that makes long‑context chat and IDE workflows much faster by managing `llama.cpp` slots, reusing cached context, and restoring saved caches from disk when needed. It speaks an OpenAI‑compatible Chat Completions API, so existing clients can connect without changes, including both streaming (SSE) and non‑stream responses depending on request settings.

## Why it’s needed

`llama.cpp` provides “slots,” each holding a conversation’s KV cache so repeated requests with the same or very similar prefix can skip recomputing the whole prompt and continue from the first mismatching token, which dramatically cuts latency for large prompts. In real teams the number of users can easily exceed the number of available slots (e.g., 20 developers but only 4 slots), so naive routing causes random slot reuse and cache overwrites that waste time and GPU/CPU cycles. This proxy solves that by steering requests to the right slot, saving evicted caches to disk, and restoring them on demand, so long prompts don’t need to be recomputed from scratch each time.

## Architecture & How it works

### Slot Selection & Balancing

- **Slots and heat**: When a request lands in a slot and its cache is valid for reuse, the slot is considered “hot,” and new requests won’t overwrite it if other options exist, preserving useful KV for future reuse.
- **Similarity matching**: The proxy computes a fast, word‑block prefix similarity between the incoming conversation and existing hot slots, and only reuses a hot slot if the similarity meets a single ratio threshold (e.g., 85% of the shorter sequence), otherwise it rejects reuse to avoid polluting the hot cache with a weakly related prompt.
- **Free and cold first**: If reuse is rejected, the proxy sends the request to a free slot or a cold slot (one not currently carrying a valuable hot cache), protecting high‑value contexts from accidental overwrites under load.
- **Oldest when full**: If there are no free or cold slots, the proxy picks the least‑recently used slot and saves its current KV cache to disk before assigning the new request, ensuring nothing valuable is lost when the pool is exhausted.
- **Concurrency safety**: Each slot is guarded with an async lock; if all are busy, the request waits for the first LRU slot to free, preventing race conditions and unintended cache overwrites during concurrent generation.

### llama-swap Compatibility

The proxy can operate in `llama-swap` mode, which is ideal for multi-model setups:
- Routes `/slots` API calls through the `/upstream/{model}/slots/` path.
- Reads the `model_id` from the request body instead of the `/v1/models` endpoint.
- Passes through `/v1/models` from the backend for proper model discovery.

### Cache Cleanup

The proxy includes an automatic periodic cleanup task that runs in the background, requiring no external cronjobs. It manages disk usage by:
- Deleting files older than a specified age (`CACHE_MAX_AGE_HOURS`).
- Ensuring the total cache size does not exceed a limit (`CACHE_MAX_SIZE_GB`).

### Operational Flow

1. Incoming request hits `proxycache`.
2. Prompt is hashed and compared against cached prefixes (LCP matching).
3. If a match is found: the proxy restores the KV cache from disk into a slot and skips the prefill.
4. If no match is found: normal inference proceeds, and the KV cache is saved to disk after the response completes.
5. Periodic cleanup removes old or excessive cache files.

### Save and restore from disk

`llama.cpp`’s HTTP server exposes slot save/restore; saving writes a cache file to the directory provided by `--slot-save-path`, and restore loads by file basename (e.g., `slotcache_<key>.bin`), which is exactly how this proxy persists and revives caches across requests and restarts. The proxy keeps small local `.meta` files describing cached prefixes for fast lookup, while `llama.cpp` owns the actual KV `.bin` files under `--slot-save-path` for correctness and performance.

## Quick Start

### 1. Start llama.cpp

Start `llama.cpp` with slots and a cache directory:

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8080 --swa-full
```

This enables the OpenAI‑compatible HTTP server, a pool of 4 slots, and a directory where slot KV caches are saved and restored by basename.

### 2. Run the proxy

```bash
# Clone and install dependencies
git clone https://github.com/dingausmwald/proxycache.git
cd proxycache
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run with default settings
python3 proxycache.py 

# Or via uvicorn
uvicorn app:app --host 0.0.0.0 --port 8081
```

Your clients should call the proxy’s `/v1/chat/completions` endpoint.

## Configuration

All configuration is handled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_URL` | `http://127.0.0.1:8000` | Backend URL (llama-swap or llama-server) |
| `N_SLOTS` | `2` | Number of slots per backend |
| `PORT` | `8081` | Port proxycache listens on |
| `BACKEND_MODE` | `llama-cpp` | `llama-cpp` or `llama-swap` |
| `META_DIR` | `./kv_meta` | Directory for `.meta.json` files |
| `BIG_THRESHOLD_WORDS` | `500` | Min words to trigger caching |
| `WORDS_PER_BLOCK` | `100` | Words per block for LCP matching |
| `LCP_TH` | `0.6` | LCP similarity threshold (0-1) |
| `REQUEST_TIMEOUT` | `600` | HTTP timeout in seconds |
| `MODEL_ID` | `llama.cpp` | Default model ID |
| `CACHE_DIR` | (empty) | llama.cpp cache dir (for cleanup) |
| `CACHE_MAX_AGE_HOURS` | `168` | Delete files older than this (0=disabled) |
| `CACHE_MAX_SIZE_GB` | `25` | Max total cache size (GB) |
| `CACHE_CLEANUP_INTERVAL_MINUTES` | `30` | Cleanup check interval (minutes) |

## Usage with llama-swap

Architecture:

```
Client (OpenWebUI, Kilo Code, etc.)
    |
    v
proxycache (:8081) - KV cache management
    |
    v
llama-swap (:9292) - model routing
    |
    v
llama-server (:PORT) - inference
```

Make sure your `llama-swap` model configs include `--slot-save-path`:

```yaml
models:
  "my-model":
    cmd: "llama-server -m model.gguf --slot-save-path /path/to/kv-cache ..."
```

Point your clients to `proxycache` instead of `llama-swap` directly.

## Systemd Service

Create `~/.config/systemd/user/proxycache.service`:

```ini
[Unit]
Description=ProxyCache for llama.cpp KV Cache Management
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/proxycache
Environment="LLAMA_URL=http://127.0.0.1:9292"
Environment="N_SLOTS=1"
Environment="META_DIR=/path/to/proxycache-meta"
Environment="PORT=5000"
Environment="BIG_THRESHOLD_WORDS=1500"
Environment="CACHE_DIR=/path/to/kv-cache"
Environment="CACHE_MAX_AGE_HOURS=0"
Environment="CACHE_MAX_SIZE_GB=100"
ExecStart=/path/to/proxycache/venv/bin/python proxycache.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now proxycache
```

## Parameters (Detailed)

- **LLAMA_SERVER_URL**: The llama.cpp server base URL, e.g., `http://127.0.0.1:8080`, which must expose the OpenAI‑compatible chat completions endpoint.
- **SLOTS_COUNT**: The number of server slots (should match `llama.cpp -np`) so the proxy can track and plan reuse/restore correctly under load.
- **SIMILARITY_MIN_RATIO**: One similarity threshold (e.g., `0.85`) controlling both active reuse and disk restore; if a match is below this ratio, the proxy will prefer a free/cold slot or restore instead of overwriting a hot slot.
- **MIN_PREFIX_* (chars/words/blocks)**: Requests below this size are treated as “small” and steered to free/cold/oldest slots to avoid disturbing valuable hot caches used by large, long‑running prompts.
- **LOCAL_META_DIR and --slot-save-path**: The proxy stores small `.meta` descriptors locally for fast candidate lookup, while `llama.cpp` reads/writes the real KV cache files under `--slot-save-path` using basename in the HTTP API.

## Why this boosts IDE and long‑context productivity

For 30–60k‑token contexts typical in project‑wide IDE assistants, recomputing a full prompt can take minutes, whereas restoring a previously cached context and continuing from the first mismatching token typically takes seconds on `llama.cpp`, dramatically improving iteration speed for large teams with limited slots.
