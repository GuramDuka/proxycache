# slot_manager.py

# -*- coding: utf-8 -*-

"""
SlotManager: per-model slot pools with lazy discovery and refresh cooldown.

- Slot pools keyed by model name, not backend index.
- refresh_slots() called inside acquire_for_request() with per-(model, backend) cooldown.
- Router mode: discovers slot counts via GET /models + child /slots.
- Non-router mode: uses GET /slots as before.
- Ring buffer: tracks cache size in memory, evicts old files when over limit.
- Cleanup: triggered after every 5 saves (min 10 min apart).
"""

import os
import time
import asyncio
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional

from config import (BACKENDS, META_DIR, CACHE_DIR, CACHE_MAX_AGE_HOURS,
                    CACHE_MAX_SIZE_GB)
import hashing as hs

log = logging.getLogger(__name__)

GSlot = Tuple[str, int, int]  # (model_name, backend_id, slot_id)

CACHE_CLEANUP_SAVE_INTERVAL = 5
CACHE_CLEANUP_MIN_INTERVAL_SECONDS = 600
REFRESH_COOLDOWN_SECONDS = 300


class SlotManager:
    def __init__(self):
        self.backends: List[Dict] = []
        self._model_to_backends: Dict[str, List[int]] = {}
        self._slot_pools: Dict[str, Dict[int, set]] = {}
        self._last_used: Dict[Tuple[str, int, int], float] = {}
        self._locks: Dict[Tuple[str, int, int], asyncio.Lock] = {}
        self._last_refresh: Dict[Tuple[str, int], float] = {}

        # Ring buffer for cache size tracking
        self._cache_ring: deque = deque()  # (key, size_bytes)
        self._total_bytes: int = 0

        # Save-triggered cleanup tracking
        self._save_count: int = 0
        self._last_cleanup_time: float = 0.0

        log.info("slot_manager n_backends=%d", len(self.backends))

    def set_clients(self, clients: List):
        self.backends = []
        for i, client in enumerate(clients):
            self.backends.append({"id": i, "client": client, "n_slots": 0})
        log.info("set_clients n_backends=%d", len(self.backends))

    def init_from_disk(self, cache_dir: str):
        """Populate ring buffer from existing cache files on disk."""
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        for f in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, f)
            if os.path.isfile(filepath):
                try:
                    size = os.stat(filepath).st_size
                    self._cache_ring.append((f, size))
                    self._total_bytes += size
                except OSError:
                    continue
        log.info("init_from_disk: %d cache files, %.1f GB total",
                 len(self._cache_ring), self._total_bytes / 1024**3)

    def _is_free(self, model_name: str, backend_id: int, slot_id: int) -> bool:
        return self._last_used.get((model_name, backend_id, slot_id), 0.0) == 0.0

    def _get_free_or_oldest_from_pool(
        self, model_name: str, backend_id: int
    ) -> Tuple[int, asyncio.Lock]:
        """Pick free slot or oldest (LRU) from a single backend's pool for a model."""
        pool = self._slot_pools.get(model_name, {}).get(backend_id)
        if not pool:
            raise RuntimeError(f"No pool for model={model_name} be={backend_id}")

        free = [s for s in pool if self._is_free(model_name, backend_id, s)]
        if free:
            return free[0], self._locks[(model_name, backend_id, free[0])]

        oldest = min(pool, key=lambda s: self._last_used.get((model_name, backend_id, s), 0.0))
        return oldest, self._locks[(model_name, backend_id, oldest)]

    def _select_from_pool(self, model_name: str) -> Tuple[str, int, int, asyncio.Lock]:
        """Pick the best backend + slot for a model (free or oldest LRU)."""
        best: Optional[Tuple[str, int, int, asyncio.Lock]] = None
        best_ts = -1.0

        backend_ids = self._model_to_backends.get(model_name, [])
        for backend_id in backend_ids:
            try:
                slot_id, lock = self._get_free_or_oldest_from_pool(model_name, backend_id)
            except RuntimeError:
                continue

            ts = self._last_used.get((model_name, backend_id, slot_id), 0.0)

            # Prefer free slots, then prefer oldest (lowest ts)
            is_free = ts == 0.0
            if is_free and best is not None:
                best_is_free = best[3] is not None and self._last_used.get((best[0], best[1], best[2]), 0.0) == 0.0
                if not best_is_free:
                    best = (model_name, backend_id, slot_id, lock)
                    continue
                else:
                    continue  # both free, keep first

            if not is_free and best is not None:
                best_is_free = self._last_used.get((best[0], best[1], best[2]), 0.0) == 0.0
                if best_is_free:
                    continue  # keep the free one

            if best is None or ts < best_ts:
                best = (model_name, backend_id, slot_id, lock)
                best_ts = ts

        if best is None:
            raise RuntimeError(f"No slots available for model={model_name}")

        return best

    def _ensure_pool(self, model_name: str, backend_id: int, n_slots: int):
        """Create or update a slot pool for (model_name, backend_id)."""
        if model_name not in self._slot_pools:
            self._slot_pools[model_name] = {}
        if backend_id not in self._slot_pools[model_name]:
            old_pool = set()
            new_pool = set(range(n_slots))
            # Add new slots
            for s in range(n_slots):
                if s not in old_pool:
                    g = (model_name, backend_id, s)
                    self._last_used[g] = 0.0
                    self._locks[g] = asyncio.Lock()
            self._slot_pools[model_name][backend_id] = new_pool
            log.info(
                "ensure_pool model=%s be=%d slots=%d",
                model_name, backend_id, n_slots,
            )
        else:
            old_count = len(self._slot_pools[model_name][backend_id])
            old_pool = self._slot_pools[model_name][backend_id]
            new_pool = set(range(n_slots))
            # Add new slots
            for s in new_pool - old_pool:
                g = (model_name, backend_id, s)
                self._last_used[g] = 0.0
                self._locks[g] = asyncio.Lock()
            # Remove old slots (only free ones)
            for s in old_pool - new_pool:
                if self._is_free(model_name, backend_id, s):
                    self._slot_pools[model_name][backend_id].discard(s)
                    self._last_used.pop((model_name, backend_id, s), None)
                    self._locks.pop((model_name, backend_id, s), None)
            self._slot_pools[model_name][backend_id] = new_pool
            log.info(
                "update_pool model=%s be=%d slots %d->%d",
                model_name, backend_id, old_count, n_slots,
            )

    def _register_backend_for_model(self, model_name: str, backend_id: int):
        """Register a backend as serving a model."""
        if model_name not in self._model_to_backends:
            self._model_to_backends[model_name] = []
        if backend_id not in self._model_to_backends[model_name]:
            self._model_to_backends[model_name].append(backend_id)

    async def refresh_slots(self, model_name: str):
        """Refresh slot counts for a model across all backends.

        Uses get_slots_info() which already handles both router and non-router modes.
        Skips backends refreshed within REFRESH_COOLDOWN_SECONDS.
        Falls back to 1 slot if discovery fails (model not loaded yet).
        """
        backend_ids = list(range(len(self.backends)))
        log.info(
            "refresh_slots model=%s n_backends=%d known_backends=%s",
            model_name, len(backend_ids),
            list(self._model_to_backends.get(model_name, [])),
        )

        if not backend_ids:
            log.error(
                "refresh_slots_no_backends model=%s — no backends configured, cannot serve request",
                model_name,
            )
            raise RuntimeError(f"No backends configured for model={model_name}")

        refreshed_any = False
        for backend_id in backend_ids:
            be = self.backends[backend_id]
            client = be.get("client")
            if not client:
                log.debug("refresh_slots_skip_no_client model=%s be=%d", model_name, backend_id)
                continue

            # Check cooldown
            refresh_key = (model_name, backend_id)
            now = time.time()
            last = self._last_refresh.get(refresh_key, 0.0)
            if now - last < REFRESH_COOLDOWN_SECONDS:
                log.debug("refresh_slots_cooldown model=%s be=%d last=%.1f", model_name, backend_id, last)
                continue

            # get_slots_info() handles both router and non-router modes internally
            try:
                slots = await client.get_slots_info(model_name)
            except Exception as e:
                log.warning(
                    "refresh_slots_get_slots_info_fail model=%s be=%d err=%s",
                    model_name, backend_id, e,
                )
                slots = None

            if slots and isinstance(slots, list):
                # Filter by model name if router mode (slots have _router_model field)
                if slots and isinstance(slots[0], dict) and "_router_model" in slots[0]:
                    model_slots = [s for s in slots if s.get("_router_model") == model_name]
                    n_slots = len(model_slots)
                    log.info(
                        "refresh_slots model=%s be=%d slots=%d (router)",
                        model_name, backend_id, n_slots,
                    )
                else:
                    # Non-router mode: all slots belong to this model
                    n_slots = len(slots)
                    log.info(
                        "refresh_slots model=%s be=%d slots=%d (non-router)",
                        model_name, backend_id, n_slots,
                    )
                self._register_backend_for_model(model_name, backend_id)
                self._ensure_pool(model_name, backend_id, n_slots)
                self._last_refresh[refresh_key] = now
                refreshed_any = True
            else:
                # Slots unavailable (model not loaded yet or discovery failed)
                log.warning(
                    "refresh_slots_model_not_loaded model=%s be=%d — 1 slot fallback",
                    model_name, backend_id,
                )
                self._register_backend_for_model(model_name, backend_id)
                self._ensure_pool(model_name, backend_id, 1)
                self._last_refresh[refresh_key] = now
                refreshed_any = True

        if not refreshed_any:
            log.warning(
                "refresh_slots_nothing_done model=%s — all backends skipped (cooldown or no client)",
                model_name,
            )
            # Ensure at least 1 slot exists so the request can proceed
            if model_name not in self._model_to_backends and backend_ids:
                first_be = backend_ids[0]
                self._register_backend_for_model(model_name, first_be)
                self._ensure_pool(model_name, first_be, 1)

    async def acquire_for_request(
        self,
        model_name: str,
        restore_key: Optional[str] = None,
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        # Refresh before selecting (with cooldown)
        await self.refresh_slots(model_name)

        # Select best backend + slot
        model_name_out, backend_id, slot_id, lock = self._select_from_pool(model_name)
        await lock.acquire()
        self._last_used[(model_name_out, backend_id, slot_id)] = time.time()

        # Restore if needed
        restored: Optional[bool] = None
        if restore_key:
            client = self.backends[backend_id]["client"]
            restored = await client.restore_slot(slot_id, restore_key, model_name)
            log.info(
                "restore_before_chat model=%s be=%d slot=%d ok=%s",
                model_name, backend_id, slot_id, restored,
            )
            if restored:
                hs.update_last_read(restore_key)

        return (model_name_out, backend_id, slot_id), lock, restored

    async def save_after(
        self,
        model_name: str,
        backend_id: int,
        slot_id: int,
        key: str,
        model_id: Optional[str] = None,
    ) -> Tuple[bool, int]:
        client = self.backends[backend_id]["client"]
        ok, size = await client.save_slot(slot_id, key, model_id)

        if ok and size > 0:
            self._cache_ring.append((key, size))
            self._total_bytes += size

            # Ring buffer eviction
            max_bytes = CACHE_MAX_SIZE_GB * 1024**3
            while self._total_bytes > max_bytes and self._cache_ring:
                old_key, old_size = self._cache_ring.popleft()
                self._total_bytes -= old_size
                cache_path = os.path.join(CACHE_DIR, old_key) if CACHE_DIR else None
                if cache_path and os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                        log.info("ring_evict: %s (%d bytes)", old_key[:16], old_size)
                    except OSError:
                        pass
                meta_path = os.path.join(META_DIR, f"{old_key}{hs.META_SUFFIX}")
                if os.path.exists(meta_path):
                    try:
                        os.remove(meta_path)
                    except OSError:
                        pass

            # Save-triggered cleanup
            self._save_count += 1
            now = time.time()
            if (self._save_count >= CACHE_CLEANUP_SAVE_INTERVAL and
                    now - self._last_cleanup_time >= CACHE_CLEANUP_MIN_INTERVAL_SECONDS):
                self._save_count = 0
                self._last_cleanup_time = now
                try:
                    hs.cleanup_old_cache(CACHE_DIR, META_DIR,
                                         CACHE_MAX_AGE_HOURS, CACHE_MAX_SIZE_GB)
                except Exception as e:
                    log.warning("save_triggered_cleanup_error: %s", e)

        return ok, size

    def release(self, model_name: str, backend_id: int, slot_id: int):
        g = (model_name, backend_id, slot_id)
        lock = self._locks.get(g)
        if lock and lock.locked():
            lock.release()
            self._last_used[g] = 0.0
