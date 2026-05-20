# slot_manager.py

# -*- coding: utf-8 -*-

"""
SlotManager: авто-обнаружение слотов, ring buffer для кеша, cleanup по триггеру сохранений.

- get_slot(): сначала свободный (ещё не использовался), иначе самый старый по времени.
- Для big: если есть restore_key — делаем restore на выбранный слот.
- Сохранение всегда после завершения запроса.
- Ring buffer: отслеживает размер кеша в памяти, удаляет старые файлы при превышении лимита.
- Cleanup: запускается после каждых 5 сохранений (мин. интервал 10 мин).
"""

import os
import time
import asyncio
import logging
from collections import deque
from typing import List, Tuple, Dict, Optional

from config import BACKENDS, META_DIR, CACHE_DIR, CACHE_MAX_AGE_HOURS, CACHE_MAX_SIZE_GB
import hashing as hs

log = logging.getLogger(__name__)

GSlot = Tuple[int, int]  # (backend_id, local_slot_id)

CACHE_CLEANUP_SAVE_INTERVAL = 5
CACHE_CLEANUP_MIN_INTERVAL_SECONDS = 600


class SlotManager:
    def __init__(self, discovered_slots: Dict[int, int]):
        self.backends = []
        total_slots = 0

        for be_id, n_slots in discovered_slots.items():
            self.backends.append({"id": be_id, "client": None, "n_slots": n_slots})
            total_slots += n_slots

        self._all_slots: List[GSlot] = [
            (be_id, s)
            for be_id, be in enumerate(self.backends)
            for s in range(be["n_slots"])
        ]

        self._last_used: Dict[GSlot, float] = {g: 0.0 for g in self._all_slots}
        self._locks: Dict[GSlot, asyncio.Lock] = {
            g: asyncio.Lock() for g in self._all_slots
        }

        # Ring buffer for cache size tracking
        self._cache_ring: deque = deque()  # (key, size_bytes)
        self._total_bytes: int = 0

        # Save-triggered cleanup tracking
        self._save_count: int = 0
        self._last_cleanup_time: float = 0.0

        log.info(
            "slot_manager n_backends=%d total_slots=%d",
            len(self.backends),
            total_slots,
        )

    def set_clients(self, clients: List):
        for i, client in enumerate(clients):
            self.backends[i]["client"] = client

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

    def _is_free(self, g: GSlot) -> bool:
        return self._last_used.get(g, 0.0) == 0.0

    def _get_free_or_oldest(self) -> Tuple[GSlot, asyncio.Lock]:
        free = [g for g in self._all_slots if self._is_free(g)]
        if free:
            g = free[0]
            return g, self._locks[g]

        g = sorted(self._all_slots, key=lambda x: self._last_used.get(x, 0.0))[0]
        return g, self._locks[g]

    async def acquire_for_request(
        self,
        restore_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Tuple[GSlot, asyncio.Lock, Optional[bool]]:
        g, lock = self._get_free_or_oldest()
        await lock.acquire()
        self._last_used[g] = time.time()
        restored: Optional[bool] = None
        if restore_key:
            client = self.backends[g[0]]["client"]
            restored = await client.restore_slot(g[1], restore_key, model_id)
            log.info(
                "restore_before_chat g=%s key=%s ok=%s",
                g,
                (restore_key[:16] if restore_key else None),
                restored,
            )
            if restored:
                hs.update_last_read(restore_key)
        return g, lock, restored

    async def refresh_slots(self):
        """Re-query slot counts from all backends. Handles increases/decreases."""
        for be in self.backends:
            be_id = be["id"]
            client = be.get("client")
            if not client:
                continue
            new_count = await client.get_slot_count()
            old_count = be["n_slots"]
            if new_count == old_count:
                continue
            if new_count > old_count:
                log.info("refresh_slots: be=%d slots %d->%d, adding %d",
                         be_id, old_count, new_count, new_count - old_count)
                for s in range(old_count, new_count):
                    g = (be_id, s)
                    self._all_slots.append(g)
                    self._last_used[g] = 0.0
                    self._locks[g] = asyncio.Lock()
            else:
                log.warning("refresh_slots: be=%d slots %d->%d, removing free slots",
                            be_id, old_count, new_count)
                self._all_slots = [g for g in self._all_slots
                                   if not (g[0] == be_id and g[1] >= new_count) or
                                   not self._is_free(g)]
            be["n_slots"] = new_count

    async def save_after(self, g: GSlot, key: str, model_id: Optional[str] = None) -> Tuple[bool, int]:
        client = self.backends[g[0]]["client"]
        ok, size = await client.save_slot(g[1], key, model_id)

        if ok and size > 0:
            self._cache_ring.append((key, size))
            self._total_bytes += size

            # Ring buffer eviction: remove oldest until under limit
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

    def release(self, g: GSlot):
        if self._locks[g].locked():
            self._locks[g].release()
            self._last_used[g] = 0.0
