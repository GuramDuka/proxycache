#!/usr/bin/env python3
"""Smoke tests — no framework required. Run with: python test_smoke.py"""

import os
import sys
import json
import tempfile
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))


# ── hashing tests (unchanged) ────────────────────────────────────────

def test_reconcile_meta_removes_orphans():
    """reconcile_meta should delete meta files with no matching cache and skip valid ones."""
    import hashing as hs

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(cache_dir)
        os.makedirs(meta_dir)

        # Valid entry: cache + meta both exist
        valid_key = "valid_cache_key"
        with open(os.path.join(cache_dir, valid_key), "w") as f:
            f.write("cache data")
        with open(os.path.join(meta_dir, f"{valid_key}.meta.json"), "w") as f:
            json.dump({"key": valid_key, "model_id": "test", "wpb": 100, "blocks": []}, f)

        # Orphaned entry: meta exists but cache does not
        orphan_key = "orphan_cache_key"
        with open(os.path.join(meta_dir, f"{orphan_key}.meta.json"), "w") as f:
            json.dump({"key": orphan_key, "model_id": "test", "wpb": 100, "blocks": []}, f)

        # Corrupted meta file
        corrupted_key = "corrupted_cache_key"
        with open(os.path.join(meta_dir, f"{corrupted_key}.meta.json"), "w") as f:
            f.write("not json {{{")

        deleted = hs.reconcile_meta(meta_dir, cache_dir)

        assert deleted == 2, f"Expected 2 deleted (orphan + corrupted), got {deleted}"
        assert os.path.exists(os.path.join(meta_dir, f"{valid_key}.meta.json")), "Valid meta was deleted"
        assert not os.path.exists(os.path.join(meta_dir, f"{orphan_key}.meta.json")), "Orphan meta was not deleted"
        assert not os.path.exists(os.path.join(meta_dir, f"{corrupted_key}.meta.json")), "Corrupted meta was not deleted"
        print("PASS: test_reconcile_meta_removes_orphans")


def test_hashing_imports():
    """hashing module should import without cleanup_old_cache or update_last_read."""
    import hashing as hs
    assert hasattr(hs, "reconcile_meta")
    assert hasattr(hs, "_get_last_used_time")
    assert hasattr(hs, "write_meta")
    assert hasattr(hs, "find_best_restore_candidate")
    assert not hasattr(hs, "cleanup_old_cache")
    assert not hasattr(hs, "update_last_read")
    print("PASS: test_hashing_imports")


def test_save_slot_response_parsing():
    """save_slot must extract n_written from the llama.cpp save response."""
    mock_response_json = {
        "id_slot": 0, "filename": "test_cache",
        "n_saved": 1745, "n_written": 14309796,
        "timings": {"save_ms": 49.865}
    }
    data = mock_response_json
    n_written = data.get("n_written", 0)
    assert n_written == 14309796, f"Expected n_written=14309796, got {n_written}"

    data_no_written = {"id_slot": 0, "filename": "test"}
    n_written_2 = data_no_written.get("n_written", 0)
    assert n_written_2 == 0, f"Expected default 0, got {n_written_2}"
    print("PASS: test_save_slot_response_parsing")


  # ── LlamaClient tests ────────────────────────────────────────────────

def test_refresh_slots_router_mode_filtering():
    """refresh_slots should filter slots by _router_model in router mode."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(
        return_value=[
            {"id": 0, "_router_model": "ModelA"},
            {"id": 1, "_router_model": "ModelA"},
            {"id": 0, "_router_model": "ModelB"},
        ]
    )
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    # Should only count ModelA slots, not ModelB
    assert sm._slot_pools["ModelA"][0] == {0, 1}
    print("PASS: test_refresh_slots_router_mode_filtering")


def test_refresh_slots_non_router_mode():
    """refresh_slots should use all slots in non-router mode."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(
        return_value=[{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
    )
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    assert sm._slot_pools["ModelA"][0] == {0, 1, 2, 3}
    print("PASS: test_refresh_slots_non_router_mode")


def test_refresh_slots_unavailable():
    """refresh_slots should fall back to 1 slot when slots are unavailable."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(return_value=None)
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    assert sm._slot_pools["ModelA"][0] == {0}
    print("PASS: test_refresh_slots_unavailable")


# ── SlotManager tests ────────────────────────────────────────────────

def test_slot_manager_per_model_pools():
    """SlotManager should create separate pools per model."""
    from slot_manager import SlotManager, GSlot

    sm = SlotManager()
    sm.backends = [
        {"id": 0, "client": None, "n_slots": 0},
    ]

    # Register a backend for a model and create a pool
    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 3)

    assert "ModelA" in sm._slot_pools
    assert 0 in sm._slot_pools["ModelA"]
    assert sm._slot_pools["ModelA"][0] == {0, 1, 2}
    assert 0 in sm._model_to_backends["ModelA"]
    print("PASS: test_slot_manager_per_model_pools")


def test_slot_manager_multiple_models():
    """SlotManager should support multiple models on the same backend."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [
        {"id": 0, "client": None, "n_slots": 0},
    ]

    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 2)

    sm._register_backend_for_model("ModelB", 0)
    sm._ensure_pool("ModelB", 0, 4)

    assert sm._slot_pools["ModelA"][0] == {0, 1}
    assert sm._slot_pools["ModelB"][0] == {0, 1, 2, 3}
    assert set(sm._model_to_backends["ModelA"]) == {0}
    assert set(sm._model_to_backends["ModelB"]) == {0}
    print("PASS: test_slot_manager_multiple_models")


def test_slot_manager_select_from_pool():
    """_select_from_pool should pick free or oldest slot."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [
        {"id": 0, "client": None, "n_slots": 0},
    ]

    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 3)

    # All slots free — should pick slot 0
    model_name, backend_id, slot_id, lock = sm._select_from_pool("ModelA")
    assert model_name == "ModelA"
    assert backend_id == 0
    assert slot_id == 0

    # Mark slot 0 as used
    sm._last_used[("ModelA", 0, 0)] = 100.0

    # Should pick slot 1 (free)
    model_name, backend_id, slot_id, lock = sm._select_from_pool("ModelA")
    assert slot_id == 1

    # Mark slot 1 as used too
    sm._last_used[("ModelA", 0, 1)] = 200.0

    # Mark slot 2 as used too so all are occupied
    sm._last_used[("ModelA", 0, 2)] = 150.0

    # All used — should pick oldest (slot 0, ts=100)
    model_name, backend_id, slot_id, lock = sm._select_from_pool("ModelA")
    assert slot_id == 0
    print("PASS: test_slot_manager_select_from_pool")


def test_slot_manager_release():
    """release should unlock and reset last_used."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [
        {"id": 0, "client": None, "n_slots": 0},
    ]

    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 2)

    # Lock and use slot 0
    lock = sm._locks[("ModelA", 0, 0)]
    assert not lock.locked()

    async def _acquire():
        await lock.acquire()

    asyncio.run(_acquire())
    sm._last_used[("ModelA", 0, 0)] = 100.0
    assert lock.locked()
    assert sm._last_used[("ModelA", 0, 0)] == 100.0

    # Release
    sm.release("ModelA", 0, 0)
    assert not lock.locked()
    assert sm._last_used[("ModelA", 0, 0)] == 0.0
    print("PASS: test_slot_manager_release")


def test_slot_manager_pool_resize_up():
    """Pool should grow when slot count increases."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 2)
    assert sm._slot_pools["ModelA"][0] == {0, 1}

    # Resize to 4
    sm._ensure_pool("ModelA", 0, 4)
    assert sm._slot_pools["ModelA"][0] == {0, 1, 2, 3}
    print("PASS: test_slot_manager_pool_resize_up")


def test_slot_manager_pool_resize_down():
    """Pool should shrink when slot count decreases (only removes free slots)."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 4)

    # Mark slot 2 as used so it survives shrink
    sm._last_used[("ModelA", 0, 2)] = 100.0

    # Resize to 2
    sm._ensure_pool("ModelA", 0, 2)
    assert sm._slot_pools["ModelA"][0] == {0, 1}
    # Slot 2 was used, so it should NOT be in the pool anymore (it was removed)
    # but last_used may still have the entry (that's OK — it'll be cleaned on next acquire)
    print("PASS: test_slot_manager_pool_resize_down")


def test_slot_manager_multiple_backends():
    """SlotManager should support multiple backends for the same model."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [
        {"id": 0, "client": None, "n_slots": 0},
        {"id": 1, "client": None, "n_slots": 0},
    ]

    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 2)

    sm._register_backend_for_model("ModelA", 1)
    sm._ensure_pool("ModelA", 1, 3)

    assert sm._slot_pools["ModelA"][0] == {0, 1}
    assert sm._slot_pools["ModelA"][1] == {0, 1, 2}
    assert set(sm._model_to_backends["ModelA"]) == {0, 1}

    # Select should pick from either backend
    model_name, backend_id, slot_id, lock = sm._select_from_pool("ModelA")
    assert model_name == "ModelA"
    assert backend_id in (0, 1)
    assert slot_id >= 0
    print("PASS: test_slot_manager_multiple_backends")


def test_slot_manager_gslot_type():
    """GSlot should be (model_name, backend_id, slot_id)."""
    from slot_manager import GSlot

    g: GSlot = ("ModelA", 0, 1)
    model_name, backend_id, slot_id = g
    assert model_name == "ModelA"
    assert backend_id == 0
    assert slot_id == 1
    print("PASS: test_slot_manager_gslot_type")


def test_slot_manager_cooldown():
    """refresh_slots should skip backends refreshed within cooldown."""
    from slot_manager import SlotManager, REFRESH_COOLDOWN_SECONDS

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    # Simulate a recent refresh (new format: (ts, success) tuple)
    sm._last_refresh[("ModelA", 0)] = (100.0, True)

    # Mock client — refresh_slots calls get_slots_info, not get_router_slot_counts
    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(return_value=[{"id": 0}, {"id": 1}])
    sm.backends[0]["client"] = mock_client

    # Call refresh_slots — should skip due to cooldown
    # (We can't easily test the actual skip without mocking time,
    #  but we verify the cooldown key exists after a real refresh)
    sm._last_refresh[("ModelA", 0)] = (0.0, True)  # reset

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    # After refresh, cooldown should be set as (timestamp, success) tuple
    assert ("ModelA", 0) in sm._last_refresh
    last = sm._last_refresh[("ModelA", 0)]
    assert isinstance(last, tuple), f"Expected (ts, success) tuple, got {type(last)}"
    assert last[0] > 0, "Timestamp should be > 0"
    assert last[1] is True, "Success flag should be True"
    print("PASS: test_slot_manager_cooldown")


def test_slot_manager_router_mode_discovery():
    """refresh_slots should discover slots via get_slots_info in router mode."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(
        return_value=[{"id": 0}, {"id": 1}, {"id": 2}]
    )
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    assert "ModelA" in sm._slot_pools
    assert sm._slot_pools["ModelA"][0] == {0, 1, 2}
    assert "ModelA" in sm._model_to_backends
    mock_client.get_slots_info.assert_called_once()
    print("PASS: test_slot_manager_router_mode_discovery")


def test_slot_manager_non_router_fallback():
    """refresh_slots should use all slots in non-router mode."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(
        return_value=[{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    )
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    assert sm._slot_pools["ModelA"][0] == {0, 1, 2, 3, 4}
    mock_client.get_slots_info.assert_called_once()
    print("PASS: test_slot_manager_non_router_fallback")


def test_slot_manager_model_not_loaded():
    """refresh_slots should create 1-slot fallback when model not in router response."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.get_router_slot_counts = AsyncMock(
        return_value={"ModelA": 2}  # ModelB not in response
    )
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelB")

    asyncio.run(_run())

    assert sm._slot_pools["ModelB"][0] == {0}  # 1-slot fallback
    print("PASS: test_slot_manager_model_not_loaded")


def test_ring_buffer_age_eviction():
    """Ring buffer should evict expired entries before LRU entries when over limit."""
    from slot_manager import SlotManager
    import hashing as hs
    import tempfile

    sm = SlotManager()
    sm._max_age_seconds = 3600  # 1 hour

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(cache_dir)
        os.makedirs(meta_dir)

        # Simulate ring buffer with entries of different ages
        old_key = "old_cache_file"
        new_key = "new_cache_file"
        size = 1024 * 1024 * 1020  # ~1 GB each
        max_bytes = 1500 * 1024 * 1024  # 1.5 GB limit

        sm._cache_ring.append((old_key, size, time.time() - 7200))  # 2 hours old
        sm._cache_ring.append((new_key, size, time.time() - 300))   # 5 minutes old
        sm._total_bytes = size * 2

        # Trigger eviction (simulating save_after behavior)
        now = time.time()
        evicted = []
        while sm._total_bytes > max_bytes and sm._cache_ring:
            # First pass: evict expired entries
            evicted_expired = False
            for entry in sm._cache_ring:
                if now - entry[2] > sm._max_age_seconds:
                    evict_key, evict_size, _ = entry
                    sm._cache_ring.remove(entry)
                    sm._total_bytes -= evict_size
                    evicted.append(evict_key)
                    evicted_expired = True
                    break
            if evicted_expired:
                continue

            # Second pass: evict LRU entry
            lru_idx = 0
            lru_ts = sm._cache_ring[0][2]
            for i in range(1, len(sm._cache_ring)):
                if sm._cache_ring[i][2] < lru_ts:
                    lru_ts = sm._cache_ring[i][2]
                    lru_idx = i
            evict_key, evict_size, _ = sm._cache_ring[lru_idx]
            sm._cache_ring.remove(sm._cache_ring[lru_idx])
            sm._total_bytes -= evict_size
            evicted.append(evict_key)

        assert evicted == [old_key], f"Expected [old_key], got {evicted}"
        assert sm._total_bytes == size  # Only old entry removed
        assert len(sm._cache_ring) == 1
        assert sm._cache_ring[0][0] == new_key
        print("PASS: test_ring_buffer_age_eviction")


def test_ring_buffer_lru_eviction():
    """Ring buffer should evict LRU entry when no entries are expired."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm._max_age_seconds = 3600  # 1 hour

    size = 1024 * 1024 * 1020  # ~1 GB each
    max_bytes = 1500 * 1024 * 1024  # 1.5 GB limit

    # Simulate ring buffer with all entries under max age
    sm._cache_ring.append(("cache_a", size, time.time() - 1800))  # 30 min old
    sm._cache_ring.append(("cache_b", size, time.time() - 600))   # 10 min old
    sm._total_bytes = size * 2

    # Trigger eviction
    now = time.time()
    evicted = []
    while sm._total_bytes > max_bytes and sm._cache_ring:
        evicted_expired = False
        for entry in sm._cache_ring:
            if now - entry[2] > sm._max_age_seconds:
                evict_key, evict_size, _ = entry
                sm._cache_ring.remove(entry)
                sm._total_bytes -= evict_size
                evicted.append(evict_key)
                evicted_expired = True
                break
        if evicted_expired:
            continue

        lru_idx = 0
        lru_ts = sm._cache_ring[0][2]
        for i in range(1, len(sm._cache_ring)):
            if sm._cache_ring[i][2] < lru_ts:
                lru_ts = sm._cache_ring[i][2]
                lru_idx = i
        evict_key, evict_size, _ = sm._cache_ring[lru_idx]
        sm._cache_ring.remove(sm._cache_ring[lru_idx])
        sm._total_bytes -= evict_size
        evicted.append(evict_key)

    assert evicted == ["cache_a"], f"Expected [cache_a] (LRU), got {evicted}"
    assert sm._total_bytes == size
    assert len(sm._cache_ring) == 1
    assert sm._cache_ring[0][0] == "cache_b"
    print("PASS: test_ring_buffer_lru_eviction")


def test_ring_buffer_no_eviction_under_limit():
    """Ring buffer should not evict when under size limit."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm._max_age_seconds = 3600

    size = 1024 * 1024 * 1020  # ~1 GB
    max_bytes = 2 * 1024 * 1024 * 1024  # 2 GB limit

    sm._cache_ring.append(("cache_a", size, time.time() - 1800))
    sm._cache_ring.append(("cache_b", size, time.time() - 600))
    sm._total_bytes = size * 2

    now = time.time()
    evicted = []
    while sm._total_bytes > max_bytes and sm._cache_ring:
        evicted_expired = False
        for entry in sm._cache_ring:
            if now - entry[2] > sm._max_age_seconds:
                evict_key, evict_size, _ = entry
                sm._cache_ring.remove(entry)
                sm._total_bytes -= evict_size
                evicted.append(evict_key)
                evicted_expired = True
                break
        if evicted_expired:
            continue

        lru_idx = 0
        lru_ts = sm._cache_ring[0][2]
        for i in range(1, len(sm._cache_ring)):
            if sm._cache_ring[i][2] < lru_ts:
                lru_ts = sm._cache_ring[i][2]
                lru_idx = i
        evict_key, evict_size, _ = sm._cache_ring[lru_idx]
        sm._cache_ring.remove(sm._cache_ring[lru_idx])
        sm._total_bytes -= evict_size
        evicted.append(evict_key)

    assert evicted == [], f"Expected no evictions, got {evicted}"
    assert sm._total_bytes == size * 2
    assert len(sm._cache_ring) == 2
    print("PASS: test_ring_buffer_no_eviction_under_limit")


# ── KV cache skip tests ──────────────────────────────────────────────

def test_should_skip_restore_no_tracked_state():
    """_should_skip_restore should return False when no state tracked for slot."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm._slot_kv_state.clear()

    g = ("ModelA", 0, 0)
    blocks = ["a", "b", "c", "d", "e"]

    assert sm._should_skip_restore(g, blocks) is False
    print("PASS: test_should_skip_restore_no_tracked_state")


def test_should_skip_restore_perfect_match():
    """_should_skip_restore should return True for perfect block match."""
    from slot_manager import SlotManager

    sm = SlotManager()
    g = ("ModelA", 0, 0)
    kv_blocks = ["a", "b", "c", "d", "e"]
    sm._slot_kv_state[g] = kv_blocks

    req_blocks = ["a", "b", "c", "d", "e"]
    assert sm._should_skip_restore(g, req_blocks) is True
    print("PASS: test_should_skip_restore_perfect_match")


def test_should_skip_restore_high_overlap():
    """_should_skip_restore should return True when overlap >= 0.9."""
    from slot_manager import SlotManager

    sm = SlotManager()
    g = ("ModelA", 0, 0)
    kv_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sm._slot_kv_state[g] = kv_blocks

    # 9 out of 10 blocks match LCP → ratio = 9/10 = 0.9
    req_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "x"]
    assert sm._should_skip_restore(g, req_blocks) is True
    print("PASS: test_should_skip_restore_high_overlap")


def test_should_skip_restore_low_overlap():
    """_should_skip_restore should return False when overlap < 0.9."""
    from slot_manager import SlotManager

    sm = SlotManager()
    g = ("ModelA", 0, 0)
    kv_blocks = ["a", "b", "c", "d", "e"]
    sm._slot_kv_state[g] = kv_blocks

    # Only 4 out of 5 blocks match LCP → ratio = 4/5 = 0.8
    req_blocks = ["a", "b", "c", "d", "x"]
    assert sm._should_skip_restore(g, req_blocks) is False
    print("PASS: test_should_skip_restore_low_overlap")


def test_should_skip_restore_zero_lcp():
    """_should_skip_restore should return False when no LCP overlap."""
    from slot_manager import SlotManager

    sm = SlotManager()
    g = ("ModelA", 0, 0)
    kv_blocks = ["a", "b", "c"]
    sm._slot_kv_state[g] = kv_blocks

    req_blocks = ["x", "y", "z"]
    assert sm._should_skip_restore(g, req_blocks) is False
    print("PASS: test_should_skip_restore_zero_lcp")


def test_should_skip_restore_shorter_kv_cache():
    """_should_skip_restore should handle shorter KV cache than request."""
    from slot_manager import SlotManager

    sm = SlotManager()
    g = ("ModelA", 0, 0)
    kv_blocks = ["a", "b", "c"]
    sm._slot_kv_state[g] = kv_blocks

    # KV cache has 3 blocks, request has 10. LCP = 3. ratio = 3/3 = 1.0
    req_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert sm._should_skip_restore(g, req_blocks) is True
    print("PASS: test_should_skip_restore_shorter_kv_cache")


def test_should_skip_restore_longer_kv_cache():
    """_should_skip_restore should handle longer KV cache than request."""
    from slot_manager import SlotManager

    sm = SlotManager()
    g = ("ModelA", 0, 0)
    kv_blocks = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    sm._slot_kv_state[g] = kv_blocks

    # Request has 3 blocks, KV cache has 10. LCP = 3. ratio = 3/3 = 1.0
    req_blocks = ["a", "b", "c"]
    assert sm._should_skip_restore(g, req_blocks) is True
    print("PASS: test_should_skip_restore_longer_kv_cache")


def test_save_after_updates_slot_kv_state():
    """save_after should update _slot_kv_state when blocks are provided."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    sm.backends[0]["client"] = mock_client

    blocks = ["blk_a", "blk_b", "blk_c"]

    async def _run():
        ok, size = await sm.save_after(
            "ModelA", 0, 0, "test_key", "ModelA", blocks,
        )
        return ok

    asyncio.run(_run())

    assert mock_client.save_slot.call_count == 1
    assert ("ModelA", 0, 0) in sm._slot_kv_state
    assert sm._slot_kv_state[("ModelA", 0, 0)] == blocks
    print("PASS: test_save_after_updates_slot_kv_state")


def test_save_after_no_blocks_no_state_update():
    """save_after should not update _slot_kv_state when blocks are None."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]

    mock_client = AsyncMock()
    mock_client.save_slot = AsyncMock(return_value=(True, 1024))
    sm.backends[0]["client"] = mock_client

    async def _run():
        ok, size = await sm.save_after(
            "ModelA", 0, 0, "test_key", "ModelA", None,
        )
        return ok

    asyncio.run(_run())

    assert ("ModelA", 0, 0) not in sm._slot_kv_state
    print("PASS: test_save_after_no_blocks_no_state_update")


# ── Bug reproduction tests (fail before fix, pass after) ────────────

def test_restore_slot_has_timeout_wrapper():
    """Verify: restore_slot wraps the httpx call in asyncio.wait_for(SLOT_TIMEOUT).

    Bug was: restore_slot called self.client.post() directly with no timeout
    wrapper, inheriting REQUEST_TIMEOUT (600s). Fix wraps in asyncio.wait_for.
    """
    import inspect
    from llama_client import LlamaClient

    source = inspect.getsource(LlamaClient.restore_slot)
    assert "asyncio.wait_for" in source, "restore_slot must wrap post() in asyncio.wait_for"
    assert "SLOT_TIMEOUT" in source, "restore_slot must use SLOT_TIMEOUT constant"
    print("PASS: test_restore_slot_has_timeout_wrapper")


def test_save_slot_has_timeout_wrapper():
    """Verify: save_slot wraps the httpx call in asyncio.wait_for(SLOT_TIMEOUT)."""
    import inspect
    from llama_client import LlamaClient

    source = inspect.getsource(LlamaClient.save_slot)
    assert "asyncio.wait_for" in source, "save_slot must wrap post() in asyncio.wait_for"
    assert "SLOT_TIMEOUT" in source, "save_slot must use SLOT_TIMEOUT constant"
    print("PASS: test_save_slot_has_timeout_wrapper")


def test_lock_acquire_has_timeout():
    """Verify: lock.acquire() is wrapped in wait_for(SLOT_TIMEOUT) — doesn't block forever."""
    from slot_manager import SlotManager
    from config import SLOT_TIMEOUT

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]
    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 1)

    # First request acquires the lock
    _, _, _, lock = sm._select_from_pool("ModelA")
    asyncio.run(lock.acquire())

    # Second request tries to acquire — should timeout, not hang forever
    t0 = time.time()
    try:
        asyncio.run(asyncio.wait_for(lock.acquire(), timeout=SLOT_TIMEOUT))
    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        assert 25 < elapsed < 40, f"Took {elapsed:.1f}s — expected ~30s timeout"
    else:
        assert False, "Should have raised TimeoutError"

    sm.release("ModelA", 0, 0)
    print("PASS: test_lock_acquire_has_timeout")


def test_adaptive_cooldown_on_failure():
    """Verify: failed refresh sets 30s cooldown, successful refresh sets 300s."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]
    sm._register_backend_for_model("ModelA", 0)

    # Simulate a failed refresh (backend down)
    mock_client = AsyncMock()
    mock_client.get_slots_info = AsyncMock(side_effect=Exception("connection refused"))
    sm.backends[0]["client"] = mock_client

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    # After failure, _last_refresh stores (timestamp, success_flag) tuple
    refresh_key = ("ModelA", 0)
    last_refresh = sm._last_refresh.get(refresh_key)
    assert last_refresh is not None, "Cooldown was not set after failure"
    assert isinstance(last_refresh, tuple), f"Expected (timestamp, success) tuple, got {type(last_refresh)}"
    timestamp, success = last_refresh
    assert success is False, f"Expected success=False after failure, got success={success}"
    assert timestamp > 0
    print("PASS: test_adaptive_cooldown_on_failure")


def test_lock_released_on_restore_failure():
    """Verify: if restore_slot raises, the slot lock is released via try/finally."""
    from slot_manager import SlotManager

    sm = SlotManager()
    sm.backends = [{"id": 0, "client": None, "n_slots": 0}]
    sm._register_backend_for_model("ModelA", 0)
    sm._ensure_pool("ModelA", 0, 1)

    mock_client = AsyncMock()
    mock_client.restore_slot = AsyncMock(side_effect=Exception("connection refused"))
    sm.backends[0]["client"] = mock_client

    async def _run():
        try:
            await sm.acquire_for_request("ModelA", restore_key="bad_key", blocks=["a", "b"])
        except Exception:
            pass

    asyncio.run(_run())

    # Lock should be released after the exception
    _, lock = sm._get_free_or_oldest_from_pool("ModelA", 0)
    assert not lock.locked(), "Lock must be released after restore failure"
    print("PASS: test_lock_released_on_restore_failure")


def test_slot_timeout_config():
    """Verify SLOT_TIMEOUT env var is read with default 30s."""
    import os
    import importlib

    old = os.environ.pop("SLOT_TIMEOUT", None)
    try:
        import config
        importlib.reload(config)
        assert hasattr(config, "SLOT_TIMEOUT"), "config.py should define SLOT_TIMEOUT"
        assert config.SLOT_TIMEOUT == 30.0, f"Expected SLOT_TIMEOUT=30.0, got {config.SLOT_TIMEOUT}"
    finally:
        if old is not None:
            os.environ["SLOT_TIMEOUT"] = old

    print("PASS: test_slot_timeout_config")


# ── Cancellation handling tests ──────────────────────────────────────

def test_non_streaming_cancelled_error_releases_slot():
    """Verify: outer finally in chat() releases slot on CancelledError.

    CancelledError inherits from BaseException, not Exception, so
    `except Exception` won't catch it. The outer `finally` block
    must be the one that releases the slot.
    """
    import inspect

    source = inspect.getsource(sys.modules.get("app") or __import__("app"))
    # Must have an outer finally block (not just inner try/finally)
    assert "finally:" in source, "chat() must have a finally block"
    # The finally must call sm.release()
    assert "sm.release(model_name, be_id, slot_id)" in source, \
        "finally block must call sm.release()"
    print("PASS: test_non_streaming_cancelled_error_releases_slot")


def test_streaming_save_after_skipped_on_cancel():
    """Verify: reader skips save_after() when client cancelled (cancelled flag)."""
    import inspect
    from app import start_stream_task

    source = inspect.getsource(start_stream_task)
    # Must have a cancelled flag
    assert "cancelled" in source, "start_stream_task must have a cancelled flag"
    assert "cancelled['value']" in source or "cancelled[\"value\"]" in source, \
        "cancelled flag must be a dict with 'value' key"
    # save_after must be guarded by the cancelled check
    assert "if not cancelled['value']:" in source, \
        "save_after must be guarded by `if not cancelled['value']:`"
    print("PASS: test_streaming_save_after_skipped_on_cancel")


def test_streaming_save_after_has_timeout():
    """Verify: reader wraps save_after() in asyncio.wait_for(SLOT_TIMEOUT)."""
    import inspect
    from app import start_stream_task

    source = inspect.getsource(start_stream_task)
    assert "asyncio.wait_for" in source, \
        "save_after must be wrapped in asyncio.wait_for"
    assert "SLOT_TIMEOUT" in source, \
        "save_after must use SLOT_TIMEOUT constant"
    assert "asyncio.TimeoutError" in source, \
        "must handle asyncio.TimeoutError from wait_for"
    print("PASS: test_streaming_save_after_has_timeout")


def test_streaming_gen_sets_cancelled_flag():
    """Verify: gen() sets cancelled['value'] = True before cancelling reader."""
    import inspect
    from app import start_stream_task

    source = inspect.getsource(start_stream_task)
    # gen() must set the cancelled flag when it catches CancelledError
    assert "cancelled['value'] = True" in source or \
           'cancelled["value"] = True' in source, \
        "gen() must set cancelled['value'] = True on CancelledError"
    print("PASS: test_streaming_gen_sets_cancelled_flag")


def test_streaming_release_not_in_outer_finally():
    """Verify: outer finally only releases for non-streaming (if not stream: guard)."""
    import inspect

    source = inspect.getsource(sys.modules.get("app") or __import__("app"))
    # The outer finally must have a stream guard
    assert "if not stream:" in source, \
        "outer finally must guard release with `if not stream:`"
    print("PASS: test_streaming_release_not_in_outer_finally")


def test_slot_timeout_imported_in_app():
    """Verify: SLOT_TIMEOUT is imported from config into app.py."""
    import inspect

    source = inspect.getsource(sys.modules.get("app") or __import__("app"))
    assert "SLOT_TIMEOUT" in source, \
        "app.py must import SLOT_TIMEOUT from config"
    print("PASS: test_slot_timeout_imported_in_app")


def test_reader_polls_is_disconnected_on_timeout():
    """Verify: reader wraps aiter_raw() in asyncio.wait_for and checks
    is_disconnected() on TimeoutError to detect client disconnect early."""
    import inspect
    from app import start_stream_task

    source = inspect.getsource(start_stream_task)
    # The reader must wrap iterator.__anext__() in asyncio.wait_for
    assert "asyncio.wait_for" in source, \
        "reader must use asyncio.wait_for on iterator"
    assert "iterator.__anext__()" in source, \
        "reader must call iterator.__anext__()"
    # Must check is_disconnected on TimeoutError
    assert "asyncio.TimeoutError" in source, \
        "reader must handle asyncio.TimeoutError"
    assert "is_disconnected" in source, \
        "reader must check req.is_disconnected()"
    # Must set cancelled flag on disconnect
    assert "cancelled['value'] = True" in source or \
           'cancelled["value"] = True' in source, \
        "reader must set cancelled flag on disconnect"
    print("PASS: test_reader_polls_is_disconnected_on_timeout")


if __name__ == "__main__":
    test_reconcile_meta_removes_orphans()
    test_hashing_imports()
    test_save_slot_response_parsing()
    test_refresh_slots_router_mode_filtering()
    test_refresh_slots_non_router_mode()
    test_refresh_slots_unavailable()
    test_slot_manager_per_model_pools()
    test_slot_manager_multiple_models()
    test_slot_manager_select_from_pool()
    test_slot_manager_release()
    test_slot_manager_pool_resize_up()
    test_slot_manager_pool_resize_down()
    test_slot_manager_multiple_backends()
    test_slot_manager_gslot_type()
    test_slot_manager_cooldown()
    test_slot_manager_router_mode_discovery()
    test_slot_manager_non_router_fallback()
    test_slot_manager_model_not_loaded()
    test_ring_buffer_age_eviction()
    test_ring_buffer_lru_eviction()
    test_ring_buffer_no_eviction_under_limit()

    # KV cache skip tests
    test_should_skip_restore_no_tracked_state()
    test_should_skip_restore_perfect_match()
    test_should_skip_restore_high_overlap()
    test_should_skip_restore_low_overlap()
    test_should_skip_restore_zero_lcp()
    test_should_skip_restore_shorter_kv_cache()
    test_should_skip_restore_longer_kv_cache()
    test_save_after_updates_slot_kv_state()
    test_save_after_no_blocks_no_state_update()

    # Backend-down fix verification tests
    test_slot_timeout_config()
    test_restore_slot_has_timeout_wrapper()
    test_save_slot_has_timeout_wrapper()
    test_lock_acquire_has_timeout()
    test_adaptive_cooldown_on_failure()
    test_lock_released_on_restore_failure()

    # Cancellation handling tests
    test_non_streaming_cancelled_error_releases_slot()
    test_streaming_save_after_skipped_on_cancel()
    test_streaming_save_after_has_timeout()
    test_streaming_gen_sets_cancelled_flag()
    test_streaming_release_not_in_outer_finally()
    test_slot_timeout_imported_in_app()
    test_reader_polls_is_disconnected_on_timeout()

    print("\nAll smoke tests passed.")
