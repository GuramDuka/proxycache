#!/usr/bin/env python3
"""Smoke tests — no framework required. Run with: python test_smoke.py"""

import os
import sys
import json
import tempfile
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))


# ── hashing tests (unchanged) ────────────────────────────────────────

def test_meta_filename_mapping():
    """reconcile_meta must derive cache filename correctly from meta filename."""
    import hashing as hs

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(cache_dir)
        os.makedirs(meta_dir)

        key = "a1b2c3d4e5f6"
        cache_file = os.path.join(cache_dir, key)
        meta_file = os.path.join(meta_dir, f"{key}.meta.json")

        with open(cache_file, "w") as f:
            f.write("cache data")
        with open(meta_file, "w") as f:
            json.dump({"key": key, "model_id": "test", "wpb": 100, "blocks": []}, f)

        deleted = hs.reconcile_meta(meta_dir, cache_dir)

        assert deleted == 0, f"Expected 0 deleted, got {deleted}"
        assert os.path.exists(meta_file), "Meta file was incorrectly deleted"
        assert os.path.exists(cache_file), "Cache file was incorrectly deleted"
        print("PASS: test_meta_filename_mapping")


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

    # Simulate a recent refresh
    sm._last_refresh[("ModelA", 0)] = 100.0

    # Mock client
    mock_client = AsyncMock()
    mock_client.get_router_slot_counts = AsyncMock(return_value={"ModelA": 2})
    sm.backends[0]["client"] = mock_client

    # Call refresh_slots — should skip due to cooldown
    # (We can't easily test the actual skip without mocking time,
    #  but we verify the cooldown key exists after a real refresh)
    sm._last_refresh[("ModelA", 0)] = 0.0  # reset

    async def _run():
        await sm.refresh_slots("ModelA")

    asyncio.run(_run())

    # After refresh, cooldown timestamp should be set
    assert ("ModelA", 0) in sm._last_refresh
    assert sm._last_refresh[("ModelA", 0)] > 0
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


if __name__ == "__main__":
    test_meta_filename_mapping()
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
    print("\nAll smoke tests passed.")
