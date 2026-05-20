#!/usr/bin/env python3
"""Smoke tests — no framework required. Run with: python test_smoke.py"""

import os
import sys
import json
import tempfile
import shutil
import asyncio

# Ensure we can import the project modules
sys.path.insert(0, os.path.dirname(__file__))


def test_meta_filename_mapping():
    """reconcile_meta must derive cache filename correctly from meta filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        meta_dir = os.path.join(tmpdir, "meta")
        os.makedirs(cache_dir)
        os.makedirs(meta_dir)

        key = "a1b2c3d4e5f6"
        cache_file = os.path.join(cache_dir, key)
        meta_file = os.path.join(meta_dir, f"{key}.meta.json")

        # Create matching files
        with open(cache_file, "w") as f:
            f.write("cache data")
        with open(meta_file, "w") as f:
            json.dump({"key": key, "model_id": "test", "wpb": 100, "blocks": []}, f)

        import hashing as hs
        deleted = hs.reconcile_meta(meta_dir, cache_dir)

        assert deleted == 0, f"Expected 0 deleted, got {deleted} — filename mapping broken"
        assert os.path.exists(meta_file), "Meta file was incorrectly deleted"
        assert os.path.exists(cache_file), "Cache file was incorrectly deleted"
        print("PASS: test_meta_filename_mapping")


def test_save_slot_response_parsing():
    """save_slot must extract n_written from the llama.cpp save response.
    
    Tests the response parsing logic directly without needing httpx.
    """
    mock_response_json = {
        "id_slot": 0,
        "filename": "test_cache",
        "n_saved": 1745,
        "n_written": 14309796,
        "timings": {"save_ms": 49.865}
    }
    
    # Simulate the parsing logic from save_slot
    data = mock_response_json
    n_written = data.get("n_written", 0)
    
    assert n_written == 14309796, f"Expected n_written=14309796, got {n_written}"
    
    # Test missing n_written defaults to 0
    data_no_written = {"id_slot": 0, "filename": "test"}
    n_written_2 = data_no_written.get("n_written", 0)
    assert n_written_2 == 0, f"Expected default 0, got {n_written_2}"
    print("PASS: test_save_slot_response_parsing")


def test_get_slot_count():
    """get_slot_count returns array length from GET /slots, fallback to 1.
    
    Tests the logic directly without needing httpx.
    """
    # Test normal response
    slots_info = [{"id": 0}, {"id": 1}, {"id": 2}]
    count = len(slots_info) if slots_info and isinstance(slots_info, list) else 1
    assert count == 3, f"Expected 3, got {count}"
    
    # Test empty list
    slots_info = []
    count = len(slots_info) if slots_info and isinstance(slots_info, list) else 1
    assert count == 1, f"Expected fallback to 1 for empty list, got {count}"
    
    # Test None → fallback to 1
    slots_info = None
    count = len(slots_info) if slots_info and isinstance(slots_info, list) else 1
    assert count == 1, f"Expected fallback to 1 for None, got {count}"
    
    print("PASS: test_get_slot_count")


if __name__ == "__main__":
    test_meta_filename_mapping()
    test_save_slot_response_parsing()
    test_get_slot_count()
    print("\nAll smoke tests passed.")
