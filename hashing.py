# hashing.py

# -*- coding: utf-8 -*-

"""
Raw-хэширование: raw_prefix без ролей, только контент, разделённый двойным переводом строки.

Блоки по 100 слов, LCP по полным SHA256-хэшам.
Key = sha256(model_id + "\\n" + raw_prefix), т.е. модель включена в ключ.

Метафайлы содержат:
- key
- model_id
- prefix_len
- wpb
- blocks
- timestamp
"""

import os
import json
import hashlib
import re
import time
import glob
import logging
from typing import List, Dict, Optional, Tuple

from config import META_DIR, WORDS_PER_BLOCK

META_SUFFIX = ".meta.json"

log = logging.getLogger(__name__)


def raw_prefix(messages: List[Dict]) -> str:
    parts = []
    for msg in messages or []:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content).strip()
        if content:
            parts.append(content)
    text = "\n\n".join(parts).strip()
    log.debug("raw_prefix len_chars=%d", len(text))
    return text


def words_from_text(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def block_hashes_from_text(text: str, wpb: int = WORDS_PER_BLOCK) -> List[str]:
    words = words_from_text(text)
    hashes: List[str] = []
    for i in range(0, len(words), wpb):
        block = " ".join(words[i:i + wpb])
        h = hashlib.sha256(block.encode("utf-8")).hexdigest()
        hashes.append(h)
    log.debug("block_hashes n_blocks=%d wpb=%d", len(hashes), wpb)
    return hashes


def lcp_blocks(blocks1: List[str], blocks2: List[str]) -> int:
    n = min(len(blocks1), len(blocks2))
    i = 0
    while i < n and blocks1[i] == blocks2[i]:
        i += 1
    return i


def prefix_key_sha256(text: str) -> str:
    """
    Базовая SHA256-обёртка; для кеша в неё передаём model_id + "\\n" + raw_prefix.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_all_meta() -> List[Dict]:
    files = sorted(
        glob.glob(os.path.join(META_DIR, "*" + META_SUFFIX)),
        key=os.path.getmtime,
        reverse=True,
    )
    metas: List[Dict] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fd:
                meta = json.load(fd)
                metas.append(meta)
        except Exception as e:
            log.warning("scan_meta_fail %s: %s", f, e)
    log.debug("scan_meta n_found=%d", len(metas))
    return metas


def find_best_restore_candidate(
    req_blocks: List[str],
    wpb: int,
    th: float,
    model_id: str,
) -> Optional[Tuple[str, float]]:
    """
    Ищет лучший кандидат для restore среди мета-файлов ТОЛЬКО текущей модели.

    Фильтруем по:
    - meta["model_id"] == model_id
    - meta["wpb"] == wpb
    """
    metas = scan_all_meta()
    best_key: Optional[str] = None
    best_ratio = 0.0

    for meta in metas:
        if meta.get("model_id") != model_id:
            continue
        if int(meta.get("wpb") or 0) != wpb:
            continue

        cand_blocks = meta.get("blocks") or []
        lcp = lcp_blocks(req_blocks, cand_blocks)
        denom = max(1, min(len(req_blocks), len(cand_blocks)))
        ratio = lcp / denom

        if ratio >= th and ratio > best_ratio:
            best_ratio = ratio
            best_key = meta.get("key")

    return (best_key, best_ratio) if best_key else None


def human_readable_time(timestamp: float) -> str:
    """Converts a Unix timestamp to a human-readable format."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def write_meta(
    key: str,
    prefix_text: str,
    blocks: List[str],
    wpb: int,
    model_id: str,
) -> None:
    """
    Записывает/перезаписывает meta-файл для key, привязанный к конкретной модели.
    """
    meta = {
        "key": key,
        "model_id": model_id,
        "prefix_len": len(prefix_text),
        "wpb": wpb,
        "blocks": blocks,
        "last_written": time.time(),
    }
    path = os.path.join(META_DIR, f"{key}{META_SUFFIX}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved cache for key %s (model: %s, %d blocks)", key[:16], model_id, len(blocks))


def reconcile_meta(meta_dir: str, cache_dir: str) -> int:
    """
    Scans all meta files and removes corrupted ones or orphans (meta files with no matching cache).
    Returns the count of files deleted.
    """
    deleted = 0
    meta_files = sorted(      glob.glob(os.path.join(meta_dir, "*" + META_SUFFIX)))

    for meta_path in meta_files:
        basename = os.path.basename(meta_path);
        #log.info("Checking meta file: %s", basename)
        
        cachename = basename.removesuffix(META_SUFFIX)
        #log.info("Cache filename: %s", cachename)

        # Check for corrupted meta files
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            log.warning("Removed corrupted meta file: %s", basename)
            try:
                os.remove(meta_path)
                deleted += 1
            except OSError:
                pass
            continue

        # Check for orphaned meta files (no matching cache on disk)
        if cache_dir and os.path.isdir(cache_dir):
            cache_path = os.path.join(cache_dir, cachename)
            #log.info("Looking for cache file: %s", cache_path)
            if not os.path.exists(cache_path):
                log.info("Removed orphan meta file (no matching cache): %s", basename)
                try:
                    os.remove(meta_path)
                    deleted += 1
                except OSError:
                    pass
    log.info("Finished reconciling meta state with llama cache dir state")
    return deleted


def _get_last_used_time(basename: str, meta_dir: str, cache_dir: str) -> float:
    """
    Determines the last-used timestamp for a cache file.
    Priority: last_read -> last_written -> timestamp -> mtime (filesystem fallback).
    """
    meta_path = os.path.join(meta_dir, f"{basename}{META_SUFFIX}")
    
    # Try to load from meta file
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Priority order for last-used timestamp
            for field in ("last_read", "last_written", "timestamp"):
                if field in meta:
                    return meta[field]
        except (json.JSONDecodeError, Exception):
            pass  # Corrupted meta, fall through to mtime
    
    # Fallback to filesystem mtime
    if cache_dir:
        cache_path = os.path.join(cache_dir, basename)
        if os.path.exists(cache_path):
            return os.path.getmtime(cache_path)
    
    return time.time()  # Ultimate fallback



