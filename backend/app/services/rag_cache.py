"""Invalidate hybrid RAG caches after semantic index mutations."""
from __future__ import annotations

import logging
from typing import Any

from .semantic_pipeline import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

_registered_chain: Any = None


def register_rag_chain(chain: Any) -> None:
    global _registered_chain
    _registered_chain = chain


def invalidate_runtime_rag_cache(collection_name: str | None = None) -> None:
    coll = collection_name or DEFAULT_COLLECTION
    try:
        from retrieval.hybrid_retriever import invalidate_bm25_cache

        invalidate_bm25_cache(coll)
    except Exception as exc:
        logger.warning("[rag-cache] bm25 invalidate failed: %s", exc)

    chain = _registered_chain
    if chain is not None and hasattr(chain, "_active_ids_cache"):
        try:
            chain._active_ids_cache.clear()
            logger.info("[rag-cache] cleared active_ids cache")
        except Exception as exc:
            logger.warning("[rag-cache] active_ids clear failed: %s", exc)
