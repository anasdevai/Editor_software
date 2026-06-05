"""Runtime construction for SOP editor actions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from embeddings.embedder import get_embedder
from chatbot.llm.provider import create_chat_llm
from retrieval.hybrid_retriever import (
    HybridRetriever,
    rag_unified_enabled,
    unified_semantic_collection,
)
from retrieval.reranker import CrossEncoderReranker

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass
class ActionRuntime:
    client: QdrantClient
    embedder: object
    reranker: object  # Can be CrossEncoderReranker or fallback reranker
    retriever: HybridRetriever | object
    llm: object
    fallback_llm: object
    collection_name: str
    retrieval_available: bool = True
    retrieval_status: str = "ready"


class _NoContextRetriever:
    dense_weight = 0.5
    bm25_weight = 0.5

    def invoke(self, _query: str):
        return []


class _NoopReranker:
    def rerank_top_n(self, _query: str, docs, top_n: int):
        fallback = (docs or [])[:max(0, int(top_n))]
        for doc in fallback:
            if not doc.metadata:
                doc.metadata = {}
            if "rerank_score" not in doc.metadata:
                doc.metadata["rerank_score"] = doc.metadata.get("hybrid_score", 0.0)
        return fallback


def _get_action_llm(temperature: float = 0.1):
    return create_chat_llm(
        temperature=temperature,
        max_output_tokens=int(os.getenv("ACTION_MAX_OUTPUT_TOKENS") or "4096"),
        max_retries=1,
    )


def _get_action_fallback_llm(temperature: float = 0.1):
    return create_chat_llm(
        temperature=temperature,
        max_output_tokens=int(os.getenv("ACTION_MAX_OUTPUT_TOKENS") or "4096"),
        max_retries=0,
    )


def build_action_runtime(
    *,
    client: QdrantClient,
    embedder: object,
    reranker: object,
    collection_name: str | None = None,
) -> ActionRuntime:
    if collection_name:
        collection = collection_name
    elif rag_unified_enabled():
        collection = unified_semantic_collection()
    else:
        collection = os.getenv("COLLECTION_SOPS", "docs_sops")
    vectorstore = QdrantVectorStore(client=client, collection_name=collection, embedding=embedder)
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        client=client,
        collection_name=collection,
        dense_top_k=int(os.getenv("ACTION_DENSE_TOP_K", "8")),
        bm25_top_k=int(os.getenv("ACTION_BM25_TOP_K", "8")),
        final_top_k=int(os.getenv("ACTION_FINAL_TOP_K", "4")),
    )
    if rag_unified_enabled():
        retriever.category_filter = "sops"
    return ActionRuntime(
        client=client,
        embedder=embedder,
        reranker=reranker,
        retriever=retriever,
        llm=_get_action_llm(),
        fallback_llm=_get_action_fallback_llm(),
        collection_name=collection,
    )


def create_action_runtime() -> ActionRuntime:
    # 1. Connect to Qdrant and Embedder
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        embedder = get_embedder()
    except Exception as qdrant_exc:
        # Keep editor actions responsive even when Qdrant is unavailable.
        print(f"[startup] Qdrant/Embedder missing, using mock runtime: {qdrant_exc}", flush=True)
        return ActionRuntime(
            client=None,
            embedder=None,
            reranker=_NoopReranker(),
            retriever=_NoContextRetriever(),
            llm=_get_action_llm(),
            fallback_llm=_get_action_fallback_llm(),
            collection_name=os.getenv("COLLECTION_SOPS", "docs_sops"),
            retrieval_available=False,
            retrieval_status=f"unavailable: {type(qdrant_exc).__name__}",
        )

    # 2. Initialize Reranker with standalone try/except to protect retrieval path
    reranker = None
    try:
        reranker = CrossEncoderReranker(top_n=5)
    except Exception as reranker_exc:
        print(
            f"[startup] Action reranker cache missing, continuing with no-op reranker: {reranker_exc}",
            flush=True,
        )
        reranker = _NoopReranker()

    return build_action_runtime(client=client, embedder=embedder, reranker=reranker)
