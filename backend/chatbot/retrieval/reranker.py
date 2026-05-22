import math
import os
from pathlib import Path
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List


def _resolve_hf_cache_dir() -> str:
    configured = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("EMBEDDING_HF_CACHE_DIR")
    if configured:
        cache_dir = Path(configured).expanduser().resolve()
    else:
        cache_dir = (Path(__file__).resolve().parents[2] / ".hf-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    return str(cache_dir)


def _is_model_cached(cache_dir: str, model_name: str) -> bool:
    model_key = model_name.replace("/", "--")
    candidate_roots = [
        Path(cache_dir) / "hub" / f"models--{model_key}",
        Path(cache_dir) / f"models--{model_key}",
    ]
    for root in candidate_roots:
        snapshots_dir = root / "snapshots"
        if snapshots_dir.exists() and any(p.is_dir() for p in snapshots_dir.iterdir()):
            return True
    return False


class CrossEncoderReranker:
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.
    Scores each (query, passage) pair and returns top-N by relevance.
    """
    def __init__(self, top_n: int = 5):
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cache_dir = _resolve_hf_cache_dir()
        is_cached = _is_model_cached(cache_dir, model_name)
        if is_cached:
            print(f"[hf-cache] Cross-encoder loading from local cache: {cache_dir}", flush=True)
        else:
            print(f"[hf-cache] Cross-encoder cache miss, attempting download into: {cache_dir}", flush=True)
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            cache_folder=cache_dir,
            local_files_only=is_cached,
        )
        self.top_n = top_n

    def _score_and_filter(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """Core scoring logic shared by rerank() and rerank_top_n()."""
        if not docs:
            return []

        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        if ranked and float(ranked[0][1]) < -4.0:
            # Keep hybrid order; use a finite score so API JSON (citations) never sees -inf.
            fallback = docs[:top_n]
            for doc in fallback:
                doc.metadata["rerank_score"] = doc.metadata.get("hybrid_score", 0.0)
                doc.metadata["rerank_degraded"] = True
            return fallback

        top = ranked[:top_n]
        for doc, score in top:
            s = float(score)
            doc.metadata["rerank_score"] = s if math.isfinite(s) else 0.0

        # Filter out very low scores to avoid noise
        filtered = [(doc, score) for doc, score in top if score > -5.0]
        if not filtered:
            filtered = top

        return [doc for doc, _ in filtered]

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank using the default top_n set at construction time."""
        return self._score_and_filter(query, docs, self.top_n)

    def rerank_top_n(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """Rerank with a caller-specified top_n (used by FederatedRetriever per section)."""
        return self._score_and_filter(query, docs, top_n)
