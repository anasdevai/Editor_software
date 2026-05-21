"""
[rag-maintenance] One-off operational script:

  1) Purge stale Qdrant + knowledge_chunks + embedding_jobs (keeps SOPs/business data).
  2) Queue a fresh full reindex.
  3) Drain the embedding queue synchronously (use the same worker code path).

Run from `backend/`:

  python -m scripts.rag_clean_rebuild
"""

import os
import sys
import time

# Make backend root importable when invoked as `python -m scripts.rag_clean_rebuild`
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    from app.database import SessionLocal
    from app.models import EmbeddingJob
    from app.services.semantic_pipeline import SemanticPipelineService
    from sqlalchemy import asc

    print("[rag-maintenance] step 1/3 — purging stale RAG state", flush=True)
    purge_counts = SemanticPipelineService.purge_all_semantic_state(
        recreate_collection=True,
        clear_embedding_jobs=True,
        clear_source_references=True,
        clear_link_suggestions=False,
    )
    print(f"[rag-maintenance] purge_counts={purge_counts}", flush=True)

    print("[rag-maintenance] step 2/3 — enqueuing fresh full reindex", flush=True)
    enqueue_counts = SemanticPipelineService.queue_full_reindex()
    print(f"[rag-maintenance] enqueue_counts={enqueue_counts}", flush=True)

    print("[rag-maintenance] step 3/3 — draining embedding queue", flush=True)
    processed = 0
    failed = 0
    start = time.time()
    while True:
        s = SessionLocal()
        try:
            nxt = (
                s.query(EmbeddingJob)
                .filter(EmbeddingJob.status == "pending")
                .order_by(asc(EmbeddingJob.created_at))
                .first()
            )
            if not nxt:
                break
            jid = nxt.id
        finally:
            s.close()
        try:
            SemanticPipelineService.process_job(jid)
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[rag-maintenance] job {jid} failed: {exc}", flush=True)
    elapsed = time.time() - start

    print(
        f"[rag-maintenance] drain_done processed={processed} failed={failed} elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
