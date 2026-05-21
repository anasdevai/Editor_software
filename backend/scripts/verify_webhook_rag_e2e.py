"""
End-to-end runtime verification: webhooks → semantic pipeline → Qdrant → hybrid RAG query.
Run with backend on http://127.0.0.1:8001
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
SECRET = os.getenv("WEBHOOK_SECRET", "")
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
if SECRET:
    HEADERS["X-Webhook-Secret"] = SECRET


def _wait_job(client: httpx.Client, entity_type: str, entity_id: str, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(
            f"{BASE}/api/semantic/status",
            params={"entity_type": entity_type, "entity_id": entity_id},
        )
        r.raise_for_status()
        data = r.json()
        st = data.get("latest_job_status")
        if st in ("completed", "failed", "cancelled"):
            return data
        time.sleep(2)
    raise TimeoutError(f"semantic job for {entity_type}:{entity_id}")


def _qdrant_count(entity_id: str) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    coll = os.getenv("SEMANTIC_QDRANT_COLLECTION", "qa_semantic_chunks")
    c = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    pts, _ = c.scroll(
        collection_name=coll,
        scroll_filter=Filter(
            must=[FieldCondition(key="entity_id", match=MatchValue(value=entity_id))]
        ),
        limit=1,
        with_payload=False,
    )
    # count via scroll with limit is approximate; use count API
    from qdrant_client.models import CountRequest, Filter as F

    res = c.count(
        collection_name=coll,
        count_filter=F(must=[FieldCondition(key="entity_id", match=MatchValue(value=entity_id))]),
        exact=True,
    )
    return int(res.count)


def main() -> int:
    print("=== Webhook + Hybrid RAG E2E ===\n")
    with httpx.Client(timeout=120.0) as client:
        cfg = client.get(f"{BASE}/api/webhooks/config")
        print("webhook config:", cfg.status_code, cfg.json())
        health = client.get(f"{BASE}/api/webhooks/health")
        print("webhook health:", health.status_code, json.dumps(health.json(), indent=2)[:500])

        marker = f"E2E-WEBHOOK-{uuid.uuid4().hex[:8]}"
        create_body = {
            "title": f"Webhook E2E {marker}",
            "sop_number": f"WH-{marker[:6]}",
            "department": "QA",
        }
        r = client.post(f"{BASE}/api/editor/docs", json=create_body)
        if r.status_code not in (200, 201):
            print("create SOP failed:", r.status_code, r.text[:400])
            return 1
        sop = r.json()
        sop_id = sop.get("doc_id") or sop.get("id")
        print("created sop:", sop_id)

        time.sleep(1)
        status = _wait_job(client, "sop", sop_id)
        print("semantic after create:", status.get("qdrant_status"), status.get("latest_job_status"))
        n1 = _qdrant_count(sop_id)
        print("qdrant chunks after create:", n1)
        if n1 < 1:
            print("FAIL: no Qdrant points after create")
            return 1

        update_body = {
            "doc_json": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Unique retrieval marker {marker} for hybrid RAG verification.",
                            }
                        ],
                    }
                ],
            }
        }
        r2 = client.put(f"{BASE}/api/editor/docs/{sop_id}", json=update_body)
        print("update:", r2.status_code)
        _wait_job(client, "sop", sop_id)
        n2 = _qdrant_count(sop_id)
        print("qdrant chunks after update:", n2)

        q = client.post(
            f"{BASE}/api/ai/query",
            json={
                "question": f"What is the unique retrieval marker in this SOP?",
                "assistant_mode": "query",
                "assistant_context": {"active_sop_id": sop_id, "editor_surface_active": True},
            },
            timeout=180.0,
        )
        q.raise_for_status()
        ans = q.json()
        stats = ans.get("retrieval_stats") or {}
        print("rag source:", stats.get("source"), "total_docs:", stats.get("total_docs"))
        print("answer contains marker:", marker in (ans.get("answer") or ""))
        if stats.get("source") not in ("rag", None) and stats.get("total_docs", 0) == 0:
            print("WARN: low retrieval stats", stats)

        notify = client.post(
            f"{BASE}/api/webhooks/notify",
            headers=HEADERS,
            json={"event": "deleted", "entity_type": "sop", "entity_id": sop_id},
        )
        print("webhook delete notify:", notify.status_code, notify.json())

        r3 = client.delete(f"{BASE}/api/editor/docs/{sop_id}")
        print("http delete:", r3.status_code)
        time.sleep(3)
        n3 = _qdrant_count(sop_id)
        print("qdrant chunks after delete:", n3)
        if n3 > 0:
            print("FAIL: Qdrant still has points after delete")
            return 1

    print("\n=== E2E PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
