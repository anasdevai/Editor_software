"""Verify RAG feature contracts without requiring live Qdrant or an LLM.

The test stubs retrieval and generation, then exercises the production
SmartRAGChain path that builds context, citations, retrieval stats, active SOP
scope, and selected editor context.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from langchain_core.documents import Document  # noqa: E402

from chatbot.rag import rag_chain as rag_mod  # noqa: E402
from chatbot.rag.rag_chain import (  # noqa: E402
    SmartRAGChain,
    _build_unified_context,
    _editor_context_documents_from_prompt,
)


FULL_SOP_SENTINEL = "FULL_SOP_SHOULD_NOT_BE_INJECTED_SENTINEL"
RELEVANT_CHUNK_SENTINEL = "RELEVANT_CHUNK_SENTINEL_NETWORK_FIREWALL"
IRRELEVANT_CHUNK_SENTINEL = "IRRELEVANT_CHUNK_SENTINEL_EMERGENCY_ACCESS"
SELECTED_SECTION_SENTINEL = "SELECTED_SECTION_SENTINEL_CAPA_SCOPE"
ACTIVE_SOP_ID = "11111111-1111-1111-1111-111111111111"


class FakeRouter:
    def route(self, query: str) -> dict[str, Any]:
        return {"collections": ["sops"], "exact_filters": {}}


class FakeReranker:
    def rerank_top_n(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        relevant = [doc for doc in docs if RELEVANT_CHUNK_SENTINEL in doc.page_content]
        return relevant[:top_n]


class FakeRetriever:
    def __init__(self) -> None:
        self.metadata_filters: dict[str, Any] = {}
        self.category_filter: str | None = None

    def invoke(self, query: str) -> list[Document]:
        allowed = {
            str(x).lower()
            for x in (self.metadata_filters or {}).get("allowed_entity_ids", [])
            if str(x).strip()
        }
        docs = [
            Document(
                page_content=(
                    f"{RELEVANT_CHUNK_SENTINEL}: Firewall rule changes require approval, "
                    "logging, review, and traceable evidence."
                ),
                metadata={
                    "ref_number": "SOP-IT-002",
                    "entity_id": ACTIVE_SOP_ID,
                    "source_id": ACTIVE_SOP_ID,
                    "title": "Network Security / Firewall SOP",
                    "entity_type": "sop",
                    "chunk_index": 2,
                    "rerank_score": 0.93,
                },
            ),
            Document(
                page_content=f"{IRRELEVANT_CHUNK_SENTINEL}: Emergency access break-glass content.",
                metadata={
                    "ref_number": "SOP-IT-003",
                    "entity_id": "22222222-2222-2222-2222-222222222222",
                    "source_id": "22222222-2222-2222-2222-222222222222",
                    "title": "Emergency Access SOP",
                    "entity_type": "sop",
                    "chunk_index": 1,
                    "rerank_score": 0.12,
                },
            ),
        ]
        if allowed:
            docs = [
                doc
                for doc in docs
                if str((doc.metadata or {}).get("entity_id") or "").lower() in allowed
            ]
        return docs


class FakeFederated:
    def __init__(self) -> None:
        self.retrievers = {"sops": FakeRetriever()}
        self.reranker = FakeReranker()


class FakePromptPipeline:
    def __init__(self, owner: "FakePrompt") -> None:
        self.owner = owner

    def __or__(self, other: Any) -> "FakePromptPipeline":
        return self

    def invoke(self, payload: dict[str, Any]) -> str:
        context = str(payload.get("context") or "")
        self.owner.last_context = context
        assert RELEVANT_CHUNK_SENTINEL in context, "relevant chunk missing from LLM context"
        assert IRRELEVANT_CHUNK_SENTINEL not in context, "irrelevant chunk leaked into LLM context"
        assert FULL_SOP_SENTINEL not in context, "full SOP text leaked into RAG context"
        assert SELECTED_SECTION_SENTINEL in context, "selected section context missing"
        return (
            "[REASONING]\n"
            "The question asks about firewall controls in the active SOP.\n\n"
            "[ANSWER]\n"
            "Summary: Firewall changes require approval, logging, and review [SOP-IT-002].\n"
            "Details: The selected CAPA scope is visible from the active editor context [SOP-IT-002].\n"
            "Sources: [SOP-IT-002]\n\n"
            "[CONFIDENCE] HIGH\n\n"
            "---CITATIONS---\n"
            "[[SOP-IT-002|Network Security / Firewall SOP|sop|Firewall rule changes require approval and logging.]]\n"
            "---SUGGESTIONS---\n"
            '["Show linked CAPAs", "Explain firewall approval evidence", "Summarize this selected section"]'
        )


class FakePrompt:
    def __init__(self) -> None:
        self.last_context = ""

    def __or__(self, other: Any) -> FakePromptPipeline:
        return FakePromptPipeline(self)


class FakeLLM:
    pass


def check(name: str, condition: bool, detail: Any = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed: {detail}")
    print(f"PASS: {name}")


def main() -> int:
    active_scope = {
        "active_sop_id": ACTIVE_SOP_ID,
        "linked_sop_ids": [],
        "linked_deviation_ids": [],
        "linked_capa_ids": [],
        "linked_audit_ids": [],
        "linked_decision_ids": [],
    }
    live_context = f"""
LIVE_ASSISTANT_CONTEXT:
ACTIVE_SOP_ID: {ACTIVE_SOP_ID}
ACTIVE_SOP: SOP-IT-002 - Network Security / Firewall SOP
SCOPE=ACTIVE_SOP_ONLY
selected_section:
  label: CAPAs
  content: {SELECTED_SECTION_SENTINEL} linked CAPA content selected by the user.
full_text:
  SOP-IT-002 title and short metadata only.
  {FULL_SOP_SENTINEL}

RAG_HINTS:
- Use active SOP scope.
"""

    editor_docs = _editor_context_documents_from_prompt(live_context)
    check("Active SOP context synchronized", len(editor_docs) == 1)
    check("Selected section synchronization works", SELECTED_SECTION_SENTINEL in editor_docs[0].page_content)
    check("Editor context is bounded, not unbounded full SOP injection", len(editor_docs[0].page_content) <= 4000)

    ctx, raw_citations = _build_unified_context([
        Document(
            page_content=f"{RELEVANT_CHUNK_SENTINEL}: Approval and logging controls.",
            metadata={
                "ref_number": "SOP-IT-002",
                "title": "Network Security / Firewall SOP",
                "entity_type": "sop",
                "chunk_index": 2,
                "rerank_score": 0.93,
            },
        )
    ], "document")
    check("Chunk references visible", "[0]" in ctx and "SOP-IT-002" in ctx)
    check("SOP traceability visible", raw_citations[0]["ref"] == "SOP-IT-002")

    prompt = FakePrompt()
    chain = SmartRAGChain(FakeFederated())
    chain.router = FakeRouter()
    chain.prompt = prompt
    chain.llm = FakeLLM()

    original_get_cfg = rag_mod.get_local_llm_config
    rag_mod.get_local_llm_config = lambda: type(
        "Cfg",
        (),
        {"provider": "fake", "model": "fake-rag-test", "base_url": "memory://test"},
    )()
    try:
        result = chain.invoke(
            "What firewall controls are required for the current SOP?\n\n" + live_context,
            category="sops",
            chat_history=[],
            active_scope=active_scope,
        )
    finally:
        rag_mod.get_local_llm_config = original_get_cfg

    stats = result.get("retrieval_stats") or {}
    citations = result.get("citations") or []
    sources = [c.get("ref") for c in citations if isinstance(c, dict)]
    debug_rows = result.get("retrieval_debug") or []

    check("Relevant chunks retrieved only", RELEVANT_CHUNK_SENTINEL in prompt.last_context and IRRELEVANT_CHUNK_SENTINEL not in prompt.last_context)
    check("No full SOP injection", FULL_SOP_SENTINEL not in prompt.last_context)
    check("Retrieval quality validated", stats.get("total_docs") == 2 and stats.get("per_section", {}).get("sops") == 1, stats)
    check("Chunk references visible in response", "SOP-IT-002" in sources, citations)
    check("SOP traceability visible in debug", any(row.get("ref") == "SOP-IT-002" for row in debug_rows), debug_rows)
    check("Active SOP scoped retrieval", stats.get("pipeline", {}).get("router", {}).get("editor_scoped") is True, stats)
    check("Selected section included in RAG prompt", SELECTED_SECTION_SENTINEL in prompt.last_context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
