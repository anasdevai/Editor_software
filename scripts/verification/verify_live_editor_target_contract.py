"""Regression checks for fail-closed sidebar -> live editor target IDs."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from chatbot.assistant.context_intelligence import build_target_resolution, resolve_scope_from_message
from app.services.target_resolver_agent import (
    TargetResolverAgentError,
    _compact_items,
    _normalize_result,
    resolve_sop_target_with_deep_agent,
)
import chatbot.actions.runtime as action_runtime_module
from app.ai_routes import _deterministic_target_analysis


def _prep() -> dict:
    return {
        "assistant_context": {
            "editor_context_contract": {
                "schema_version": 1,
                "document_id": "sop-live-1",
                "targets": {
                    "sections": [
                        {"id": "section_1_0_40", "label": "1. Purpose", "type": "section"},
                        {"id": "section_2_40_110", "label": "2. Scope", "type": "section"},
                    ],
                    "tables": [
                        {
                            "id": "table_1_111_180",
                            "label": "Revision History",
                            "type": "table",
                            "owning_section": "Document Control",
                        }
                    ],
                    "paragraphs": [
                        {
                            "id": "paragraph_1_4_35",
                            "label": "P1 - 1. Purpose",
                            "type": "paragraph",
                            "order": 1,
                            "owning_section": "1. Purpose",
                        }
                    ],
                },
            }
        }
    }


def test_exact_section_label_returns_live_id() -> None:
    result = build_target_resolution(
        {"target_scope": "section", "section_hint": "Purpose"},
        prep=_prep(),
    )
    assert result["target_id"] == "section_1_0_40"
    assert result["target_type"] == "section"
    assert result["target_label"] == "1. Purpose"


def test_exact_table_label_returns_live_id() -> None:
    result = build_target_resolution(
        {"target_scope": "table", "section_hint": "Revision History"},
        prep=_prep(),
    )
    assert result["target_id"] == "table_1_111_180"
    assert result["target_type"] == "table"
    assert result["owning_section"] == "Document Control"


def test_unknown_label_never_invents_id() -> None:
    result = build_target_resolution(
        {"target_scope": "section", "section_hint": "Imaginary Safety Section"},
        prep=_prep(),
    )
    assert result["target_id"] is None
    assert result["target_type"] is None


def test_partial_label_does_not_silently_choose_section() -> None:
    result = build_target_resolution(
        {"target_scope": "section", "section_hint": "Pur"},
        prep=_prep(),
    )
    assert result["target_id"] is None
    scope = resolve_scope_from_message(
        "rewrite the Pur section",
        _prep()["assistant_context"]["editor_context_contract"]["targets"]["sections"],
    )
    assert not scope or scope.get("section_id") != "section_1_0_40"


def test_supported_typo_still_resolves_exact_live_section() -> None:
    sections = [
        {"id": "section-purpose", "label": "1. Purpose"},
        {"id": "section-zweck", "label": "2. Zweck"},
    ]
    scope = resolve_scope_from_message("rewrite the Zwect section", sections)
    assert scope and scope.get("section_id") == "section-zweck"


def test_ambiguous_single_candidate_is_not_auto_promoted() -> None:
    result = _normalize_result(
        {
            "target_type": "section",
            "target_id": None,
            "confidence": 0.7,
            "requires_clarification": True,
            "candidate_targets": [
                {"id": "section_1_0_40", "label": "1. Purpose", "target_type": "section"}
            ],
        },
        model_name="test-model",
        valid_targets={
            "section_1_0_40": {
                "id": "section_1_0_40",
                "label": "1. Purpose",
                "target_type": "section",
            }
        },
    )
    assert result["target_id"] is None
    assert result["requires_clarification"] is True
    assert len(result["candidate_targets"]) == 1


def test_line_number_resolves_to_real_paragraph_block_id() -> None:
    result = build_target_resolution(
        {"target_scope": "selection", "line_number": 1, "resolved_scope": {"level": "line"}},
        prep=_prep(),
    )
    assert result["target_id"] == "paragraph_1_4_35"
    assert result["target_type"] == "paragraph"


def test_document_and_selection_use_reserved_ids() -> None:
    full = build_target_resolution({"target_scope": "full_document"}, prep=_prep())
    selection = build_target_resolution({"target_scope": "selection"}, prep=_prep())
    assert full["target_id"] == "doc_root"
    assert selection["target_id"] == "selection"


def test_backend_never_manufactures_missing_editor_ids() -> None:
    compacted = _compact_items(
        [{"label": "Purpose", "target_type": "section", "from": 1, "to": 20}],
        20,
    )
    assert compacted == []


def test_backend_rejects_conflicting_live_ids_before_llm() -> None:
    try:
        resolve_sop_target_with_deep_agent(
            user_query="rewrite purpose",
            action="rewrite",
            sections=[{"id": "same-id", "label": "Purpose", "target_type": "section", "from": 1, "to": 20}],
            tables=[{"id": "same-id", "label": "Controls", "target_type": "table", "from": 30, "to": 60}],
            paragraphs=[],
            document_tree=[],
            selection={"empty": True},
            active_scope={},
            sop_metadata={},
            document_excerpt="Purpose",
        )
    except TargetResolverAgentError as exc:
        assert "collision" in str(exc).lower()
    else:
        raise AssertionError("Conflicting live target IDs must fail before LLM resolution")


def test_action_runtime_degrades_when_collection_validation_fails() -> None:
    originals = {
        "QdrantClient": action_runtime_module.QdrantClient,
        "get_embedder": action_runtime_module.get_embedder,
        "CrossEncoderReranker": action_runtime_module.CrossEncoderReranker,
        "build_action_runtime": action_runtime_module.build_action_runtime,
    }
    try:
        action_runtime_module.QdrantClient = lambda **_kwargs: object()
        action_runtime_module.get_embedder = lambda: object()
        action_runtime_module.CrossEncoderReranker = lambda **_kwargs: object()

        def fail_collection_validation(**_kwargs):
            raise RuntimeError("collection endpoint unavailable")

        action_runtime_module.build_action_runtime = fail_collection_validation
        runtime = action_runtime_module.create_action_runtime()
        assert runtime.retrieval_available is False
        assert runtime.retriever.invoke("anything") == []
        assert "RuntimeError" in runtime.retrieval_status
    finally:
        for name, value in originals.items():
            setattr(action_runtime_module, name, value)


def test_parent_child_and_table_collisions_resolve_by_specificity() -> None:
    sections = [
        {"id": "sec-parent", "label": "Abbreviations and Definitions Abbreviations", "target_type": "section", "text_excerpt": ""},
        {"id": "sec-abbr", "label": "Abbreviations", "target_type": "section", "text_excerpt": ""},
        {"id": "sec-def", "label": "Definitions", "target_type": "section", "text_excerpt": ""},
        {"id": "sec-scope", "label": "Scope", "target_type": "section", "text_excerpt": ""},
    ]
    tables = [
        {"id": "table-3", "label": "Table 3 - Abbreviations and Definitions Abbreviations", "target_type": "table", "owning_section": "Abbreviations and Definitions Abbreviations", "text_excerpt": ""},
        {"id": "table-4", "label": "Table 4 - Scope", "target_type": "table", "owning_section": "Scope", "text_excerpt": ""},
    ]

    definitions = _deterministic_target_analysis(
        user_query="rewrite the Definitions section",
        sections=sections,
        tables=tables,
        selection={"empty": True},
        active_scope={},
    )
    combined = _deterministic_target_analysis(
        user_query="rewrite the Abbreviations and Definitions section",
        sections=sections,
        tables=tables,
        selection={"empty": True},
        active_scope={},
    )
    scope_table = _deterministic_target_analysis(
        user_query="rewrite the Scope table",
        sections=sections,
        tables=tables,
        selection={"empty": True},
        active_scope={},
    )

    assert definitions and definitions["target_id"] == "sec-def"
    assert combined and combined["target_id"] == "sec-parent"
    assert scope_table and scope_table["target_id"] == "table-4"


if __name__ == "__main__":
    test_exact_section_label_returns_live_id()
    test_exact_table_label_returns_live_id()
    test_unknown_label_never_invents_id()
    test_partial_label_does_not_silently_choose_section()
    test_supported_typo_still_resolves_exact_live_section()
    test_ambiguous_single_candidate_is_not_auto_promoted()
    test_line_number_resolves_to_real_paragraph_block_id()
    test_document_and_selection_use_reserved_ids()
    test_backend_never_manufactures_missing_editor_ids()
    test_backend_rejects_conflicting_live_ids_before_llm()
    test_action_runtime_degrades_when_collection_validation_fails()
    test_parent_child_and_table_collisions_resolve_by_specificity()
    print("live editor target contract: PASS")
