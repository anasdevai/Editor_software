import asyncio
import json
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
import app.ai_routes as routes

# Mock LLM to simulate dynamic semantic resolution
class MockTargetResolutionLLM:
    def invoke(self, messages, *args, **kwargs):
        user_msg = next(msg[1] for msg in messages if msg[0] == "user")
        payload = json.loads(user_msg)
        query = str(payload.get("user_query") or "").lower()
        
        target_id = None
        target_type = "section"
        target_label = None
        
        # Multilingual mapping inside Mock LLM to simulate semantic target matching
        mapped_query = query
        if "geltungsbereich" in query:
            mapped_query += " scope"
        elif "anderungsverlauf" in query or "history" in query:
            mapped_query += " history"
            
        if "table" in mapped_query or "tablwe" in mapped_query:
            target_type = "table"
            tables = payload.get("table_index") or []
            if tables:
                tbl = next((t for t in tables if "history" in t.get("id") or "history" in str(t.get("owning_section")).lower() or "history" in str(t.get("label")).lower()), tables[0])
                target_id = tbl.get("id")
                target_label = tbl.get("label")
        else:
            target_type = "section"
            sections = payload.get("section_index") or []
            if sections:
                if "scope" in mapped_query:
                    sec = next((s for s in sections if "scope" in s.get("id") or "scope" in str(s.get("label")).lower()), sections[0])
                else:
                    sec = next((s for s in sections if "history" in s.get("id") or "history" in str(s.get("label")).lower()), sections[0])
                target_id = sec.get("id")
                target_label = sec.get("label")
                
        result = {
            "target_type": target_type,
            "target_id": target_id,
            "target_label": target_label,
            "owning_section": "",
            "confidence": 0.95,
            "requires_clarification": False,
            "candidate_targets": [],
            "reasoning_summary": "Resolved dynamically via Mock LLM."
        }
        
        class MockResponse:
            content = json.dumps(result)
            
        return MockResponse()

def fail_deep_agent(**_kwargs):
    raise routes.TargetResolverAgentError("skip deep agent in tests")

async def test_procedure_matching():
    routes.resolve_sop_target_with_deep_agent = fail_deep_agent
    routes.create_chat_llm = lambda *args, **kwargs: MockTargetResolutionLLM()
    routes.TARGET_DEEP_AGENT_TIMEOUT_SECONDS = 1
    routes.TARGET_LLM_FALLBACK_TIMEOUT_SECONDS = 1

    payload = {
        "user_query": "rewrite the procedure section",
        "action": "rewrite",
        "full_text": "4. Procedure\nOT Security monitors firewall rules.\n4.1 Procedure for deviation handling\nCAPA actions must be documented.",
        "document_excerpt": "4. Procedure section followed by 4.1 Procedure sub-section.",
        "selection": {"empty": True, "text": ""},
        "active_scope": {"sectionName": "Scope"},
        "section_index": [
            {"id": "sec-procedure", "type": "section", "label": "4. Procedure", "text": "OT Security monitors firewall rules."},
            {"id": "sec-sub-procedure", "type": "section", "label": "4.1 Procedure for deviation handling", "text": "CAPA actions must be documented."},
        ],
        "table_index": [],
        "paragraph_index": [],
        "document_tree": [
            {"id": "sec-procedure", "type": "section", "label": "4. Procedure"},
            {"id": "sec-sub-procedure", "type": "section", "label": "4.1 Procedure for deviation handling"},
        ],
    }

    result = await routes.analyze_sop_target(payload)
    print("Test Result JSON:")
    print(json.dumps(result, indent=2, sort_keys=True))
    
    # Assertions to ensure "4. Procedure" is picked with no clarification requirement
    assert result["target_type"] == "section"
    assert result["target_id"] == "sec-procedure"
    assert result["target_label"] == "4. Procedure"
    assert result["requires_clarification"] is False
    print("\n--- ALL TARGET RESOLUTION ASSERTIONS PASSED ---")

async def test_history_table_matching():
    routes.resolve_sop_target_with_deep_agent = fail_deep_agent
    routes.create_chat_llm = lambda *args, **kwargs: MockTargetResolutionLLM()
    routes.TARGET_DEEP_AGENT_TIMEOUT_SECONDS = 1
    routes.TARGET_LLM_FALLBACK_TIMEOUT_SECONDS = 1

    # Scenario: Both "Document History" (section) and "Document History" (table) exist.
    payload_base = {
        "action": "rewrite",
        "full_text": "Scope section\nTable 4 - Scope\nSome description.\nDocument History\nVersion 1.0",
        "document_excerpt": "Scope, Table 4 - Scope, Document History table",
        "selection": {"empty": True, "text": ""},
        "active_scope": {"sectionName": "Scope"},
        "section_index": [
            {"id": "sec-scope", "type": "section", "label": "Scope", "text": "Some scope."},
            {"id": "sec-history", "type": "section", "label": "Document History", "text": "History metadata."},
        ],
        "table_index": [
            {"id": "tbl-scope", "type": "table", "label": "Table 4 - Scope", "text": "Scope metadata."},
            {"id": "tbl-history", "type": "table", "label": "Table 2", "text": "Version 1.0 details.", "owning_section": "Document History"},
        ],
        "paragraph_index": [],
        "document_tree": [
            {"id": "sec-scope", "type": "section", "label": "Scope"},
            {"id": "tbl-scope", "type": "table", "label": "Table 4 - Scope"},
            {"id": "sec-history", "type": "section", "label": "Document History"},
            {"id": "tbl-history", "type": "table", "label": "Table 2", "owning_section": "Document History"},
        ],
    }

    # Case 1: Query with table designator only
    payload_1 = {**payload_base, "user_query": "rewrite the Document History tablwe"}
    result_1 = await routes.analyze_sop_target(payload_1)
    print("Test 1 Result (Document History Table, typo):")
    print(json.dumps(result_1, indent=2, sort_keys=True))
    assert result_1["target_type"] == "table"
    assert result_1["target_id"] == "tbl-history"
    assert result_1["requires_clarification"] is False

    # Case 2: Query with both table and section designators ("table section")
    payload_2 = {**payload_base, "user_query": "rewrite the Document History table section"}
    result_2 = await routes.analyze_sop_target(payload_2)
    print("Test 2 Result (Document History table section):")
    print(json.dumps(result_2, indent=2, sort_keys=True))
    assert result_2["target_type"] == "table"
    assert result_2["target_id"] == "tbl-history"
    assert result_2["requires_clarification"] is False

    # Case 3: Query with section designator only
    payload_3 = {**payload_base, "user_query": "rewrite the Document History section"}
    result_3 = await routes.analyze_sop_target(payload_3)
    print("Test 3 Result (Document History section):")
    print(json.dumps(result_3, indent=2, sort_keys=True))
    assert result_3["target_type"] == "section"
    assert result_3["target_id"] == "sec-history"
    assert result_3["requires_clarification"] is False

    print("\n--- ALL HISTORY TABLE ASSERTIONS PASSED ---")

async def test_multilingual_concept_matching():
    # Test target resolution on literal matches and cross-lingual matches
    payload_base = {
        "action": "rewrite",
        "full_text": "Scope section\nTable 4 - Scope\nSome description.\nDocument History\nVersion 1.0",
        "document_excerpt": "Scope, Table 4 - Scope, Document History table",
        "selection": {"empty": True, "text": ""},
        "active_scope": {"sectionName": "Scope"},
        "section_index": [
            {"id": "sec-scope", "type": "section", "label": "Scope", "text": "Some scope."},
            {"id": "sec-history", "type": "section", "label": "Document History", "text": "History metadata."},
            {"id": "sec-geltungsbereich", "type": "section", "label": "Geltungsbereich", "text": "Geltungsbereich details."},
        ],
        "table_index": [
            {"id": "tbl-scope", "type": "table", "label": "Table 4 - Scope", "text": "Scope metadata."},
            {"id": "tbl-history", "type": "table", "label": "Table 2", "text": "Version 1.0 details.", "owning_section": "Document History"},
        ],
        "paragraph_index": [],
        "document_tree": [
            {"id": "sec-scope", "type": "section", "label": "Scope"},
            {"id": "tbl-scope", "type": "table", "label": "Table 4 - Scope"},
            {"id": "sec-history", "type": "section", "label": "Document History"},
            {"id": "tbl-history", "type": "table", "label": "Table 2", "owning_section": "Document History"},
            {"id": "sec-geltungsbereich", "type": "section", "label": "Geltungsbereich"},
        ],
    }

    # Case 1: Deterministic literal exact match (German to German)
    det_german = routes._deterministic_target_analysis(
        user_query="bearbeite den Geltungsbereich",
        sections=payload_base["section_index"],
        tables=[],
        selection=payload_base["selection"],
        active_scope=payload_base["active_scope"]
    )
    print("Deterministic exact match Geltungsbereich -> Geltungsbereich:")
    print(json.dumps(det_german, indent=2))
    assert det_german is not None
    assert det_german["target_id"] == "sec-geltungsbereich"

    # Case 2: Deterministic literal exact match (English to English)
    det_english = routes._deterministic_target_analysis(
        user_query="rewrite the Scope section",
        sections=payload_base["section_index"],
        tables=[],
        selection=payload_base["selection"],
        active_scope=payload_base["active_scope"]
    )
    print("Deterministic exact match Scope -> Scope:")
    print(json.dumps(det_english, indent=2))
    assert det_english is not None
    assert det_english["target_id"] == "sec-scope"

    # Case 3: Dynamic cross-lingual target resolution (German query targeting English section) via LLM
    routes.resolve_sop_target_with_deep_agent = fail_deep_agent
    routes.create_chat_llm = lambda *args, **kwargs: MockTargetResolutionLLM()
    routes.TARGET_DEEP_AGENT_TIMEOUT_SECONDS = 1
    routes.TARGET_LLM_FALLBACK_TIMEOUT_SECONDS = 1

    # German query "bearbeite den Geltungsbereich" resolves to English "Scope"
    payload_cross_1 = {
        **payload_base,
        "user_query": "bearbeite den Geltungsbereich",
        # Remove the German "Geltungsbereich" section from indexes to force it to match English "Scope"
        "section_index": [
            {"id": "sec-scope", "type": "section", "label": "Scope", "text": "Some scope."},
            {"id": "sec-history", "type": "section", "label": "Document History", "text": "History metadata."},
        ],
        "document_tree": [
            {"id": "sec-scope", "type": "section", "label": "Scope"},
            {"id": "sec-history", "type": "section", "label": "Document History"},
        ],
    }
    result_cross_1 = await routes.analyze_sop_target(payload_cross_1)
    print("Dynamic cross-lingual match Geltungsbereich -> Scope:")
    print(json.dumps(result_cross_1, indent=2, sort_keys=True))
    assert result_cross_1["target_type"] == "section"
    assert result_cross_1["target_id"] == "sec-scope"
    assert result_cross_1["requires_clarification"] is False

    # German query "zeige den anderungsverlauf" resolves to English "Document History"
    payload_cross_2 = {
        **payload_base,
        "user_query": "zeige den anderungsverlauf",
        "section_index": [
            {"id": "sec-scope", "type": "section", "label": "Scope", "text": "Some scope."},
            {"id": "sec-history", "type": "section", "label": "Document History", "text": "History metadata."},
        ],
        "document_tree": [
            {"id": "sec-scope", "type": "section", "label": "Scope"},
            {"id": "sec-history", "type": "section", "label": "Document History"},
        ],
    }
    result_cross_2 = await routes.analyze_sop_target(payload_cross_2)
    print("Dynamic cross-lingual match Änderungsverlauf -> Document History:")
    print(json.dumps(result_cross_2, indent=2, sort_keys=True))
    assert result_cross_2["target_type"] == "section"
    assert result_cross_2["target_id"] == "sec-history"
    assert result_cross_2["requires_clarification"] is False

    print("\n--- ALL MULTILINGUAL CONCEPT ASSERTIONS PASSED ---")

async def main():
    await test_procedure_matching()
    await test_history_table_matching()
    await test_multilingual_concept_matching()

if __name__ == "__main__":
    asyncio.run(main())
