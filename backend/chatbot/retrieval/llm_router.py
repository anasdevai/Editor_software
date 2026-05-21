import json
import logging
import os
import re
from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chatbot.llm.provider import create_chat_llm
from retrieval.query_router import route_query

ROUTER_PROMPT_TEMPLATE = """\
Given this user query, classify it for Qdrant retrieval routing.
Return ONLY a valid JSON object. No explanation. No markdown.

USER QUERY: {user_question}

Return this exact JSON structure:
{{
  "collections": [],
  "exact_filters": {{}},
  "language": "",
  "query_type": ""
}}

Field definitions:

"collections" → array, one or more of: "sops", "deviations", "capas", "audits", "decisions"
"exact_filters" → object with any of these keys (omit key if not applicable):
  {{
    "sop_number":        "SOP-IT-001",   // if user mentions exact SOP ID
    "deviation_number":  "DEV-IT-401",   // if user mentions exact DEV ID
    "impact_level":      "Critical",     // if user asks about a severity level
    "external_status":   "open",         // if user asks about active/open items
    "department":        "IT/Security"   // if user asks about a specific dept
  }}
"language" → one of: "en", "de", "mixed"
"query_type" → one of:
  "lookup"          — user wants a specific record by ID
  "compare"         — user wants to compare two or more records
  "summarize"       — user wants a summary of a topic or collection
  "cross_reference" — user wants to link deviations to SOPs or vice versa
  "status_check"    — user wants to know if something is open/closed/active

Routing rules (apply in order):
  1. Query mentions "SOP-" prefix        → collections must include "sops"
  2. Query mentions "DEV-" prefix        → collections must include "deviations"
  3. Query asks about "version", "what does it say", "content of"
                                         → include "sops"
  4. Query asks "which deviation relates to SOP-X"
                                         → collections: ["sops", "deviations"]
  5. Query asks "all open deviations"    → exact_filters: {{ "external_status": "open" }}
  6. Query asks "critical issues"        → exact_filters: {{ "impact_level": "Critical" }}
  7. No exact ID mentioned               → exact_filters: {{}} (empty)
  8. German terms detected (e.g., "Abweichung", "Zugriffsmanagement", "Notfall")
                                         → language: "de", search "sops" + "deviations"
"""

class LLMRouter:
    def __init__(self, llm: Optional[Any] = None):
        if llm:
            self.llm = llm
        else:
            self.llm = create_chat_llm(
                temperature=0.0, # Zero temperature for precise classification
                max_output_tokens=int(os.getenv("ROUTER_MAX_OUTPUT_TOKENS", "256")),
                max_retries=1,
            )
        
        self.prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self, query: str) -> Dict:
        """
        Routes the query using LLM and returns a dictionary with collections and filters.
        """
        limited_query = (query or "").strip()
        # Fast-path: deterministic keyword routing avoids an extra LLM call
        # for most operational queries and improves response latency.
        keyword_collections = route_query(limited_query)
        has_explicit_id = bool(re.search(r"\b(SOP|DEV|CAPA|AUDIT|DEC)-[A-Z0-9-]+\b", limited_query, re.IGNORECASE))
        if has_explicit_id or len(keyword_collections) == 1:
            return {
                "collections": keyword_collections,
                "exact_filters": {},
                "language": "en",
                "query_type": "lookup" if has_explicit_id else "summarize",
            }

        try:
            max_query_chars = int(os.getenv("ROUTER_MAX_QUERY_CHARS", "2000"))
            if max_query_chars > 0 and len(limited_query) > max_query_chars:
                limited_query = limited_query[: max_query_chars - 1].rstrip() + "…"
            response_text = self.chain.invoke({"user_question": limited_query})
            # Clean up potential markdown formatting if LLM ignores the "No markdown" instruction
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            route_data = json.loads(response_text)

            valid_sections = {"sops", "deviations", "capas", "audits", "decisions"}
            cols = route_data.get("collections")
            if not isinstance(cols, list):
                cols = []
            normalized = []
            for c in cols:
                if not isinstance(c, str):
                    continue
                key = c.strip().lower()
                if key in valid_sections:
                    normalized.append(key)
            # Empty list from the LLM used to skip all retrievers and force refusal.
            if not normalized:
                normalized = list(keyword_collections or []) or [
                    "sops",
                    "deviations",
                    "capas",
                    "audits",
                    "decisions",
                ]
            route_data["collections"] = normalized

            return route_data
        except Exception as e:
            logging.error(f"LLM Routing failed: {e}")
            # Robust fallback to keyword-based scanner
            return {
                "collections": keyword_collections,
                "exact_filters": {},
                "language": "en",
                "query_type": "summarize"
            }
