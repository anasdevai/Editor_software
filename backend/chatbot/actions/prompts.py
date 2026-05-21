"""Prompt builders for SOP editor actions.
Language priority: German (de). All other languages are fully supported.
The AI always detects the language of the input text and responds in the same language.

**Canonical source** for `/api/ai/action` prompt text: this module only.
``action/prompts.py`` re-exports these symbols; do not duplicate prompt strings elsewhere.
"""

import json
import re
from typing import Any, Dict, Literal

from schemas.sop_actions import ActionRequest, JustifyRequest

EditScope = Literal["section_only", "full_document"]

# Logged by the FastAPI app as ``source_file`` for observability (keep in sync with this path).
AI_ACTION_PROMPT_SOURCE_FILE = "chatbot/actions/prompts.py"

# Improve / Rewrite: no Qdrant/RAG — LLM uses only system-style instructions + document fields + section text.
IMPROVE_REWRITE_NO_RAG_CONTEXT = (
    "(Kein RAG.) Nutze nur Metadaten + unten stehenden Text. / "
    "(No RAG.) Use only metadata + quoted text below."
)

_LANGUAGE_RULE = """LANGUAGE: Match the input language (German if input is German). Do not mix languages. Keep identifiers, codes, and abbreviations unchanged."""

_SPEED_FIRST = """OUTPUT: Return exactly one valid JSON object. No markdown, no code fences, no explanation, no sources. Be concise."""

_JSON_ESCAPING_RULE = """JSON RULES: Encode newlines as \\n, tabs as \\t, quotes as \\", backslashes as \\\\ inside string values. No literal control characters inside strings."""

_PRESERVE_CORE = """PRESERVE (never alter):
- All IDs: SOP-*, DEV-*, CAPA-*, AUD-*, DEC-*, form names, thresholds, dates, frequencies, versions
- Every section, block, and record: deviations, CAPAs, audit findings, decisions, references, trailing content; item count and order unchanged
- Register-line format: Datum:, Beschreibung:, Ursache:, Aktion:, Verantwortlich:, Finding:, Entscheidung:, Risiko:, Begründung: as separate short lines
- Punctuation habits: do not add sentence-final periods to terse register lines unless already consistent in input
- Named vendors, tools, systems, ports, protocols, values exactly — never convert to examples"""

_THREE_C_IMPROVE_STANDARD = """3C SOP IMPROVEMENT STANDARD:
- Clarity: Correct grammar, sentence flow, vague terms, unclear abbreviations, and unclear responsibility without changing meaning.
- Consistency: Keep the original structure, numbering, field labels, formatting, terminology, paragraph boundaries, and compact register style.
- Compliance: Preserve audit-relevant facts and improve GMP/QA wording without adding new requirements, controls, dates, systems, owners, approvals, or regulatory claims.
- Final self-check: Confirm the output preserves the same meaning, scope, records, IDs, and required fields as TEXT."""

_THREE_C_REWRITE_STANDARD = """3C SOP REWRITE STANDARD:
- Clarity: Rewrite vague, passive, or informal wording into clear, role-based, action-oriented SOP language.
- Consistency: Keep the same section order, numbering, terminology, register format, IDs, record structure, and document tone unless EDIT_SCOPE is FULL_DOCUMENT and TEXT is missing required SOP backbone sections.
- Compliance: Strengthen GMP/QA control language only when supported by TEXT, metadata, or a visible risk in the provided content. Do not invent approvals, limits, systems, owners, dates, forms, thresholds, or regulatory references.
- Final self-check: Verify the rewritten SOP is clear, consistent, compliant, and that no original ID, record, section, or required field was removed."""

_META_USAGE = """METADATA: Use NLP_STRUCTURE_AND_PARAMETERS and database metadata for style, terminology, and structure alignment only. If metadata conflicts with TEXT, preserve TEXT meaning."""

_RECORD_ID_RE = re.compile(r"\b(?:DEV|CAPA|AUD|DEC)-[A-Z0-9]+-\d+\b", re.IGNORECASE)
TRACEABILITY_SECTION_HEADER_RE = re.compile(
    r"\b(?:DEVIATIONS?|CAPAS?|AUDIT(?:\s+FINDINGS?)?|DECISIONS?|ABWEICHUNGEN?)\b",
    re.IGNORECASE,
)
_SOP_BACKBONE_IN_OUTPUT_RE = [
    re.compile(r"(?:^|\n)\s*1\.\s*ZWECK\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*2\.\s*GELTUNGSBEREICH\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*3\.\s*VERANTWORTLICH", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*4\.\s*VERFAHREN\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*5\.\s*DOKUMENTATION\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*6\.\s*ABWEICHUNGSVERWALTUNG\b", re.IGNORECASE | re.MULTILINE),
]


def is_traceability_register_block(text: str) -> bool:
    """True when TEXT is a DEV/CAPA/AUD/DEC register block, not a full SOP."""
    raw = (text or "").strip()
    if len(raw) < 80:
        return False
    record_ids = _RECORD_ID_RE.findall(raw)
    if len(record_ids) < 2:
        return False
    head = raw[:1500]
    if re.search(
        r"(?:^|\n)\s*(?:1\.|##)\s*(?:ZWECK|PURPOSE|GELTUNGSBEREICH|SCOPE)\b",
        head,
        re.IGNORECASE | re.MULTILINE,
    ):
        return False
    if re.search(r"(?:^|\n)\s*(?:4\.|##)\s*(?:VERFAHREN|PROCEDURE)\b", head, re.IGNORECASE | re.MULTILINE):
        return False
    return True


def violates_section_only_scope(original: str, output: str) -> bool:
    """Detect when the model returned a full SOP though only a section was requested."""
    orig = (original or "").strip()
    out = (output or "").strip()
    if not orig or not out:
        return False
    for pattern in _SOP_BACKBONE_IN_OUTPUT_RE:
        if pattern.search(out) and not pattern.search(orig):
            return True
    if len(out) > max(len(orig) * 2, len(orig) + 1200) and not is_traceability_register_block(orig):
        if re.search(r"\bVersion:\s*\d", out, re.IGNORECASE) and not re.search(r"\bVersion:\s*\d", orig, re.IGNORECASE):
            return True
    return False


def _traceability_section_kind(text: str) -> str | None:
    """Infer which traceability register the text refers to (capas vs deviations, etc.)."""
    raw = (text or "")[:800]
    has_capa = bool(re.search(r"\bcapas?\b", raw, re.IGNORECASE))
    has_dev = bool(re.search(r"\bdeviations?\b|\babweichungen?\b", raw, re.IGNORECASE))
    has_dec = bool(re.search(r"\bdecisions?\b|\bentscheidungen?\b", raw, re.IGNORECASE))
    has_aud = bool(re.search(r"\baudit\b", raw, re.IGNORECASE))
    if has_capa and not has_dev:
        return "capas"
    if has_dev and not has_capa:
        return "deviations"
    if has_dec:
        return "decisions"
    if has_aud:
        return "audit"
    return None


def extract_register_slice_from_output(output: str, original: str = "") -> str | None:
    """Recover deviations/register block if the model returned a full SOP."""
    raw = (output or "").strip()
    if not raw:
        return None
    kind = _traceability_section_kind(original)
    patterns_by_kind: dict[str, tuple[str, ...]] = {
        "capas": (
            r"(?im)(?:🟠\s*)?CAPAS?\b.*",
            r"(?im)\bCAPA-[A-Z0-9]+-\d+\b.*",
        ),
        "deviations": (
            r"(?im)(?:🔴\s*)?DEVIATIONS\b.*",
            r"(?im)\bDEV-[A-Z0-9]+-\d+\b.*",
        ),
        "decisions": (r"(?im)(?:🟡\s*)?DECISIONS?\b.*",),
        "audit": (r"(?im)AUDIT\b.*",),
    }
    ordered: list[str] = []
    if kind and kind in patterns_by_kind:
        ordered.extend(patterns_by_kind[kind])
    ordered.extend(
        (
            r"(?im)(?:🔴\s*)?DEVIATIONS\b.*",
            r"(?im)(?:🟠\s*)?CAPAS?\b.*",
            r"(?im)(?:🟡\s*)?DECISIONS?\b.*",
            r"(?im)ABWEICHUNGSVERWALTUNG\b.*",
            r"(?im)(?:^|\n)\s*6\.\s*ABWEICHUNGSVERWALTUNG\b.*",
        )
    )
    seen: set[str] = set()
    for pattern in ordered:
        if pattern in seen:
            continue
        seen.add(pattern)
        match = re.search(pattern, raw)
        if match:
            return raw[match.start() :].strip()
    dev_match = re.search(r"\bDEV-[A-Z0-9]+-\d+\b", raw, re.IGNORECASE)
    if dev_match:
        line_start = raw.rfind("\n", 0, dev_match.start())
        chunk = raw[line_start + 1 if line_start >= 0 else 0 :].strip()
        if _RECORD_ID_RE.search(chunk):
            return chunk
    return None


def resolve_edit_scope(request: ActionRequest) -> EditScope:
    """Infer whether the LLM must touch only the selection/section or the full SOP."""
    explicit = getattr(request, "edit_scope", None)
    section_type = (request.section_type or "").strip().lower()
    section_title = (request.section_title or "").strip().lower()
    full_doc_signals = (
        explicit == "full_document"
        or section_type in ("full document", "full sop")
        or section_title
        in (
            "full document",
            "full sop",
            "gesamte sop",
            "komplette sop",
            "entire sop",
            "whole sop",
        )
    )
    if full_doc_signals:
        return "full_document"

    text = request.section_text or ""
    if is_traceability_register_block(text):
        return "section_only"

    if explicit == "section_only":
        return "section_only"

    return "section_only"


def _scope_directive(request: ActionRequest, action: str) -> str:
    scope = resolve_edit_scope(request)
    section = (request.section_title or "Selected text").strip()
    action_label = "rewrite" if action == "rewrite" else "improve"

    if scope == "full_document":
        return f"""EDIT_SCOPE: FULL_DOCUMENT
- Task: {action_label} the COMPLETE SOP body in TEXT (entire document provided below).
- Output MUST start at the very beginning of TEXT (title/metadata, section 1 Purpose/Zweck, etc.) and continue through ALL sections in the same order as TEXT.
- FORBIDDEN: starting at DEVIATIONS, CAPAs, audit/decision registers, or any mid-document traceability block unless that block is literally the first content in TEXT.
- Use NLP_STRUCTURE_AND_PARAMETERS for document-wide style, domain, roles, sections, compliance refs, risks, and rewrite_improve_parameters.
- For FULL_DOCUMENT only: add standard backbone sections only when TEXT lacks them: Purpose, Scope, Responsibilities, Procedure, Documentation, Review/Approval.
- Preserve every ID (SOP-*, DEV-*, CAPA-*, AUD-*, DEC-*), register line, deviation/CAPA/audit/decision block, and trailing traceability content.
- Output can be longer than input only when required to add missing backbone sections for FULL_DOCUMENT scope."""

    register_note = ""
    section_title = (request.section_title or "").strip()
    if is_traceability_register_block(request.section_text or "") or TRACEABILITY_SECTION_HEADER_RE.search(
        section_title
    ):
        register_note = f"""
TRACEABILITY_SECTION_MODE (named block: "{section_title}"):
- TEXT is ONE complete traceability section: section heading (if present) + ALL records under it (DEV/CAPA/AUD/DEC entries and their fields).
- The user asked to rewrite/improve THIS SECTION ONLY — not a single heading line, not the full SOP.
- Match the section kind in the target title: CAPAs block → output CAPAs only (CAPA-* IDs); DEVIATIONS block → DEVIATIONS only (DEV-* IDs). Never substitute or merge a different register (e.g. do not output DEVIATIONS when TEXT is CAPAs).
- Output MUST include: (1) the section heading line if present in TEXT, (2) every record entry that appears in TEXT with the same IDs and field labels (Linked DEV, Status, Fällig, Aktion, Verantwortlich, etc.).
- NEVER output SOP title, Version, Status, Purpose, Scope, Responsibilities, Procedure, or Documentation from other sections.
- NEVER stop after the heading — include all CAPA/DEV/AUD/DEC items until the section ends in TEXT.
- Keep the exact record count and order; improve grammar and clarity inside each entry only.
- Output length must stay close to input (about 80–130% of character count)."""

    return f"""EDIT_SCOPE: SECTION_ONLY
- Target section/selection: "{section}" (type: {request.section_type})
- Task: {action_label} ONLY the passage in TEXT — do NOT output the full SOP.
- FORBIDDEN: adding Purpose, Scope, Responsibilities, Procedure, Documentation, Review, or other headings not already inside TEXT.
- FORBIDDEN: rewriting or inventing content for other sections (e.g. if TEXT is DEVIATIONS only, do not add Procedure or Scope).
- FORBIDDEN: outputting "SOP-IT-001", Version, Status, Abteilung, or numbered backbone sections 1–5 unless they are already in TEXT.
- Keep the same structural units as TEXT (headings, lists, tables, register lines). Keep output length within 70–130% of input unless grammar repair requires minor variance.
- Use NLP_STRUCTURE_AND_PARAMETERS only to align tone, terminology, and micro-structure of THIS block — not to expand into a complete SOP.
- Return only the improved/rewritten block that replaces TEXT in the editor.{register_note}"""


def _doc_block(request: ActionRequest, context: str) -> str:
    scope = resolve_edit_scope(request)
    return f"""DOCUMENT
  title: {request.sop_title}
  section: {request.section_title}
  type: {request.section_type}
  edit_scope: {scope}
CONTEXT: {context}"""


def _nlp_section(nlp_block: str) -> str:
    nb = (nlp_block or "").strip()
    if not nb:
        return ""
    return f"\nNLP_STRUCTURE_AND_PARAMETERS:\n{nb}\n"


def _format_nlp_profile_context(
    profile_md: str = "",
    profile_json: dict | None = None,
    detected_nlp: dict | None = None,
) -> str:
    blocks = []
    if profile_md and profile_md.strip():
        blocks.append(f"### ACTIVE PROFILE RULES (profile.md):\n{profile_md.strip()}")
    if profile_json:
        blocks.append(f"### ACTIVE PROFILE CONFIGURATION (JSON):\n{json.dumps(profile_json, ensure_ascii=False, indent=2)}")
    if detected_nlp:
        lines = []
        for key, val in detected_nlp.items():
            if val:
                lines.append(f"- {key}: {json.dumps(val, ensure_ascii=False)}")
        if lines:
            blocks.append("### CURRENT SOP DETECTED NLP PARAMETERS:\n" + "\n".join(lines))
    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks) + "\n"


def build_improve_prompt(
    request: ActionRequest,
    context: str,
    nlp_block: str = "",
    profile_md: str = "",
    profile_json: dict | None = None,
    detected_nlp: dict | None = None,
) -> str:
    context_extra = _format_nlp_profile_context(profile_md, profile_json, detected_nlp)
    return f"""You are a senior GMP/QA SOP editor. TASK: light editorial polish — not a full-document rewrite.
{_SPEED_FIRST}
{_JSON_ESCAPING_RULE}
{_LANGUAGE_RULE}
{_doc_block(request, context)}
{_scope_directive(request, "improve")}
{_nlp_section(nlp_block)}
{context_extra}
{_META_USAGE}
{_PRESERVE_CORE}
{_THREE_C_IMPROVE_STANDARD}

IMPROVE RULES:
- Fix only grammar, missing articles, unclear abbreviations, passive ownership, vague responsibility, and non-GMP wording.
- Align with the Active Profile rules, configuration, and detected NLP parameters if available. Maintain consistency with style suggestions, preferred tone, and terminology.
- Keep the original sentence shape, list/table style, numbering, blank-line rhythm, and paragraph boundaries unless the original structure is broken.
- Never introduce bullets, numbering, labels, or headings not present in the original.
- Never add steps, approvals, systems, requirements, or compliance claims.
- Keep compact register statements compact — do not inflate into narrative prose.
- When EDIT_SCOPE is SECTION_ONLY: output must replace only the targeted block; never return a full SOP skeleton.
- Before returning: compare output against TEXT and restore any missing section, record, field, or ID present in TEXT.

TEXT:
\"\"\"{request.section_text}\"\"\"
Return only:
{{"improved_text":"..."}}"""


def build_summarize_prompt(request: ActionRequest, context: str, nlp_block: str = "") -> str:
    return f"""You are a senior GMP/QA communications lead. TASK: produce a concise executive summary of the SOP text (no full rewrite).
{_SPEED_FIRST}
{_JSON_ESCAPING_RULE}
{_LANGUAGE_RULE}
{_doc_block(request, context)}
{_nlp_section(nlp_block)}
{_META_USAGE}
{_PRESERVE_CORE}

SUMMARY RULES:
- 6–12 short bullets or 2 tight paragraphs maximum.
- Cover: purpose, scope, critical controls, key roles, records, and review cadence when present in the text.
- Do not invent facts, dates, systems, or approvals that are not present in TEXT.
- Keep identifiers and codes exactly as written.

TEXT:
\"\"\"{request.section_text}\"\"\"
Return only:
{{"improved_text":"..."}}"""


def build_analyze_prompt(request: ActionRequest, context: str, nlp_block: str = "") -> str:
    return f"""You are a senior GMP/QA compliance reviewer. TASK: structured compliance analysis of the SOP excerpt (not a rewrite).
{_SPEED_FIRST}
{_JSON_ESCAPING_RULE}
{_LANGUAGE_RULE}
{_doc_block(request, context)}
{_nlp_section(nlp_block)}
{_META_USAGE}
{_PRESERVE_CORE}

ANALYSIS RULES:
- Output a numbered list (plain lines separated by \\n) covering: clarity, control strength, evidence/records, training, change control, and residual risks.
- Reference only themes visible in TEXT; use bracketed placeholders for required information that is missing from TEXT.
- Do not add regulatory citations unless already present in the text.

TEXT:
\"\"\"{request.section_text}\"\"\"
Return only:
{{"improved_text":"..."}}"""


def build_rewrite_prompt(
    request: ActionRequest,
    context: str,
    nlp_block: str = "",
    profile_md: str = "",
    profile_json: dict | None = None,
    detected_nlp: dict | None = None,
) -> str:
    scope = resolve_edit_scope(request)
    full_backbone = ""
    if scope == "full_document":
        full_backbone = """
FULL SOP BACKBONE (add when missing from TEXT, in input language):
  Purpose/Zweck · Scope/Geltungsbereich · Responsibilities/Verantwortlichkeiten · Procedure/Verfahren ·
  Acceptance Criteria · Documentation/Records · Review/Approval/Lifecycle ·
  Training (if relevant) · Appendices/Traceability (if records present)
"""
    context_extra = _format_nlp_profile_context(profile_md, profile_json, detected_nlp)
    return f"""You are a senior GMP/QA SOP architect. TASK: structural rewrite into industry-ready SOP language for the scope in EDIT_SCOPE.
{_SPEED_FIRST}
{_JSON_ESCAPING_RULE}
{_LANGUAGE_RULE}
{_doc_block(request, context)}
{_scope_directive(request, "rewrite")}
{_nlp_section(nlp_block)}
{context_extra}
{_META_USAGE}
{_PRESERVE_CORE}
{_THREE_C_REWRITE_STANDARD}
{full_backbone}
REWRITE RULES:
- Follow EDIT_SCOPE strictly: SECTION_ONLY → never emit a full SOP; FULL_DOCUMENT → rewrite from the first line of TEXT through the end in document order.
- Follow the Active Profile JSON, Markdown (profile.md), and current SOP detected NLP parameters. Strictly apply rewrite rules, terminology preferences, RACI patterns, tone guidelines, and workflow patterns.
- Single section/heading (e.g. CAPAs, DEVIATIONS, Procedure): rewrite only lines in TEXT; never swap CAPAs for DEVIATIONS or vice versa.
- Use bracketed placeholders only for missing controls clearly implied by TEXT inside that section.
- Apply rewrite_improve_parameters from NLP_STRUCTURE_AND_PARAMETERS only for tone, formality, numbering, and domain vocabulary; never use them to change facts.

LANGUAGE & STYLE:
- Active voice, named accountable roles, precise verbs, consistent controlled vocabulary.
- For required missing facts, use bracketed placeholders: "[Zu definieren: verantwortliche Rolle]", "[To define: retention period]".
- Never invent dates, systems, owners, limits, forms, thresholds, or approvals.

RECORD / REGISTER MODE (DEV/CAPA/AUDIT/DECISION entries):
- Terse-record mode: fix grammar/clarity only; do not expand compact lines into formal narrative.
- Avoid filler: "Es existiert", "Es erfolgte", "wurde … durchgeführt" — keep concise factual form.
- Relocate deviation/CAPA/audit/decision logs to a traceability section only if original already separates them.

CONTROLS (add only when supported by TEXT, metadata, named roles, or visible risks):
  trigger · frequency/SLA · evidence record · approval gate · verification step ·
  exception handling · escalation · acceptance criterion · retention/location · effectiveness review

- Before returning: compare output against TEXT and restore any missing section, record, field, or ID.

TEXT:
\"\"\"{request.section_text}\"\"\"
Return only:
{{"rewritten_text":"..."}}"""


def build_section_only_rewrite_retry_prompt(request: ActionRequest, context: str, nlp_block: str = "") -> str:
    """Emergency retry when the model returned a full SOP for a section-only request."""
    section = (request.section_title or "Selected section").strip()
    return f"""CRITICAL RETRY — previous answer was wrong (full SOP or heading-only).
{_SPEED_FIRST}
{_JSON_ESCAPING_RULE}
{_LANGUAGE_RULE}
{_doc_block(request, context)}
EDIT_SCOPE: SECTION_ONLY — STRICT
- Section: "{section}"
- Return the COMPLETE section from TEXT: heading line (if any) + ALL records (CAPA/DEV/AUD/DEC entries) — NOT just the heading.
- Preserve section kind: if TEXT is CAPAs (CAPA-* IDs), do NOT output DEVIATIONS; if TEXT is DEVIATIONS (DEV-* IDs), do NOT output CAPAs.
- Do NOT include: SOP title, Version, Status, Abteilung, sections 1–5 (Zweck, Geltungsbereich, Verfahren, etc.).
- Same record IDs and count as TEXT.
{_nlp_section(nlp_block[:2500] if nlp_block else "")}
TEXT (rewrite this entire section only):
\"\"\"{request.section_text}\"\"\"
Return only:
{{"rewritten_text":"..."}}"""


def build_section_only_improve_retry_prompt(request: ActionRequest, context: str, nlp_block: str = "") -> str:
    return build_section_only_rewrite_retry_prompt(request, context, nlp_block).replace(
        '"rewritten_text"', '"improved_text"'
    ).replace("rewrite", "improve", 1)


def _gap_scope_directive(request: ActionRequest) -> str:
    scope = resolve_edit_scope(request)
    section = (request.section_title or "Selected text").strip()
    if scope == "full_document":
        return f"""AUDIT_SCOPE: FULL_DOCUMENT
- Analyze the COMPLETE open SOP provided in TEXT (all sections and traceability records).
- Report gaps across the entire document; cite section names and record IDs (SOP-*, DEV-*, CAPA-*, AUD-*, DEC-*)."""
    return f"""AUDIT_SCOPE: SECTION_ONLY
- Analyze ONLY the section in TEXT: "{section}" — not other SOP sections.
- Do NOT report gaps in Purpose, Scope, Procedure, etc. unless they appear inside TEXT.
- If TEXT is a CAPAs/DEVIATIONS/DECISIONS register block, audit only those entries and their fields (status, dates, linkages, actions).
- Cite evidence from TEXT only; do not invent gaps for parts of the SOP not included in TEXT."""


def build_gap_check_prompt(
    request: ActionRequest,
    context: str,
    nlp_block: str = "",
    profile_md: str = "",
    profile_json: dict | None = None,
    detected_nlp: dict | None = None,
) -> str:
    context_extra = _format_nlp_profile_context(profile_md, profile_json, detected_nlp)
    return f"""You are a senior GMP/QA compliance auditor. TASK: audit-grade gap check of the selected SOP text.
{_SPEED_FIRST}
{_JSON_ESCAPING_RULE}
{_LANGUAGE_RULE}
SOP: "{request.sop_title}" | Section: "{request.section_title}" ({request.section_type})
edit_scope: {resolve_edit_scope(request)}

{_gap_scope_directive(request)}

HYBRID_RAG_REFERENCE_CONTEXT:
{context}
{_nlp_section(nlp_block)}
{context_extra}
CONTEXT USAGE:
- TEXT is the primary audit evidence.
- RAG context: compare against expected controls, related SOP language, and compliance patterns.
- Active Profile JSON & Markdown, and SOP Detected NLP parameters: check if the text violates the style, tone, RACI, workflow, compliance requirements, or terminology specified in the profile or detected parameters.
- Report a gap only when supported by TEXT, NLP metadata, profile rules, or RAG context — not generic GMP knowledge alone.
- If RAG is absent or unrelated, state: "Gap check based on TEXT and NLP metadata only."

AUDIT METHOD:
1. Identify expected SOP structure from TEXT, metadata, NLP sections, and Active Profile.
2. Check each required element for presence, specificity, and actionability.
3. Compare deviations/CAPAs/audits/decisions/controls/dates/statuses for internal consistency.
4. Cite exact evidence (section name or record ID: SOP-*, DEV-*, CAPA-*, AUD-*, DEC-*) for every gap.

GAP CATEGORIES:
- Missing sections: Purpose, Scope, Responsibilities, Procedure, Documentation, Review/Approval
- Missing role ownership, approver, executor, escalation path, QA oversight
- Missing frequencies, deadlines, SLAs, trigger conditions, effective/review/closure dates
- Missing controls: verification step, access control, dual control, monitoring, alarm criteria, acceptance criteria
- Documentation gaps: form name, record location, retention, evidence, timestamp, signature
- Linkage gaps: missing IDs, inconsistent statuses, open CAPAs without closure, findings without CAPA, decisions without rationale
- Ambiguous wording: "regelmäßig", "zeitnah", "bei Bedarf", "sofort", "ausreichend" without measurable criteria
- Metadata inconsistencies: SOP number/title/version/status/department conflicts between TEXT and database metadata

OUTPUT RULES:
- Practical audit findings — not rewritten SOP prose.
- When AUDIT_SCOPE is SECTION_ONLY: keep the report focused (about 400–900 words); cite only gaps for the section in TEXT.
- Prioritize compliance gaps over style/grammar observations.
- If no material gaps found, state clearly and list residual assumptions.
- Do not propose a new SOP version, new status, or relocate DEV/CAPA/AUDIT logs unless TEXT already uses appendix structure.
- Localize headings to input language. German → "Zusammenfassung", "RAG/NLP-Grundlage", "Festgestellte Lücken", "Empfohlene Korrekturen", "Vorgeschlagener SOP-Ergänzungstext", "Verbleibende Annahmen".

TEXT:
\"\"\"{request.section_text}\"\"\"

Return only one JSON object:
{{"analysis":"Zusammenfassung/Summary:\\n...\\n\\nRAG/NLP-Grundlage/Basis:\\n...\\n\\nFestgestellte Lücken/Identified Gaps:\\n1. Gap: ...\\n   Evidence: ...\\n   Risk/Impact: ...\\n   Recommended Fix: ...\\n\\nEmpfohlene Korrekturen/Recommended Fixes:\\n1. ...\\n\\nVorgeschlagener SOP-Ergänzungstext/Suggested SOP Text:\\n...\\n\\nVerbleibende Annahmen/Residual Assumptions:\\n..."}}
No markdown. No sources. No text outside JSON."""


def build_convert_prompt(request: ActionRequest) -> str:
    return f"""Du bist ein erfahrener GMP/QA Dokumentationsspezialist.
You are a senior GMP/QA technical writer and regulatory documentation specialist.

{_LANGUAGE_RULE}

Konvertiere den folgenden Rohtext in ein vollständig strukturiertes SOP-Dokument.
Convert the following raw text into a properly structured SOP document.

═══════════════════════════════════════════════════════════════
DOKUMENTKONTEXT / DOCUMENT CONTEXT
═══════════════════════════════════════════════════════════════
SOP-Titel / SOP Title: "{request.sop_title}"

ROHTEXT / RAW TEXT:
\"\"\"{request.section_text}\"\"\"

═══════════════════════════════════════════════════════════════
PFLICHTANFORDERUNGEN / MANDATORY REQUIREMENTS
═══════════════════════════════════════════════════════════════
  • Alle fünf Abschnitte sind PFLICHT. Kein Abschnitt darf fehlen.
    All five sections are MANDATORY. No section may be omitted.
  • Falls ein Abschnitt nicht genug Informationen hat, schreibe:
    "[Zu definieren — [spezifisches Detail] vor SOP-Freigabe festlegen]"
  • Schreibe "procedure" als JSON-Array von Strings, einen Schritt pro String.
  • Verwende GMP-konforme Sprache: imperative Verben, benannte Rollen, keine Mehrdeutigkeit.
  • Minimum 5 Schritte im Verfahrensabschnitt / Minimum 5 steps in the procedure section.

Gib NUR ein gültiges JSON-Objekt mit genau diesen Schlüsseln zurück:
Return ONLY a valid JSON object with exactly these keys:
{{
  "purpose": "Ein Satz: Was diese SOP regelt und warum sie existiert / One sentence: what this SOP governs and why",
  "scope": "Vollständige Geltungsbereichsdefinition mit Rollen, Systemen und ggf. Ausnahmen / Full scope definition",
  "responsibilities": "Benannte Rollen mit spezifischen, imperativen Verpflichtungen / Named roles with specific obligations",
  "procedure": [
    "Schritt 1: [Benannte Rolle] soll [Aktion] mit [Methode/Werkzeug] / Step 1: ...",
    "Schritt 2: [Benannte Rolle] soll [Aktion] und dokumentieren in [Formularname] / Step 2: ...",
    "Schritt 3: ...",
    "Schritt 4: ...",
    "Schritt 5: ..."
  ],
  "documentation": "Alle Formulare, Protokolle und Aufzeichnungen: Name, Aufbewahrungsort, Aufbewahrungsfrist / All records: name, location, retention period"
}}"""


def build_convert_retry_prompt(request: ActionRequest) -> str:
    return build_convert_prompt(request) + (
        "\n\n═══════════════════════════════════════════════════════════════\n"
        "KRITISCHE WIEDERHOLUNGSANWEISUNG / CRITICAL RETRY INSTRUCTION\n"
        "═══════════════════════════════════════════════════════════════\n"
        "Deine vorherige Antwort war kein gültiges JSON oder enthielt fehlende Schlüssel.\n"
        "Your previous response was not valid JSON or was missing required keys.\n"
        "Du MUSST NUR ein gültiges JSON-Objekt mit genau diesen fünf Schlüsseln zurückgeben:\n"
        "  'purpose', 'scope', 'responsibilities', 'procedure' (als Array), 'documentation'\n"
        "Alle fünf Schlüssel müssen vorhanden und nicht leer sein.\n"
        "Verwende professionellen Platzhaltertext wenn Quellinformationen unvollständig sind.\n"
        "KEIN Markdown, KEINE Erklärung, KEIN Text außerhalb des JSON-Objekts."
    )


def build_justify_prompt(request: JustifyRequest) -> str:
    return f"""Du bist ein leitender GMP/QA Compliance-Schreiber, der GMP-Audit-Trail-Einträge erstellt,
die regulatorischen Inspektionsanforderungen entsprechen.
You are a senior GMP/QA compliance writer generating GMP audit trail entries.

{_LANGUAGE_RULE}

═══════════════════════════════════════════════════════════════
ÄNDERUNGSKONTEXT / CHANGE CONTEXT
═══════════════════════════════════════════════════════════════
SOP-Titel / SOP Title    : "{request.sop_title}"
Abschnittstitel / Section: "{request.section_title}"
Abschnittstyp / Type     : {request.section_type}
Änderungstyp / Change    : {request.change_type}

ORIGINALTEXT / ORIGINAL TEXT:
\"\"\"{request.old_text}\"\"\"

NEUER TEXT / NEW TEXT:
\"\"\"{request.new_text}\"\"\"

═══════════════════════════════════════════════════════════════
AUFGABE / TASK
═══════════════════════════════════════════════════════════════
Erstelle eine formelle, rechtlich vertretbare Begründung für diese Änderung.
Write a formal, legally defensible justification for this change.

ANFORDERUNGEN / REQUIREMENTS:
  • Nennt explizit die SOP: "{request.sop_title}"
  • Nennt explizit den Abschnitt: "{request.section_title}"
  • Beschreibt WAS sich geändert hat (Art der Änderung)
  • Erklärt WARUM die Änderung vorgenommen wurde
  • Beschreibt WIE die Änderung Compliance, Risikominimierung oder Qualität verbessert
  • Genau 2 bis 3 Sätze — nicht mehr, nicht weniger
  • Formelle, professionelle Sprache (kein "ich/wir")
  • Vergangenheitsform (die Änderung wurde vorgenommen)

Gib NUR ein gültiges JSON-Objekt zurück / Return ONLY a valid JSON object:
{{
  "justification": "2-3 formelle Sätze mit expliziter Nennung der SOP und des Abschnitts sowie der spezifischen Begründung.",
  "change_category": "eines von genau: clarity_improvement | compliance_alignment | error_correction | process_update | regulatory_requirement",
  "regulatory_reference": "Spezifische regulatorische Klausel (z.B. 'ISO 13485:2016 Abschnitt 4.2.4') oder null"
}}"""