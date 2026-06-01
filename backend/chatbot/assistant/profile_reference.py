"""
Detect when the user wants rewrite/improve rules from a specific SOP profile (profile.md),
not the default profile attached to the open document.
"""

from __future__ import annotations

import re
from typing import Any

_GENERIC_PROFILE_REF_STOP = frozenset(
    {
        "saved",
        "default",
        "current",
        "open",
        "active",
        "stored",
        "detected",
        "client",
        "clients",
        "this",
        "that",
        "the",
    }
)


def _clean_profile_ref_token(raw: str) -> str:
    ref = re.sub(r"\s+", " ", str(raw or "").strip(" .:-\"'"))
    if not ref:
        return ""
    ref = re.sub(
        r"^(?:rewrite|improve|revise|gap\s*check|summarize|shorten|expand|change|update|make|"
        r"umschreib|verbesser|überarbeit|ueberarbeit|this|the|full|entire|whole|any|section|sop|document)\s+",
        "",
        ref,
        flags=re.IGNORECASE,
    ).strip()
    for marker in (" on ", " in ", " using ", " use ", " with ", " per ", " nach "):
        if marker in f" {ref.lower()} ":
            ref = re.split(rf"\b{marker.strip()}\b", ref, flags=re.IGNORECASE)[-1].strip()
    return ref.strip(" .:-\"'")


def extract_editorial_profile_reference(text: str) -> str | None:
    """
    Pull editorial profile name from phrases like:
    - rewrite the full section in Emergency access sop profile
    - improve using SOP-IT-003 profile
    - rewrite in german_sop2 profile style
    """
    msg = str(text or "").strip()
    if not msg:
        return None

    if re.search(r"\bgerman\s+(?:pharma|pharmaceutical)\s+(?:sop\s+)?profile\b", msg, re.IGNORECASE):
        return "German_Pharma_SOP_Profile"

    patterns = [
        r'\b(?:in|with|using|use|per|nach)\s+"([^"]{2,120})"\s+(?:sop\s+)?profile\b',
        r"\b(?:in|with|using|use|per|nach)\s+'([^']{2,120})'\s+(?:sop\s+)?profile\b",
        r"\b(?:rewrite|improve|revise|gap\s*check)\b[\s\S]{0,160}?"
        r"\b(?:in|with|using|per|nach)\s+(?:the\s+)?([a-z0-9][a-z0-9._/&()\- ]{2,100}?)\s+(?:sop\s+)?profile\b",
        r"\b(?:in|with|using|use|per|nach)\s+(?:the\s+)?([a-z0-9][a-z0-9._/&()\- ]{2,100}?)\s+(?:sop\s+)?profile\b",
        r"\b(SOP-[A-Z0-9-]+)\s+profile\b",
        r"\b([a-z0-9][a-z0-9._/-]+(?:\s+[a-z0-9][a-z0-9._/-]+){0,5})\s+profile\s+style\b",
        r"\b(?:in|using|use)\s+([a-z0-9][a-z0-9._/-]+)\s+style\b",
        r"\bstyle\s+of\s+(SOP-[A-Z0-9-]+|[a-z0-9][a-z0-9._/-]+(?:\s+[a-z0-9._/-]+){0,5})\b",
        r"\b(emergency\s+access|break[-\s]?glass|network\s+security)(?:\s+\([^)]+\))?\s+profile\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if not match:
            continue
        token = _clean_profile_ref_token(match.group(1))
        if (
            token
            and len(token) >= 2
            and token.lower() not in _GENERIC_PROFILE_REF_STOP
        ):
            return token
    return None


def editorial_profile_sop_number_hint(reference: str) -> str | None:
    """Map UI-style profile labels to SOP numbers for DB profile resolution."""
    raw = str(reference or "").strip()
    if not raw:
        return None
    if re.match(r"^SOP-[A-Z0-9-]+$", raw, re.IGNORECASE):
        return raw.upper()
    low = raw.lower()
    if re.search(r"\b(?:emergency\s*access|break[-\s]?glass|notfall(?:zugriff)?)\b", low):
        return "SOP-IT-003"
    if re.search(r"\b(?:network\s+security|firewall|ot[/\s]?it)\b", low):
        return "SOP-IT-002"
    return None


def editorial_profile_instruction_suffix(reference: str) -> str:
    """Phrase that ``_resolve_explicit_style_override`` in ai_routes can parse."""
    ref = str(reference or "").strip()
    if not ref:
        return ""
    return f'Apply using "{ref}" profile style.'


def build_editorial_profile_hints(
    user_message: str,
    *,
    open_sop_number: str = "",
    open_sop_title: str = "",
) -> list[str]:
    ref = extract_editorial_profile_reference(user_message)
    if not ref:
        return []
    doc = " / ".join(x for x in [open_sop_number, open_sop_title] if x).strip() or "the document open in the editor"
    return [
        f"EDITORIAL PROFILE (user-requested): {ref} — load profile.md / JSON for this profile from the database.",
        f"CONTENT SOURCE (mandatory): rewrite/improve the TEXT of {doc} only. "
        "Preserve this open SOP's record IDs, dates, system names, thresholds, and sensitive operational facts.",
        "Apply the editorial profile to structure, control language, modals, terminology, and procedure wording — "
        "do not replace the open SOP with content copied from another SOP document.",
        editorial_profile_instruction_suffix(ref),
    ]
