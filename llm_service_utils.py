import importlib.util
import re
from pathlib import Path
from typing import Any

_COMPLIANCE_ID_PATTERN = re.compile(
    r'\b([A-Z]{2,8}-[A-Z]{0,4}-?\d{3,})\b'
)

def _extract_compliance_ids(text: str) -> list:
    """Extract all compliance IDs from a block of text."""
    return list(dict.fromkeys(_COMPLIANCE_ID_PATTERN.findall(text)))


def _extract_id_lines(text: str, ids: list) -> dict:
    """For each ID, extract the full line from source text."""
    id_lines = {}
    for line in text.splitlines():
        for cid in ids:
            if cid in line and cid not in id_lines:
                id_lines[cid] = line.strip()
    return id_lines


def _verify_and_restore(rewritten: str, source_text: str, required_ids: list, source_lines: dict) -> str:
    """
    Post-generation completeness check.
    Any required ID missing from the rewrite output is appended verbatim from source.
    """
    missing = [cid for cid in required_ids if cid not in rewritten]
    if not missing:
        return rewritten

    restore_block = "\n\n<!-- Compliance Integrity Restore: the following entries were missing from the rewrite and have been restored verbatim -->\n"
    for cid in missing:
        restore_block += f"\n{source_lines.get(cid, cid)}"

    print(f"[ID Guard] Restored {len(missing)} missing IDs: {missing}")
    return rewritten + restore_block


def _load_build_action_llm_context_block():
    """Load backend formatter so repo-root scripts and the API share one implementation."""
    path = Path(__file__).resolve().parent / "backend" / "app" / "services" / "nlp" / "llm_action_context.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("llm_action_context_dynamic", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "build_action_llm_context_block", None)


def build_action_llm_context_block(
    nlp_bundle: dict[str, Any] | None,
    style_profile: dict[str, Any] | None,
    task: str,
) -> str:
    """
    Compact NLP + style text for Improve / Rewrite / Gap-check prompts.
    Delegates to backend/app/services/nlp/llm_action_context.py when available.
    """
    fn = _load_build_action_llm_context_block()
    if fn is not None:
        return fn(nlp_bundle, style_profile, task)
    if not nlp_bundle:
        return ""
    return f"NLP_BUNDLE_KEYS={list(nlp_bundle.keys())}|task={task}"
