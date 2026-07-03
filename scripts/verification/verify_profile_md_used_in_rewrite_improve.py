"""Verify profile.md is injected into rewrite and improve SOP prompts.

This is a prompt-level guard: it proves the generated profile.md and profile JSON
reach the two editor action prompt builders that the API uses for rewrite/improve.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from chatbot.actions.prompts import build_improve_prompt, build_rewrite_prompt  # noqa: E402
from schemas.sop_actions import ActionRequest  # noqa: E402


PROFILE_MD_SENTINEL = "PROFILE_MD_SENTINEL_REWRITE_IMPROVE_VERIFICATION"
PROFILE_JSON_SENTINEL = "PROFILE_JSON_SENTINEL_RULE_APPLY_QA_STYLE"


def _load_profile_md() -> str:
    profile_path = ROOT / "verify_out" / "SOP-QA-007_adaptive_profile.md"
    if not profile_path.exists():
        raise AssertionError(f"Missing generated profile.md fixture: {profile_path}")
    return profile_path.read_text(encoding="utf-8") + f"\n\n## Verification Sentinel\n- {PROFILE_MD_SENTINEL}\n"


def _make_request(action: str) -> ActionRequest:
    return ActionRequest(
        document_id="verification-doc",
        section_id="zweck",
        sop_title="SOP-QA-007",
        section_title="Zweck",
        section_type="Paragraph",
        section_text=(
            "Zweck\n"
            "Diese SOP beschreibt den Umgang mit Abweichungen. "
            "Abweichungen muessen rechtzeitig gemeldet, bewertet und dokumentiert werden."
        ),
        instruction=f"{action} the Zweck section using the active profile.md",
        edit_scope="section_only",
    )


def _assert_profile_used(prompt: str, action: str) -> None:
    checks = {
        "profile.md block": "### ACTIVE PROFILE RULES (profile.md):" in prompt,
        "profile.md content": PROFILE_MD_SENTINEL in prompt,
        "profile JSON block": "### ACTIVE PROFILE CONFIGURATION (JSON):" in prompt,
        "profile JSON rewrite rule": PROFILE_JSON_SENTINEL in prompt,
        "profile application": "PROFILE APPLICATION (active profile.md / JSON" in prompt,
        f"{action} action line": f"- {action.upper()}:" in prompt,
        "profile parameter contract": "profile.md / profile JSON are parameter sheets only" in prompt,
        "section text retained as source": "Diese SOP beschreibt den Umgang mit Abweichungen" in prompt,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"{action} prompt missing: {', '.join(failed)}")


def main() -> int:
    profile_md = _load_profile_md()
    profile_json = {
        "language": {"primary_language": "de"},
        "domain": "QA_QMS",
        "rewrite_rules": [PROFILE_JSON_SENTINEL],
        "rewrite_improve_parameters": {
            "tone": "formal_regulatory",
            "modal_verbs": ["muss", "darf nicht"],
            "style": "controlled German QA SOP wording",
        },
    }
    detected_nlp = {
        "domain": "QA_QMS",
        "language": "de",
        "section_hint": "Zweck",
    }

    improve_prompt = build_improve_prompt(
        _make_request("improve"),
        context="",
        nlp_block="NLP_STRUCTURE_AND_PARAMETERS: verification fixture",
        profile_md=profile_md,
        profile_json=profile_json,
        detected_nlp=detected_nlp,
    )
    rewrite_prompt = build_rewrite_prompt(
        _make_request("rewrite"),
        context="",
        nlp_block="NLP_STRUCTURE_AND_PARAMETERS: verification fixture",
        profile_md=profile_md,
        profile_json=profile_json,
        detected_nlp=detected_nlp,
    )

    _assert_profile_used(improve_prompt, "improve")
    _assert_profile_used(rewrite_prompt, "rewrite")

    print("PASS: profile.md is injected and applied for improve prompt")
    print("PASS: profile.md is injected and applied for rewrite prompt")
    print("PASS: profile JSON rewrite_rules are included for both actions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
