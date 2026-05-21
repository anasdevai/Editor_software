from __future__ import annotations

from typing import Any


def get_style_prompt_injection(style_profile: dict[str, Any] | None) -> str:
    if not style_profile:
        return ""

    roles = style_profile.get("roles") or []
    role_block = ""
    if roles:
        role_block = (
            "\nDetected Roles: "
            + ", ".join(str(role) for role in roles[:8])
            + "\n- Assign responsibilities to these roles where applicable."
        )

    return (
        "\n[NLP-DRIVEN STYLE STEERING]\n"
        "Use the detected document profile as soft guidance. Do not override retrieved facts.\n"
        f"- Primary Style: {style_profile.get('primary_style', 'procedural')}\n"
        f"- Tone: {style_profile.get('primary_tone', 'formal')}\n"
        f"- Formality: {style_profile.get('formality_level', 'standard')}\n"
        f"- Strictness: {style_profile.get('strictness_level', 'moderate')}\n"
        f"- Numbering: preserve {style_profile.get('numbering_type', 'simple')} numbering where relevant.\n"
        f"- Structure: follow {style_profile.get('format_pattern', 'standard prose')} where relevant.\n"
        f"- Modality: prefer {style_profile.get('compliance_weight', 'recommended')} compliance language."
        f"{role_block}\n"
    )


def inject_style_into_system_prompt(system_prompt: str, style_profile: dict[str, Any] | None) -> str:
    style_block = get_style_prompt_injection(style_profile)
    return f"{system_prompt}\n\n{style_block}" if style_block else system_prompt
