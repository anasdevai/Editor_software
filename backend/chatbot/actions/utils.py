"""Utility helpers for SOP action processing."""

from __future__ import annotations

import json
import logging
import os
import re
import string
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from langchain_core.documents import Document
from pydantic import ValidationError

from schemas.sop_actions import (
    ConvertResponse,
    GapCheckResponse,
    ImproveResponse,
    RewriteResponse,
)

logger = logging.getLogger(__name__)


ACTION_CONTEXT_EXCERPT_CHARS = 350

ACTION_LLM_EMPTY_RETRY_SUFFIX = (
    "\n\n[System: Your previous completion was empty. Reply with exactly one JSON object, "
    "a non-empty string value, no markdown and no code fences.]"
)


def _action_model_context_tokens() -> int:
    try:
        return max(4096, int(os.getenv("ACTION_MODEL_CONTEXT_TOKENS", "32768")))
    except (TypeError, ValueError):
        return 32768


def _action_prompt_soft_limit_chars() -> int:
    raw = os.getenv("ACTION_PROMPT_SOFT_LIMIT")
    if raw and raw.strip():
        try:
            return max(4000, int(raw))
        except (TypeError, ValueError):
            pass
    return max(8000, int(_action_model_context_tokens() * 0.75))


def truncate_prompt_for_llm(prompt: str, max_chars: int | None = None) -> str:
    """
    Keep the instruction prefix and the tail of the prompt (selected TEXT is usually at the end).
    Reduces empty completions from local OpenAI-compatible servers when context is exceeded.
    """
    p = prompt or ""
    if max_chars is None:
        max_chars = _action_prompt_soft_limit_chars()
    if len(p) <= max_chars:
        return p
    marker = (
        "\n\n[... prompt truncated for model context; rules above still apply; "
        "treat the following tail as the full TEXT to edit ...]\n\n"
    )
    head = min(4500, max_chars // 3)
    tail = max_chars - head - len(marker)
    if tail < 3000:
        head = 2000
        tail = max(2500, max_chars - head - len(marker))
    return p[:head] + marker + p[-tail:]


def invoke_llm_nonempty(
    call_llm,
    prompt: str,
    *,
    schema_name: str,
    audit_log: list[dict[str, Any]],
    step: str,
) -> str:
    """Re-invoke with truncated prompt when the model returns whitespace-only output."""
    text = call_llm(prompt)
    if text and text.strip():
        return text
    logger.warning(
        "[ai-action-llm-empty] step=%s schema=%s prompt_len=%s",
        step,
        schema_name,
        len(prompt or ""),
    )
    audit_log.append({"event": "llm_empty_output", "step": step, "timestamp": utc_now_iso()})
    soft = _action_prompt_soft_limit_chars()
    shrunk = truncate_prompt_for_llm(prompt, max(4000, int(soft * 0.75)))
    payload = (shrunk + ACTION_LLM_EMPTY_RETRY_SUFFIX) if shrunk.strip() else (prompt + ACTION_LLM_EMPTY_RETRY_SUFFIX)
    text2 = call_llm(payload)
    if text2 and text2.strip():
        logger.info("[ai-action-llm-recovered] step=%s schema=%s via=truncated_repeat", step, schema_name)
        return text2
    tiny = truncate_prompt_for_llm(prompt, max(4000, int(soft * 0.5)))
    text3 = call_llm(tiny + ACTION_LLM_EMPTY_RETRY_SUFFIX)
    if text3 and text3.strip():
        logger.info("[ai-action-llm-recovered] step=%s schema=%s via=tiny_repeat", step, schema_name)
        return text3
    return text or ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_json(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^\uFEFF", "", raw)
    # Remove a single wrapping ```json ... ``` block if present
    if "```" in raw:
        raw = re.sub(r"^\s*```(?:json|JSON)?\s*", "", raw, flags=re.MULTILINE).strip()
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```\s*$", "", raw).strip()
    return raw


def strip_sources_citations_noise(text: str) -> str:
    """Remove trailing Sources / citations blocks that break JSON extraction."""
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(
        r"(?is)\n?\s*(?:📎\s*)?sources?\s*:.*$",
        "",
        s,
    )
    s = re.sub(r"(?is)\n?\s*---\s*CITATIONS?\s*---.*$", "", s)
    s = re.sub(r"(?is)\n?\s*---\s*SUGGESTIONS?\s*---.*$", "", s)
    return s.rstrip()


def normalize_action_input_text(text: str) -> str:
    """
    Prepare editor selection for /api/ai/action: keep newlines for SOP structure,
    strip trailing Sources/citation blocks, trim excessive blank lines.
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = strip_sources_citations_noise(s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{5,}", "\n\n\n\n", s)
    return s.strip()


def _preview_for_log(text: str, max_len: int = 800) -> str:
    return (text or "").replace("\n", "\\n").replace("\r", "\\r")[:max_len]


def strip_prologue_before_json(text: str) -> str:
    """Drop a short assistant preface before the first JSON object (if it looks like boilerplate)."""
    s = (text or "").strip()
    if not s:
        return ""
    first = s.find("{")
    if first <= 0:
        return s
    pre = s[:first].strip()
    if len(pre) > 400 or len(pre) < 2:
        return s
    if re.match(
        r"(?is)^(here(is|’s|s)?\b|below\b|sure[,!]?\s*$|ok(ay)?[,!]?\s*$|this is|output\s*:|result\s*:|the json is|assistant\s*:|response\s*:)",
        pre,
    ):
        return s[first:].strip()
    return s


_GAP_PLAIN_HEADINGS = re.compile(
    r"(?is)summary\s*:|identified gaps\s*:|risk/impact\s*:|recommended fixes\s*:",
)


def _escape_raw_control_chars_inside_json_strings(s: str) -> str:
    """
    Local LLMs often emit pretty-printed JSON with raw newlines inside string values.
    json.loads rejects those. Walk from the first '{' and escape literal CR/LF/tab/control
    chars only while inside JSON double-quoted strings (respecting backslash escapes).
    """
    start = s.find("{")
    if start < 0:
        return s
    prefix = s[:start]
    body = s[start:]
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    while i < len(body):
        c = body[i]
        if in_str:
            if esc:
                out.append(c)
                esc = False
                i += 1
                continue
            if c == "\\":
                if i + 5 < len(body) and body[i + 1] == "u":
                    hx = body[i + 2 : i + 6]
                    if len(hx) == 4 and all(ch in string.hexdigits for ch in hx):
                        out.extend(body[i : i + 6])
                        i += 6
                        continue
                out.append(c)
                esc = True
                i += 1
                continue
            if c == '"':
                out.append(c)
                in_str = False
                i += 1
                continue
            if c == "\r":
                nxt = body[i + 1] if i + 1 < len(body) else ""
                out.extend(["\\", "n"])
                i += 2 if nxt == "\n" else 1
                continue
            if c == "\n":
                out.extend(["\\", "n"])
                i += 1
                continue
            if c == "\t":
                out.extend(["\\", "t"])
                i += 1
                continue
            if ord(c) < 32:
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if c == '"':
            in_str = True
        out.append(c)
        i += 1
    return prefix + "".join(out)


def _prepare_for_json_parse(raw: str) -> tuple[str, list[str]]:
    """Strip noise, then repair illegal control characters inside JSON string values."""
    notes: list[str] = []
    s = strip_prologue_before_json(strip_sources_citations_noise(clean_json(raw)))
    repaired = _escape_raw_control_chars_inside_json_strings(s)
    if repaired != s:
        notes.append("escaped_controls_inside_json_strings")
    return repaired, notes


def extract_loose_json_string_value(raw: str, key: str) -> str | None:
    """
    When JSON is truncated or malformed, recover the main string field value.
    Accepts unterminated closing quote (takes rest of buffer as content).
    """
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', raw, flags=re.IGNORECASE)
    if not m:
        return None
    i = m.end()
    parts: list[str] = []
    esc = False
    while i < len(raw):
        c = raw[i]
        if esc:
            esc_map = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
            if c == "u" and i + 4 < len(raw):
                hx = raw[i + 1 : i + 5]
                if len(hx) == 4 and all(ch in string.hexdigits for ch in hx):
                    try:
                        parts.append(chr(int(hx, 16)))
                        i += 5
                        esc = False
                        continue
                    except ValueError:
                        pass
            parts.append(esc_map.get(c, c))
            esc = False
            i += 1
            continue
        if c == "\\":
            esc = True
            i += 1
            continue
        if c == '"':
            break
        parts.append(c)
        i += 1
    out = "".join(parts).strip()
    return out if out else None


def _is_safe_recovered_text(text: str) -> bool:
    if "\x00" in text:
        return False
    if len(text) > 600_000:
        return False
    return True


def _recovery_response_from_raw(raw: str, schema: type[Any]) -> Any | None:
    """Best-effort structured object when JSON parsing fails but the model returned usable text."""
    cleaned = strip_sources_citations_noise(clean_json(raw)).strip()
    if not cleaned:
        return None

    if schema is RewriteResponse:
        t = extract_loose_json_string_value(cleaned, "rewritten_text")
        if t is None or len(t.strip()) < 2 or not _is_safe_recovered_text(t):
            return None
        return RewriteResponse(rewritten_text=t.strip())

    if schema is ImproveResponse:
        t = extract_loose_json_string_value(cleaned, "improved_text")
        if t is None or len(t.strip()) < 2 or not _is_safe_recovered_text(t):
            return None
        return ImproveResponse(improved_text=t.strip())

    if schema is GapCheckResponse:
        t = extract_loose_json_string_value(cleaned, "analysis")
        if t is None or len(t.strip()) < 10 or not _is_safe_recovered_text(t):
            return None
        return GapCheckResponse(analysis=t.strip())

    return None


def _gap_check_plaintext_fallback(raw: str) -> GapCheckResponse | None:
    """If the model ignored JSON and returned the gap template as plain text, accept it."""
    body = strip_sources_citations_noise(clean_json(raw))
    if not body or body.lstrip().startswith("{"):
        return None
    if not _GAP_PLAIN_HEADINGS.search(body):
        return None
    return GapCheckResponse(analysis=body.strip())


def _load_first_json_object_from_text(s: str) -> dict[str, Any]:
    """
    Parse the first complete JSON object from a model response (ignore prose after it).
    Tries multiple '{' positions when an early brace is a false start.
    """
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    if not starts:
        raise json.JSONDecodeError("no JSON object in model output", s, 0)
    last_err: Exception | None = None
    for start in starts[:40]:
        try:
            data, _end = decoder.raw_decode(s, start)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError) as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    raise json.JSONDecodeError("no valid JSON object in model output", s, 0)


def _load_first_json_object(raw: str) -> dict[str, Any]:
    """Backward-compatible entry: prepare (repair strings) then raw_decode."""
    prepared, _notes = _prepare_for_json_parse(raw)
    return _load_first_json_object_from_text(prepared)


def _coerce_gap_check_loose_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Accept legacy {'gaps': [...]} payloads by flattening into analysis text."""
    if "analysis" in data and data.get("analysis"):
        return data
    gaps = data.get("gaps")
    if not isinstance(gaps, list) or not gaps:
        return data
    lines: list[str] = []
    for i, g in enumerate(gaps, 1):
        if not isinstance(g, dict):
            continue
        issue = str(g.get("issue") or "").strip()
        expl = str(g.get("explanation") or "").strip()
        rec = str(g.get("recommendation") or "").strip()
        lines.append(f"{i}. {issue}\n   Explanation: {expl}\n   Recommendation: {rec}")
    if not lines:
        return data
    return {"analysis": "\n\n".join(lines)}


def _coerce_model(data: dict[str, Any], schema: type[Any]) -> Any:
    if schema is GapCheckResponse and isinstance(data, dict):
        data = _coerce_gap_check_loose_dict(dict(data))
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)
    return schema(**data)  # type: ignore[call-arg]


def _plaintext_single_key_fallback(raw: str, schema: type[Any]) -> Any:
    """When the model returns the answer without JSON braces (rare but valid for power users)."""
    text = clean_json(raw).strip()
    if not text or text[0] in ("{", "["):
        raise ValueError("not plaintext fallback")
    if schema is ImproveResponse:
        return ImproveResponse(improved_text=text)
    if schema is RewriteResponse:
        return RewriteResponse(rewritten_text=text)
    if schema is GapCheckResponse:
        return GapCheckResponse(analysis=text)
    raise ValueError("not single-key schema")


def parse_model_output(raw: str, schema: type[Any]) -> Any:
    if not (raw and raw.strip()):
        raise ValueError("empty model output")
    t = clean_json(raw).lstrip()
    if schema is GapCheckResponse:
        wrapped = _gap_check_plaintext_fallback(raw)
        if wrapped is not None:
            return wrapped

    if schema in (ImproveResponse, RewriteResponse, GapCheckResponse) and not t.startswith(("{", "[")):
        return _plaintext_single_key_fallback(raw, schema)

    prepared, repair_notes = _prepare_for_json_parse(raw)
    if repair_notes:
        logger.info(
            "[ai-action-json-repair] notes=%s raw_preview=%s sanitized_preview=%s",
            repair_notes,
            _preview_for_log(raw),
            _preview_for_log(prepared),
        )

    clean_no_escape = strip_prologue_before_json(strip_sources_citations_noise(clean_json(raw)))
    candidates: list[str] = []
    if prepared:
        candidates.append(prepared)
    if clean_no_escape and clean_no_escape not in candidates:
        candidates.append(clean_no_escape)

    last_err: Exception | None = None
    for cand in candidates:
        try:
            data = _load_first_json_object_from_text(cand)
            return _coerce_model(data, schema)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            last_err = exc
            continue

    rec = _recovery_response_from_raw(prepared, schema) or _recovery_response_from_raw(raw, schema)
    if rec is not None:
        return rec

    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("model output not parseable", prepared, 0)


def _format_validation_errors(exc: ValidationError) -> str:
    parts: list[str] = []
    try:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()) if x != "__root__")
            msg = err.get("msg", "")
            if loc:
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(msg or str(err))
    except Exception:
        return str(exc)
    return "; ".join(parts) if parts else str(exc)


def _summarize_parse_error(exc: Exception, schema: type[Any]) -> str:
    if isinstance(exc, ValidationError):
        return _format_validation_errors(exc)
    return str(exc)


def _strict_json_retry_addon(schema: type[Any]) -> str:
    if schema is RewriteResponse:
        return (
            "\n\nYou returned invalid JSON. Return only this valid JSON object with escaped string content:\n"
            '{"rewritten_text":"your rewritten text here"}\n'
            "Do not include line breaks outside escaped \\n. Do not include markdown, code fences, explanation, or Sources."
        )
    if schema is ImproveResponse:
        return (
            "\n\nYou returned invalid JSON. Return only this valid JSON object with escaped string content:\n"
            '{"improved_text":"your improved text here"}\n'
            "Do not include line breaks outside escaped \\n. Do not include markdown, code fences, explanation, or Sources."
        )
    if schema is GapCheckResponse:
        return (
            "\n\nYou returned invalid JSON. Return only this valid JSON object with escaped string content:\n"
            '{"analysis":"your gap analysis here"}\n'
            "Do not include line breaks outside escaped \\n. Do not include markdown, code fences, explanation, or Sources."
        )
    return (
        "\n\nCRITICAL INSTRUCTION: Your previous response was not valid JSON or failed validation. "
        "Return ONLY a single JSON object matching the requested keys. No markdown fences. No explanation. "
        "No 'Sources' line. No text before the opening { or after the closing }."
    )


def _json_slice_heuristic(blob: str) -> str:
    """If the model wrapped JSON with noise, slice from first { to last }."""
    s = strip_sources_citations_noise(clean_json(blob))
    a = s.find("{")
    b = s.rfind("}")
    if a >= 0 and b > a:
        return s[a : b + 1]
    return s


def _final_schema_recovery(*blobs: str, schema: type[Any]) -> Any | None:
    for blob in blobs:
        if not (blob and blob.strip()):
            continue
        for variant in (blob, _json_slice_heuristic(blob)):
            rec = _recovery_response_from_raw(variant, schema)
            if rec is not None:
                return rec
    return None


def parse_with_retry(
    *,
    raw: str,
    schema: type[Any],
    prompt: str,
    call_llm,
    audit_log: list[dict[str, Any]],
) -> Any:
    schema_name = getattr(schema, "__name__", str(schema))
    soft = _action_prompt_soft_limit_chars()

    raw_in = raw
    if not (raw or "").strip():
        p0 = truncate_prompt_for_llm(prompt, soft) if len(prompt) > soft else prompt
        raw = invoke_llm_nonempty(
            call_llm,
            p0,
            schema_name=schema_name,
            audit_log=audit_log,
            step="initial_refetch",
        )
        if (raw or "").strip():
            logger.info(
                "[ai-action-llm-refetch] schema=%s got_len=%s (was_empty_len=%s)",
                schema_name,
                len(raw or ""),
                len(raw_in or ""),
            )

    if not (raw or "").strip():
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "The language model returned no text (empty response), even after a truncated retry. "
                    "The prompt may exceed your local model's context window, or the LLM server may be misconfigured."
                ),
                "schema": schema_name,
                "validation_or_parse_error": "empty model output",
                "hint": (
                    "Shorten the selected text, check LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL, "
                    "or lower ACTION_PROMPT_SOFT_LIMIT (recommended 6000-8000) so the server receives a smaller prompt."
                ),
            },
        )

    raw_preview = _preview_for_log(raw or "", 800)
    logger.info(
        "[ai-action-parse] attempt=1 schema=%s raw_len=%s raw_preview=%s",
        schema_name,
        len(raw or ""),
        raw_preview,
    )
    try:
        parsed = parse_model_output(raw, schema)
        audit_log.append({"event": "parse_success", "attempt": 1, "timestamp": utc_now_iso()})
        logger.info("[ai-action-parse-success] attempt=1 schema=%s", schema_name)
        return parsed
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        err_txt = _summarize_parse_error(exc, schema)
        logger.warning(
            "[ai-action-parse-failed] attempt=1 schema=%s error=%s",
            schema_name,
            err_txt,
        )
        audit_log.append(
            {
                "event": "parse_failed",
                "attempt": 1,
                "timestamp": utc_now_iso(),
                "error": err_txt,
            }
        )
        recovered = _final_schema_recovery(raw, schema=schema)
        if recovered is not None:
            audit_log.append(
                {"event": "parse_plaintext_recovery", "attempt": 1, "timestamp": utc_now_iso()},
            )
            logger.info("[ai-action-parse-success] attempt=1 schema=%s via=recovery", schema_name)
            return recovered

    retry_prompt = prompt + _strict_json_retry_addon(schema)
    retry_prompt_send = truncate_prompt_for_llm(retry_prompt, soft) if len(retry_prompt) > soft else retry_prompt
    retry_raw = invoke_llm_nonempty(
        call_llm,
        retry_prompt_send,
        schema_name=schema_name,
        audit_log=audit_log,
        step="retry_parse",
    )
    audit_log.append({"event": "llm_retry", "attempt": 2, "timestamp": utc_now_iso()})
    logger.info(
        "[ai-action-parse] attempt=2 schema=%s retry_raw_len=%s preview=%s",
        schema_name,
        len(retry_raw or ""),
        _preview_for_log(retry_raw or "", 800),
    )

    try:
        parsed = parse_model_output(retry_raw, schema)
        audit_log.append({"event": "parse_success", "attempt": 2, "timestamp": utc_now_iso()})
        logger.info("[ai-action-parse-success] attempt=2 schema=%s", schema_name)
        return parsed
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc2:
        err2 = _summarize_parse_error(exc2, schema)
        logger.warning(
            "[ai-action-parse-failed] attempt=2 schema=%s error=%s",
            schema_name,
            err2,
        )
        audit_log.append(
            {
                "event": "parse_failed",
                "attempt": 2,
                "timestamp": utc_now_iso(),
                "error": err2,
            }
        )
        recovered = _final_schema_recovery(retry_raw, raw, schema=schema)
        if recovered is not None:
            audit_log.append(
                {"event": "parse_plaintext_recovery", "attempt": 2, "timestamp": utc_now_iso()},
            )
            logger.info("[ai-action-parse-success] attempt=2 schema=%s via=recovery", schema_name)
            return recovered

    strict_extra = (
        "\n\nCRITICAL JSON RULES: Return exactly one compact JSON object. "
        "String values must be valid JSON strings: use \\n for newlines, \\\" for quotes, \\\\ for backslashes. "
        "Prefer a single line of JSON. No markdown. No text before or after the JSON object. "
        "Do not include Sources, citations, or CAPA tables outside the JSON."
    )
    if schema in (RewriteResponse, ImproveResponse, GapCheckResponse):
        strict_extra += (
            " If the section is long, shorten the wording slightly so the JSON stays small and well-formed."
        )
    strict_prompt = prompt + strict_extra + _strict_json_retry_addon(schema)
    strict_send = truncate_prompt_for_llm(strict_prompt, soft) if len(strict_prompt) > soft else strict_prompt
    third_raw = invoke_llm_nonempty(
        call_llm,
        strict_send,
        schema_name=schema_name,
        audit_log=audit_log,
        step="strict_parse",
    )
    audit_log.append({"event": "llm_retry", "attempt": 3, "timestamp": utc_now_iso()})
    logger.info(
        "[ai-action-parse] attempt=3 schema=%s third_raw_len=%s preview=%s",
        schema_name,
        len(third_raw or ""),
        _preview_for_log(third_raw or "", 800),
    )
    try:
        parsed = parse_model_output(third_raw, schema)
        audit_log.append({"event": "parse_success", "attempt": 3, "timestamp": utc_now_iso()})
        logger.info("[ai-action-parse-success] attempt=3 schema=%s", schema_name)
        return parsed
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc3:
        err3 = _summarize_parse_error(exc3, schema)
        logger.error(
            "[ai-action-parse-failed] attempt=3 schema=%s error=%s audit=%s",
            schema_name,
            err3,
            audit_log,
        )
        audit_log.append(
            {
                "event": "parse_failed",
                "attempt": 3,
                "timestamp": utc_now_iso(),
                "error": err3,
            }
        )
        recovered = _final_schema_recovery(third_raw, retry_raw, raw, schema=schema)
        if recovered is not None:
            audit_log.append(
                {"event": "parse_plaintext_recovery", "attempt": 3, "timestamp": utc_now_iso()},
            )
            logger.info("[ai-action-parse-success] attempt=3 schema=%s via=recovery", schema_name)
            return recovered

        third_empty = not (third_raw or "").strip()
        err_low = str(err3).lower()
        if third_empty or "empty model output" in err_low:
            user_msg = (
                "The language model returned no text or only whitespace (empty response). "
                "That usually means the combined prompt exceeded the local model's context window, "
                "or the server dropped the completion. Try a shorter selection, or raise the server's context limit."
            )
            hint = (
                "You can set ACTION_PROMPT_SOFT_LIMIT (recommended 6000-8000) in the backend environment to pre-truncate prompts. "
                "Also confirm LOCAL_LLM_BASE_URL and the model are reachable."
            )
        else:
            user_msg = "The model response could not be parsed into the required shape after 3 attempts."
            hint = (
                "Try a shorter selection, or select raw SOP text instead of a prior formatted AI report."
            )

        raise HTTPException(
            status_code=422,
            detail={
                "message": user_msg,
                "schema": schema_name,
                "validation_or_parse_error": err3,
                "hint": hint,
            },
        ) from exc3


def validate_convert_response(parsed: ConvertResponse) -> None:
    missing: list[str] = []
    if len(parsed.purpose.strip()) < 10:
        missing.append("purpose")
    if len(parsed.scope.strip()) < 10:
        missing.append("scope")
    if len(parsed.responsibilities.strip()) < 10:
        missing.append("responsibilities")
    if not parsed.procedure or not any(step.strip() for step in parsed.procedure):
        missing.append("procedure")
    if len(parsed.documentation.strip()) < 10:
        missing.append("documentation")

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Convert output is incomplete. Missing or empty sections: {', '.join(missing)}",
        )


def format_chunks(chunks: list[Document]) -> str:
    if not chunks:
        return "No relevant context found."

    formatted: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        source = (
            metadata.get("title")
            or metadata.get("sop_title")
            or metadata.get("source")
            or metadata.get("document_title")
            or metadata.get("sop_number")
            or f"Source {index}"
        )
        excerpt = chunk.page_content[:ACTION_CONTEXT_EXCERPT_CHARS].strip()
        formatted.append(f"[{source}]\n{excerpt}")
    return "\n\n".join(formatted)


def extract_source_titles(chunks: list[Document]) -> list[str]:
    seen: set[str] = set()
    titles: list[str] = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        title = (
            metadata.get("title")
            or metadata.get("sop_title")
            or metadata.get("source")
            or metadata.get("document_title")
            or metadata.get("sop_number")
        )
        if title and title not in seen:
            seen.add(title)
            titles.append(str(title))
    return titles
