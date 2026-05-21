from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import dotenv_values, load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


_LOGGER = logging.getLogger(__name__)
_LOGGED_ONCE = False
_LOCK = threading.Lock()
_CHAT_LLM_CACHE: dict[tuple[float, Optional[int], int, str, str], ChatOpenAI] = {}
_OPENAI_CLIENT_CACHE: dict[tuple[str, str, str], OpenAI] = {}
_ENV_FILE_PATH = Path(__file__).resolve().parents[2] / ".env"

# Ensure chatbot runtime uses repository .env values even when shell env
# contains stale overrides from earlier sessions.
load_dotenv(_ENV_FILE_PATH, override=True)


@dataclass(frozen=True)
class LocalLLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str


def _file_env() -> dict[str, str]:
    raw = dotenv_values(_ENV_FILE_PATH) if _ENV_FILE_PATH.exists() else {}
    return {str(k): str(v) if v is not None else "" for k, v in raw.items()}


def get_local_llm_config() -> LocalLLMConfig:
    file_env = _file_env()
    provider = str(file_env.get("LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "local_openai")).strip().lower()
    base_url = str(file_env.get("LOCAL_LLM_BASE_URL") or os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.56.1:1234/v1")).strip()
    model = str(file_env.get("LOCAL_LLM_MODEL") or os.getenv("LOCAL_LLM_MODEL", "qwen/qwen2.5-vl-7b:2")).strip()
    api_key = str(file_env.get("LOCAL_LLM_API_KEY") or os.getenv("LOCAL_LLM_API_KEY", "local-key")).strip() or "local-key"
    return LocalLLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )


def get_local_llm_timeout_seconds() -> int:
    """Single-request timeout for the OpenAI-compatible HTTP client (must match .env)."""
    fe = _file_env()
    raw = fe.get("LOCAL_LLM_TIMEOUT_SECONDS") or os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120")
    try:
        return max(5, int(float(raw)))
    except (TypeError, ValueError):
        return 120


def get_local_llm_health_models_timeout_seconds() -> float:
    """Timeout for GET /v1/models during health checks (remote LAN hosts may need >8s)."""
    fe = _file_env()
    raw = fe.get("LOCAL_LLM_HEALTH_MODELS_TIMEOUT") or os.getenv("LOCAL_LLM_HEALTH_MODELS_TIMEOUT", "45")
    try:
        return max(3.0, float(raw))
    except (TypeError, ValueError):
        return 45.0


def get_local_llm_health_chat_timeout_seconds() -> float:
    """Timeout for optional POST /v1/chat/completions probe."""
    fe = _file_env()
    raw = fe.get("LOCAL_LLM_HEALTH_CHAT_TIMEOUT") or os.getenv("LOCAL_LLM_HEALTH_CHAT_TIMEOUT", "60")
    try:
        return max(5.0, float(raw))
    except (TypeError, ValueError):
        return 60.0


def get_chat_pipeline_timeout_seconds() -> int:
    """
    Outer asyncio budget for the full RAG thread (retrieval + rerank + LLM).
    Defaults to LOCAL_LLM_TIMEOUT + retrieval budget so wait_for does not cancel
    a slow local model before ChatOpenAI's own timeout.
    """
    fe = _file_env()
    llm_t = get_local_llm_timeout_seconds()
    rag_budget = int(float(fe.get("RAG_RETRIEVAL_BUDGET_SECONDS") or os.getenv("RAG_RETRIEVAL_BUDGET_SECONDS", "90")))
    pad = int(float(fe.get("CHAT_PIPELINE_PAD_SECONDS") or os.getenv("CHAT_PIPELINE_PAD_SECONDS", "30")))
    explicit = fe.get("CHAT_QUERY_TIMEOUT_SECONDS") or os.getenv("CHAT_QUERY_TIMEOUT_SECONDS")
    if explicit is not None and str(explicit).strip() != "":
        try:
            ex = int(float(explicit))
        except (TypeError, ValueError):
            ex = llm_t + rag_budget + pad
        return max(ex, llm_t + 15)
    return llm_t + rag_budget + pad


def _connectivity_troubleshooting(models_error: str | None, models_url: str | None = None) -> list[str]:
    if not models_error:
        return []
    err = (models_error or "").lower()
    hints: list[str] = []
    probe = f" Try: curl {models_url}" if models_url else ""
    if "timed out" in err or "timeout" in err:
        hints.extend(
            [
                "TCP to the LLM host did not complete in time. This is usually network/firewall, not the model name.",
                "On the PC running LM Studio / llama.cpp: enable 'Listen on network' (or bind to 0.0.0.0), not only 127.0.0.1.",
                "On that same PC: allow inbound TCP on the server port in the OS firewall (Private network profile).",
                f"From this machine (where uvicorn runs): ping the LLM host IP; then open or curl the models URL.{probe}",
                "Both machines must be on the same LAN or Host-Only segment (e.g. VirtualBox Host-Only adapters).",
                "Increase LOCAL_LLM_HEALTH_MODELS_TIMEOUT in .env if the first response is very slow.",
            ]
        )
    if "refused" in err or "unreachable" in err:
        hints.append("Connection refused: nothing is listening on that host:port, or a firewall is rejecting the SYN.")
    if "name or service not known" in err or "getaddrinfo" in err:
        hints.append("DNS/hostname resolution failed; try IP address only in LOCAL_LLM_BASE_URL.")
    return hints


def check_local_llm_api_health(
    *,
    chat_probe: bool = False,
    models_timeout: float | None = None,
    chat_timeout: float | None = None,
) -> dict[str, Any]:
    """
    Probe OpenAI-compatible /v1/models and optionally /v1/chat/completions.
    Uses sync urllib (same process as uvicorn worker).
    """
    if models_timeout is None:
        models_timeout = get_local_llm_health_models_timeout_seconds()
    if chat_timeout is None:
        chat_timeout = get_local_llm_health_chat_timeout_seconds()
    cfg = get_local_llm_config()
    base = cfg.base_url.rstrip("/")
    out: dict[str, Any] = {
        "llm_base_url": cfg.base_url,
        "llm_provider": cfg.provider,
        "llm_model": cfg.model,
        "models_url": f"{base}/models",
        "models_probe_timeout_seconds": models_timeout,
        "models_status": None,
        "model_ids": [],
        "configured_model_found": False,
        "models_error": None,
        "chat_completions_probe": None,
        "connectivity_troubleshooting": [],
    }
    try:
        req = urllib.request.Request(
            out["models_url"],
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=models_timeout) as resp:
            out["models_status"] = resp.status
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
            data = body.get("data") or []
            out["model_ids"] = [str(x.get("id")) for x in data if isinstance(x, dict) and x.get("id")]
            out["configured_model_found"] = cfg.model in out["model_ids"]
    except urllib.error.HTTPError as e:
        out["models_error"] = f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        out["models_error"] = f"{type(e).__name__}: {e}"

    out["connectivity_troubleshooting"] = _connectivity_troubleshooting(out.get("models_error"), out.get("models_url"))

    if chat_probe and out.get("model_ids"):
        probe_model = cfg.model if cfg.model in out["model_ids"] else out["model_ids"][0]
        url = f"{base}/chat/completions"
        payload = json.dumps(
            {
                "model": probe_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
                "temperature": 0,
            }
        ).encode("utf-8")
        chat_out: dict[str, Any] = {
            "url": url,
            "model": probe_model,
            "status": None,
            "error": None,
            "chat_probe_timeout_seconds": chat_timeout,
        }
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=chat_timeout) as resp:
                chat_out["status"] = resp.status
        except urllib.error.HTTPError as e:
            chat_out["error"] = f"HTTP {e.code}: {e.reason}"
            try:
                chat_out["body"] = e.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                pass
        except Exception as e:
            chat_out["error"] = f"{type(e).__name__}: {e}"
        out["chat_completions_probe"] = chat_out

    return out


def classify_llm_exception(exc: BaseException) -> str:
    """Stable machine-readable stage for API payloads and logs."""
    if isinstance(exc, TimeoutError):
        return "llm_timeout"
    try:
        import asyncio

        if isinstance(exc, asyncio.TimeoutError):
            return "llm_timeout"
    except Exception:
        pass
    msg = str(exc or "").lower()
    if "model" in msg and ("not found" in msg or "does not exist" in msg or "invalid" in msg or "unknown model" in msg):
        return "model_not_found"
    if "connection refused" in msg or "failed to establish a new connection" in msg or "name or service not known" in msg:
        return "connection_refused"
    if "timed out" in msg or "timeout" in msg:
        return "llm_timeout"
    if "error code: 401" in msg or "401" in msg and "unauthorized" in msg:
        return "auth_error"
    if "error code: 404" in msg:
        return "http_404"
    if "error code: 503" in msg or "status code: 503" in msg:
        return "llm_http_503"
    if "error code: 502" in msg:
        return "llm_http_502"
    return "llm_error"


def log_active_provider_once() -> None:
    global _LOGGED_ONCE
    if _LOGGED_ONCE:
        return
    cfg = get_local_llm_config()
    msg = f"[llm-provider] provider={cfg.provider} base_url={cfg.base_url} model={cfg.model}"
    print(msg, flush=True)
    _LOGGER.info(msg)
    _LOGGED_ONCE = True


def create_chat_llm(
    *,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    max_retries: int = 1,
    use_cache: bool = True,
) -> ChatOpenAI:
    cfg = get_local_llm_config()
    if cfg.provider != "local_openai":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{cfg.provider}'. Expected 'local_openai'."
        )
    log_active_provider_once()
    cache_key = (float(temperature), max_output_tokens, int(max_retries), cfg.base_url, cfg.model)
    if use_cache:
        with _LOCK:
            cached = _CHAT_LLM_CACHE.get(cache_key)
            if cached is not None:
                return cached

    llm_timeout = float(get_local_llm_timeout_seconds())
    llm = ChatOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        timeout=llm_timeout,
        max_retries=max_retries,
    )
    if use_cache:
        with _LOCK:
            _CHAT_LLM_CACHE[cache_key] = llm
    return llm


def create_openai_client(*, use_cache: bool = True) -> OpenAI:
    cfg = get_local_llm_config()
    if cfg.provider != "local_openai":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{cfg.provider}'. Expected 'local_openai'."
        )
    log_active_provider_once()
    cache_key = (cfg.provider, cfg.base_url, cfg.model)
    if use_cache:
        with _LOCK:
            cached = _OPENAI_CLIENT_CACHE.get(cache_key)
            if cached is not None:
                return cached
    client = OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout=float(get_local_llm_timeout_seconds()),
    )
    if use_cache:
        with _LOCK:
            _OPENAI_CLIENT_CACHE[cache_key] = client
    return client


def is_local_llm_unreachable_error(exc: BaseException) -> bool:
    """
    True when the chatbot cannot reach or use the configured local OpenAI endpoint.
    Narrower than before: do not treat arbitrary '503' substrings as unreachable.
    """
    if isinstance(exc, TimeoutError):
        return True
    try:
        import asyncio

        if isinstance(exc, asyncio.TimeoutError):
            return True
    except Exception:
        pass
    msg = str(exc or "").lower()
    markers = (
        "connection refused",
        "max retries exceeded",
        "timed out",
        "timeout",
        "failed to establish a new connection",
        "connecterror",
        "apiconnectionerror",
        "service unavailable",
        "error code: 503",
        "status code: 503",
        "error code: 502",
        "status code: 502",
        "error code: 504",
        "name or service not known",
        "nodename nor servname",
        "could not resolve host",
    )
    return any(m in msg for m in markers)

