"""
Resolve and validate WEBHOOK_ENDPOINT_* / API_BASE_URL from environment.

Uses existing .env keys only — no duplicate endpoint configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# Existing env keys (see backend/.env)
ENV_WEBHOOK_SOPS = "WEBHOOK_ENDPOINT_SOPS"
ENV_WEBHOOK_DEVIATIONS = "WEBHOOK_ENDPOINT_DEVIATIONS"
ENV_WEBHOOK_CAPAS = "WEBHOOK_ENDPOINT_CAPAS"
ENV_WEBHOOK_DECISIONS = "WEBHOOK_ENDPOINT_DECISIONS"
ENV_WEBHOOK_AUDITS = "WEBHOOK_ENDPOINT_AUDITS"
ENV_WEBHOOK_SECRET = "WEBHOOK_SECRET"
ENV_API_BASE = "API_BASE_URL"

# Relative paths when only API_BASE_URL + ENDPOINT_* are set
ENV_ENDPOINT_SOPS = "ENDPOINT_SOPS"
ENV_ENDPOINT_DEVIATIONS = "ENDPOINT_DEVIATIONS"
ENV_ENDPOINT_CAPAS = "ENDPOINT_CAPAS"
ENV_ENDPOINT_DECISIONS = "ENDPOINT_DECISIONS"
ENV_ENDPOINT_AUDITS = "ENDPOINT_AUDITS"

ENTITY_WEBHOOK_KEYS: dict[str, tuple[str, str, str]] = {
    "sops": ("sop", ENV_WEBHOOK_SOPS, ENV_ENDPOINT_SOPS),
    "deviations": ("deviation", ENV_WEBHOOK_DEVIATIONS, ENV_ENDPOINT_DEVIATIONS),
    "capas": ("capa", ENV_WEBHOOK_CAPAS, ENV_ENDPOINT_CAPAS),
    "decisions": ("decision", ENV_WEBHOOK_DECISIONS, ENV_ENDPOINT_DECISIONS),
    "audits": ("audit_finding", ENV_WEBHOOK_AUDITS, ENV_ENDPOINT_AUDITS),
}


@dataclass(frozen=True)
class WebhookEndpointConfig:
    entity_key: str
    entity_type: str
    url: str
    source: str  # "webhook_env" | "api_base_fallback"


def _api_base_url() -> str:
    return (os.getenv(ENV_API_BASE) or "http://127.0.0.1:8001").strip().rstrip("/")


def normalize_internal_url(url: str) -> str:
    """
    Map docker-style hosts (backend:8000) to the configured API_BASE_URL host/port
    so local uvicorn on 8001 works with the same .env.
    """
    raw = (url or "").strip()
    if not raw:
        return raw
    base = _api_base_url()
    if not base:
        return raw
    # Full URL: align host/port with API_BASE_URL when pointing at legacy backend:8000
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        base_parsed = urlparse(base)
        host = (parsed.hostname or "").lower()
        if host in ("backend", "localhost", "127.0.0.1") and (
            parsed.port in (8000, 8001, None) or (host == "backend" and parsed.port != base_parsed.port)
        ):
            path = parsed.path or ""
            query = f"?{parsed.query}" if parsed.query else ""
            port = base_parsed.port
            netloc = base_parsed.hostname or "127.0.0.1"
            if port and port not in (80, 443):
                netloc = f"{netloc}:{port}"
            scheme = base_parsed.scheme or parsed.scheme or "http"
            return f"{scheme}://{netloc}{path}{query}"
        return raw
    # Relative path
    return f"{base}/{raw.lstrip('/')}"


def resolve_entity_webhook_url(entity_key: str) -> WebhookEndpointConfig | None:
    if entity_key not in ENTITY_WEBHOOK_KEYS:
        return None
    entity_type, webhook_env, endpoint_env = ENTITY_WEBHOOK_KEYS[entity_key]
    explicit = (os.getenv(webhook_env) or "").strip()
    if explicit:
        return WebhookEndpointConfig(
            entity_key=entity_key,
            entity_type=entity_type,
            url=normalize_internal_url(explicit),
            source="webhook_env",
        )
    rel = (os.getenv(endpoint_env) or "").strip()
    if rel:
        return WebhookEndpointConfig(
            entity_key=entity_key,
            entity_type=entity_type,
            url=normalize_internal_url(rel),
            source="api_base_fallback",
        )
    return None


def get_all_webhook_endpoints() -> list[WebhookEndpointConfig]:
    out: list[WebhookEndpointConfig] = []
    for key in ENTITY_WEBHOOK_KEYS:
        cfg = resolve_entity_webhook_url(key)
        if cfg and cfg.url:
            out.append(cfg)
    return out


def get_webhook_secret() -> str:
    return (os.getenv(ENV_WEBHOOK_SECRET) or "").strip()


def validate_webhook_configuration() -> dict[str, Any]:
    """Return structured validation for health endpoint and startup logs."""
    base = _api_base_url()
    endpoints = get_all_webhook_endpoints()
    return {
        "api_base_url": base,
        "webhook_secret_configured": bool(get_webhook_secret()),
        "endpoints": [
            {
                "entity_key": e.entity_key,
                "entity_type": e.entity_type,
                "url": e.url,
                "source": e.source,
            }
            for e in endpoints
        ],
        "complete": len(endpoints) == len(ENTITY_WEBHOOK_KEYS),
    }
