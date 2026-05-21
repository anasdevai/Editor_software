"""
Webhook integration API — uses existing WEBHOOK_ENDPOINT_* and WEBHOOK_SECRET env vars.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from .services.webhook_config import get_webhook_secret, validate_webhook_configuration
from .services.webhook_service import (
    handle_entity_event,
    probe_webhook_endpoints,
    sync_all_from_webhooks,
)

webhook_router = APIRouter(prefix="/api/webhooks", tags=["Webhooks"])


class WebhookNotifyPayload(BaseModel):
    event: str = Field(..., description="created | updated | deleted | linked | imported")
    entity_type: str
    entity_id: uuid.UUID
    version_id: Optional[uuid.UUID] = None


def _verify_webhook_secret(x_webhook_secret: str | None) -> None:
    expected = get_webhook_secret()
    if not expected:
        return
    if not x_webhook_secret or x_webhook_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Webhook-Secret")


@webhook_router.get("/config")
def webhook_config():
    """Validate existing WEBHOOK_ENDPOINT_* / API_BASE_URL configuration."""
    return validate_webhook_configuration()


@webhook_router.get("/health")
async def webhook_health():
    """Probe each WEBHOOK_ENDPOINT_* URL (GET, read-only)."""
    return await probe_webhook_endpoints()


@webhook_router.post("/sync")
async def webhook_sync(
    background_tasks: BackgroundTasks,
    async_run: bool = True,
    x_webhook_secret: str | None = Header(default=None, alias="X-Webhook-Secret"),
):
    """
    Fetch all entities from configured webhook URLs and queue semantic reindex jobs.
    """
    _verify_webhook_secret(x_webhook_secret)

    if async_run:
        import asyncio

        def _run_sync() -> None:
            asyncio.run(sync_all_from_webhooks(process=True))

        background_tasks.add_task(_run_sync)
        return {"status": "started", "message": "Webhook sync running in background"}

    return await sync_all_from_webhooks(process=True)


@webhook_router.post("/notify")
def webhook_notify(
    payload: WebhookNotifyPayload,
    x_webhook_secret: str | None = Header(default=None, alias="X-Webhook-Secret"),
):
    """
    External or internal entity lifecycle notification → semantic pipeline / Qdrant CRUD.
    """
    _verify_webhook_secret(x_webhook_secret)
    try:
        return handle_entity_event(
            payload.event,
            payload.entity_type,
            payload.entity_id,
            payload.version_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
