from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any, Protocol


###############################################################################
class LLMClientRuntimeOwner(Protocol):
    client: Any | None
    model: str
    client_lock: asyncio.Lock
    client_loop_id: int | None
    client_provider: str | None
    runtime_revision: int


###############################################################################
def _get_runtime_signature(owner: LLMClientRuntimeOwner) -> tuple[str, str] | None:
    value = getattr(owner, "runtime_signature", None)
    return value if isinstance(value, tuple) else None


###############################################################################
def _set_runtime_signature(
    owner: LLMClientRuntimeOwner,
    signature: tuple[str, str] | None,
) -> None:
    if hasattr(owner, "runtime_signature"):
        setattr(owner, "runtime_signature", signature)


###############################################################################
def _set_retry_attempts(owner: LLMClientRuntimeOwner, provider: str) -> None:
    if hasattr(owner, "extraction_retry_attempts"):
        setattr(
            owner,
            "extraction_retry_attempts",
            4 if provider in {"openai", "gemini"} else 2,
        )


###############################################################################
def _needs_refresh(
    owner: LLMClientRuntimeOwner,
    *,
    normalized_provider: str,
    revision: int,
    signature: tuple[str, str] | None,
    current_loop_id: int,
    track_revision: bool,
    track_signature: bool,
) -> bool:
    revision_changed = track_revision and owner.runtime_revision != revision
    signature_changed = track_signature and _get_runtime_signature(owner) != signature
    return (
        owner.client is None
        or owner.client_provider != normalized_provider
        or revision_changed
        or signature_changed
        or owner.client_loop_id != current_loop_id
    )


###############################################################################
async def ensure_runtime_client(
    owner: LLMClientRuntimeOwner,
    *,
    provider: str,
    model: str,
    revision: int,
    client_factory: Callable[[str, str], Any],
    signature: tuple[str, str] | None = None,
    track_revision: bool = True,
    track_signature: bool = False,
) -> None:
    async with owner.client_lock:
        current_loop_id = id(asyncio.get_running_loop())
        normalized_provider = provider.strip()
        normalized_model = model.strip()
        if owner.client_provider == "injected" and owner.client is not None:
            owner.model = normalized_model
            owner.runtime_revision = revision
            _set_runtime_signature(owner, signature)
            owner.client_loop_id = current_loop_id
            return
        needs_refresh = _needs_refresh(
            owner,
            normalized_provider=normalized_provider,
            revision=revision,
            signature=signature,
            current_loop_id=current_loop_id,
            track_revision=track_revision,
            track_signature=track_signature,
        )
        if needs_refresh:
            if owner.client is not None:
                with contextlib.suppress(Exception):
                    await owner.client.close()
            owner.client = client_factory(normalized_provider, normalized_model)
            owner.client_provider = normalized_provider
            _set_retry_attempts(owner, normalized_provider)
        owner.runtime_revision = revision
        _set_runtime_signature(owner, signature)
        owner.client_loop_id = current_loop_id
        owner.model = normalized_model
        if (
            owner.client is not None
            and normalized_model
            and hasattr(owner.client, "default_model")
        ):
            owner.client.default_model = normalized_model  # type: ignore[attr-defined]
