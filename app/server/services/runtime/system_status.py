# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

import httpx

from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.llm.providers import CloudProviderDescriptor
from domain.system_status import (
    MemoryUsage,
    ProviderAvailability,
    SystemStatusResponse,
)
from services.llm.model_config import ModelConfigService
from services.runtime.system_telemetry import SystemTelemetry
from repositories.serialization.access_keys import AccessKeySerializer

OLLAMA_PROBE_TIMEOUT_S = 1.5
OLLAMA_PROBE_TTL_S = 10.0
PROVIDER_TTL_S = 10.0
VRAM_TTL_S = 5.0
USABLE_CATALOG_STATUSES = frozenset({"available", "cached"})

###############################################################################
class SystemStatusService:
    """Aggregates cheap, cached readings for the global status bar."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        model_config: ModelConfigService | None = None,
        telemetry: SystemTelemetry | None = None,
        access_keys: AccessKeySerializer | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfigService()
        self.telemetry = telemetry or SystemTelemetry()
        self.access_keys = access_keys or AccessKeySerializer()
        self._ollama_cache: tuple[float, bool] | None = None
        self._providers_cache: tuple[float, ProviderAvailability] | None = None
        self._vram_cache: tuple[float, MemoryUsage | None] | None = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    async def get_status(self) -> SystemStatusResponse:
        async with self._lock:
            providers, ollama_reachable, cpu, ram, vram = await asyncio.gather(
                self._cached_providers(),
                self._cached_ollama(),
                asyncio.to_thread(self.telemetry.read_cpu_percent),
                asyncio.to_thread(self.telemetry.read_ram),
                self._cached_vram(),
            )
        return SystemStatusResponse(
            providers=providers,
            ollama="reachable" if ollama_reachable else "unreachable",
            cpu_percent=cpu,
            ram=ram,
            vram=vram,
        )

    # -------------------------------------------------------------------------
    async def _cached_providers(self) -> ProviderAvailability:
        return await self._cached("_providers_cache", PROVIDER_TTL_S, self._count_providers)

    # -------------------------------------------------------------------------
    async def _cached_ollama(self) -> bool:
        return await self._cached("_ollama_cache", OLLAMA_PROBE_TTL_S, self._probe_ollama)

    # -------------------------------------------------------------------------
    async def _cached_vram(self) -> MemoryUsage | None:
        return await self._cached(
            "_vram_cache",
            VRAM_TTL_S,
            lambda: asyncio.to_thread(self.telemetry.read_vram),
        )

    # -------------------------------------------------------------------------
    async def _cached(self, attribute: str, ttl_s: float, loader: Callable[[], Awaitable]):
        cached = getattr(self, attribute)
        now = time.monotonic()
        if cached is not None and now - cached[0] < ttl_s:
            return cached[1]
        value = await loader()
        setattr(self, attribute, (now, value))
        return value

    # -------------------------------------------------------------------------
    async def _probe_ollama(self) -> bool:
        base_url = get_server_settings().llm_defaults.ollama_host_default.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_PROBE_TIMEOUT_S) as client:
                response = await client.get(f"{base_url}/api/tags")
                return response.is_success
        except httpx.HTTPError:
            return False

    # -------------------------------------------------------------------------
    async def _count_providers(self) -> ProviderAvailability:
        """Cloud providers with an active key and a usable cached catalog."""
        snapshot = self.model_config.load_current_snapshot()
        descriptors = await self.model_config.discover_provider_descriptors(snapshot)
        active = sum(1 for item in descriptors if self._is_operational(item))
        return ProviderAvailability(active=active, total=len(descriptors))

    # -------------------------------------------------------------------------
    def _is_operational(self, descriptor: CloudProviderDescriptor) -> bool:
        if descriptor.catalog_status not in USABLE_CATALOG_STATUSES:
            return False
        try:
            return self.access_keys.get_active_key(descriptor.credential_scope) is not None
        except Exception:
            logger.debug("Could not read access keys for %s", descriptor.id)
            return False
