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
    OllamaControl,
    ProviderAvailability,
    SystemStatusResponse,
)
from services.llm.model_config import ModelConfigService
from services.runtime.ollama_process import OllamaProcessManager, ollama_process_manager
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
        ollama_process: OllamaProcessManager | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfigService()
        self.telemetry = telemetry or SystemTelemetry()
        self.access_keys = access_keys or AccessKeySerializer()
        self.ollama_process = ollama_process or ollama_process_manager
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
            ollama_control=self._ollama_control(ollama_reachable),
            cpu_percent=cpu,
            ram=ram,
            vram=vram,
        )

    # -------------------------------------------------------------------------
    def _ollama_control(self, reachable: bool) -> OllamaControl:
        process = self.ollama_process
        base_url = self._ollama_base_url()
        executable_found = process.find_executable() is not None
        local = process.is_local_url(base_url)
        if process.starting:
            state = "starting"
        elif reachable:
            state = "running"
        elif process.last_error:
            state = "error"
        else:
            state = "stopped"
        return OllamaControl(
            state=state,
            managed=process.owned,
            can_start=not reachable and not process.starting and local and executable_found,
            can_stop=process.owned and not process.starting,
            executable_found=executable_found,
            error=process.last_error if not reachable else None,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _ollama_base_url() -> str:
        return get_server_settings().llm_defaults.ollama_host_default.rstrip("/")

    # -------------------------------------------------------------------------
    async def start_ollama(self) -> None:
        """Launch the local Ollama service and refresh the cached probe."""
        try:
            await self.ollama_process.start(
                self._ollama_base_url(),
                is_online=self._probe_ollama,
                wait_timeout_s=get_server_settings().runtime.ollama_server_start_timeout,
                poll_interval_s=0.5,
            )
        finally:
            self._ollama_cache = None

    # -------------------------------------------------------------------------
    async def stop_ollama(self) -> bool:
        """Stop the Ollama process owned by this application, if any."""
        try:
            return await asyncio.to_thread(self.ollama_process.stop)
        finally:
            self._ollama_cache = None

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
        base_url = self._ollama_base_url()
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_PROBE_TIMEOUT_S) as client:
                response = await client.get(f"{base_url}/api/tags")
                return response.is_success
        except httpx.HTTPError:
            return False

    # -------------------------------------------------------------------------
    async def _count_providers(self) -> ProviderAvailability:
        """Cloud providers with an active key and a usable cached catalog."""
        descriptors = await self.model_config.discover_provider_descriptors()
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
