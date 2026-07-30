from __future__ import annotations

import hashlib
import re

from pydantic import ValidationError

from common.constants import GEMINI_API_BASE, OPENAI_API_BASE
from common.utils.logger import logger
from configurations.startup import get_server_settings
from domain.llm.providers import CloudModelDescriptor
from domain.model_configs import CatalogProviderId, LocalCatalogMetadata
from repositories.serialization.access_keys import AccessKeySerializer
from repositories.serialization.provider_model_catalog_cache import (
    ProviderModelCatalogCacheRecord,
    ProviderModelCatalogCacheSerializer,
)
from services.llm.provider_registry import provider_registry


def catalog_configuration_fingerprint(provider: CatalogProviderId) -> str:
    if provider == "ollama":
        endpoint = str(
            get_server_settings().llm_defaults.ollama_host_default or ""
        ).strip()
        credential_fingerprint = "none"
        catalog_endpoint = "api/tags"
    else:
        definition = provider_registry.get(provider)
        endpoint_by_provider = {
            "openai": OPENAI_API_BASE,
            "gemini": GEMINI_API_BASE,
            "deepseek": "https://api.deepseek.com",
            "anthropic": "https://api.anthropic.com",
            "opencode_zen": "https://opencode.ai",
            "opencode_go": "https://opencode.ai",
        }
        endpoint = endpoint_by_provider[provider]
        catalog_endpoint = str(definition.models_endpoint or "").strip()
        try:
            active_key = AccessKeySerializer().get_active_key(
                definition.credential_scope
            )
            credential_fingerprint = str(
                getattr(active_key, "fingerprint", None) or "none"
            )
        except Exception:
            credential_fingerprint = "unavailable"
    raw = (
        f"{provider}|{endpoint.rstrip('/')}|{catalog_endpoint.rstrip('/')}|"
        f"{credential_fingerprint}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_catalog_record(
    catalog_cache: ProviderModelCatalogCacheSerializer, provider: CatalogProviderId
) -> ProviderModelCatalogCacheRecord | None:
    return catalog_cache.get(provider, catalog_configuration_fingerprint(provider))


def local_catalog_metadata(
    catalog_cache: ProviderModelCatalogCacheSerializer,
) -> LocalCatalogMetadata:
    record = load_catalog_record(catalog_cache, "ollama")
    if record is None:
        return LocalCatalogMetadata(
            status="not_loaded",
            message="Refresh Ollama to load installed models.",
        )
    if record.last_attempt_status == "success":
        return LocalCatalogMetadata(
            status="available",
            updated_at=record.last_success_at,
            message="Ollama catalog loaded from the saved cache.",
        )
    if record.models:
        return LocalCatalogMetadata(
            status="cached",
            updated_at=record.last_success_at,
            message=(
                "Showing installed models from the last successful Ollama refresh. "
                f"Latest refresh failed: {record.last_error or 'provider unavailable'}"
            ),
        )
    return LocalCatalogMetadata(
        status="unavailable",
        message=record.last_error or "Ollama is temporarily unavailable.",
    )


def cloud_models_from_record(
    record: ProviderModelCatalogCacheRecord | None,
) -> list[CloudModelDescriptor]:
    if record is None:
        return []
    models: list[CloudModelDescriptor] = []
    for item in record.models:
        try:
            models.append(CloudModelDescriptor.model_validate(item))
        except ValidationError:
            logger.warning("Ignoring malformed cached provider model entry.")
            continue
    return models


def sanitize_catalog_error(message: str) -> str:
    normalized = " ".join(message.split())
    normalized = re.sub(
        r"(?i)(api[-_ ]?key|authorization|bearer|token|secret|password)\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        normalized,
    )
    normalized = re.sub(r"(?i)\bbearer\s+\S+", "Bearer <redacted>", normalized)
    return normalized[:500] or "The provider model catalog could not be refreshed."
