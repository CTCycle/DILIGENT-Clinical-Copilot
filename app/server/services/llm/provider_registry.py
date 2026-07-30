from __future__ import annotations

import json
from pathlib import Path

from common.paths import CATALOGS_PATH
from domain.llm.providers import CloudProviderDefinition, CloudProviderId

###############################################################################
class ProviderRegistry:

    # -------------------------------------------------------------------------
    def __init__(
        self, definitions: tuple[CloudProviderDefinition, ...] | None = None
    ) -> None:
        self._definitions = definitions or self._load()
        self._by_id = {item.provider_id: item for item in self._definitions}
        if len(self._by_id) != len(self._definitions):
            raise ValueError("duplicate cloud provider id")

    # -------------------------------------------------------------------------
    @staticmethod
    def _load() -> tuple[CloudProviderDefinition, ...]:
        path = Path(CATALOGS_PATH) / "cloud_providers.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            CloudProviderDefinition.model_validate(item)
            for item in payload["providers"]
        )

    # -------------------------------------------------------------------------
    def all(self) -> tuple[CloudProviderDefinition, ...]:
        return self._definitions

    # -------------------------------------------------------------------------
    def get(self, provider_id: str) -> CloudProviderDefinition:
        try:
            return self._by_id[provider_id]  # type: ignore[index]
        except KeyError as exc:
            raise ValueError(f"Unsupported cloud provider: {provider_id}") from exc

    # -------------------------------------------------------------------------
    def is_valid_model(self, provider_id: CloudProviderId, model: str) -> bool:
        definition = self.get(provider_id)
        return not definition.models or model in definition.models


provider_registry = ProviderRegistry()
