from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from common.exceptions import ServiceError, ServiceValidationError
from common.paths import CONFIGURATIONS_FILE
from configurations.management import ConfigurationManager
from configurations.runtime_updates import persist_configuration_blocks
from configurations.startup import get_configuration_manager
from domain.settings.configuration import ServerSettings
from domain.settings.runtime_ui import (
    RUNTIME_SETTINGS_DEFAULTS,
    AdvancedRuntimeSettings,
    DataRuntimeSettings,
    GeneralRuntimeSettings,
    IntegrationRuntimeSettings,
    RuntimeSettingsCategory,
    RuntimeSettingsStateResponse,
    RuntimeSettingsUpdateRequest,
    RuntimeSettingsValues,
)


class RuntimeSettingsService:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path or CONFIGURATIONS_FILE)

    def get_state(self) -> RuntimeSettingsStateResponse:
        settings = self._load_server_settings()
        return RuntimeSettingsStateResponse(
            values=self._values_from_server_settings(settings),
            defaults=RUNTIME_SETTINGS_DEFAULTS.model_copy(deep=True),
            updated_at=self._updated_at(),
        )

    def update_state(
        self,
        payload: RuntimeSettingsUpdateRequest,
    ) -> RuntimeSettingsStateResponse:
        current = self.get_state().values
        updates = payload.model_dump(exclude_unset=True, exclude_none=True)
        candidate_payload = current.model_dump(mode="python")
        for category, category_updates in updates.items():
            candidate_payload[category].update(category_updates)

        try:
            candidate = RuntimeSettingsValues.model_validate(candidate_payload)
        except ValueError as exc:
            raise ServiceValidationError(str(exc)) from exc

        self._persist_categories(candidate, set(updates))
        return self.get_state()

    def reset_category(
        self,
        category: RuntimeSettingsCategory,
    ) -> RuntimeSettingsStateResponse:
        current = self.get_state().values.model_dump(mode="python")
        current[category] = getattr(RUNTIME_SETTINGS_DEFAULTS, category).model_dump(
            mode="python"
        )
        candidate = RuntimeSettingsValues.model_validate(current)
        self._persist_categories(candidate, {category})
        return self.get_state()

    def _load_server_settings(self) -> ServerSettings:
        try:
            if self.config_path.resolve() == Path(CONFIGURATIONS_FILE).resolve():
                return get_configuration_manager().server_settings
            return ConfigurationManager(config_path=str(self.config_path)).server_settings
        except RuntimeError as exc:
            raise ServiceError(
                "Runtime settings could not be loaded.",
                retryable=True,
            ) from exc

    def _persist_categories(
        self,
        values: RuntimeSettingsValues,
        categories: set[str],
    ) -> None:
        block_updates: dict[str, dict[str, object]] = {}
        if "general" in categories:
            block_updates["jobs"] = values.general.model_dump(mode="python")
        if "data" in categories:
            block_updates["ingestion"] = values.data.model_dump(mode="python")
        if "integrations" in categories:
            block_updates.setdefault("runtime", {}).update(
                values.integrations.model_dump(mode="python")
            )
        if "advanced" in categories:
            block_updates.setdefault("runtime", {}).update(
                values.advanced.model_dump(mode="python")
            )

        try:
            persist_configuration_blocks(self.config_path, block_updates)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ServiceError(
                "Runtime settings could not be saved.",
                retryable=True,
            ) from exc

    @staticmethod
    def _values_from_server_settings(settings: ServerSettings) -> RuntimeSettingsValues:
        return RuntimeSettingsValues(
            general=GeneralRuntimeSettings(
                polling_interval=settings.jobs.polling_interval,
            ),
            data=DataRuntimeSettings(
                drug_name_min_length=settings.ingestion.drug_name_min_length,
                drug_name_max_length=settings.ingestion.drug_name_max_length,
                drug_name_max_tokens=settings.ingestion.drug_name_max_tokens,
            ),
            integrations=IntegrationRuntimeSettings(
                livertox_download_timeout=settings.runtime.livertox_download_timeout,
                rxnav_request_timeout=settings.runtime.rxnav_request_timeout,
                rxnav_max_concurrency=settings.runtime.rxnav_max_concurrency,
            ),
            advanced=AdvancedRuntimeSettings(
                default_llm_timeout=settings.runtime.default_llm_timeout,
                clinical_llm_timeout=settings.runtime.clinical_llm_timeout,
                livertox_llm_timeout=settings.runtime.livertox_llm_timeout,
                minimum_llm_timeout=settings.runtime.minimum_llm_timeout,
                cloud_llm_timeout_cap=settings.runtime.cloud_llm_timeout_cap,
                local_llm_timeout_cap=settings.runtime.local_llm_timeout_cap,
                max_excerpt_length=settings.runtime.max_excerpt_length,
            ),
        )

    def _updated_at(self) -> datetime | None:
        try:
            timestamp = self.config_path.stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
