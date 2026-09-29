from __future__ import annotations

from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from common.exceptions import ServiceError, ServiceValidationError
from configurations.management import (
    build_default_application_configuration_payload,
    environment_snapshot_from_os_env,
)
from configurations.startup import get_configuration_manager, get_server_settings
from domain.settings.configuration import ServerSettings
from domain.settings.runtime_ui import (
    RUNTIME_SETTINGS_DEFAULTS,
    AdvancedRuntimeSettings,
    DataRuntimeSettings,
    GeneralRuntimeSettings,
    IntegrationRuntimeSettings,
    MatchingRuntimeSettings,
    RuntimeSettingsCategory,
    RuntimeSettingsStateResponse,
    RuntimeSettingsUpdateRequest,
    RuntimeSettingsValues,
)
from repositories.serialization.application_configuration import (
    ApplicationConfigurationSerializer,
)


###############################################################################
class RuntimeSettingsService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        serializer: ApplicationConfigurationSerializer | None = None,
    ) -> None:
        self._serializer = serializer

    # -------------------------------------------------------------------------
    def get_state(self) -> RuntimeSettingsStateResponse:
        settings = self._load_server_settings()
        return RuntimeSettingsStateResponse(
            values=self._values_from_server_settings(settings),
            defaults=RUNTIME_SETTINGS_DEFAULTS.model_copy(deep=True),
            updated_at=self._updated_at(),
        )

    # -------------------------------------------------------------------------
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

        self._persist_values(candidate, set(updates))
        return self.get_state()

    # -------------------------------------------------------------------------
    def reset_category(
        self,
        category: RuntimeSettingsCategory,
    ) -> RuntimeSettingsStateResponse:
        current = self.get_state().values.model_dump(mode="python")
        current[category] = getattr(RUNTIME_SETTINGS_DEFAULTS, category).model_dump(
            mode="python"
        )
        candidate = RuntimeSettingsValues.model_validate(current)
        self._persist_values(candidate, {category})
        return self.get_state()

    # -------------------------------------------------------------------------
    def _load_server_settings(self) -> ServerSettings:
        try:
            return get_server_settings()
        except RuntimeError as exc:
            raise ServiceError(
                "Runtime settings could not be loaded.",
                retryable=True,
            ) from exc

    # -------------------------------------------------------------------------
    def _serializer_or_create(self) -> ApplicationConfigurationSerializer:
        if self._serializer is None:
            self._serializer = ApplicationConfigurationSerializer()
        return self._serializer

    # -------------------------------------------------------------------------
    def _persist_values(
        self,
        values: RuntimeSettingsValues,
        categories: set[str],
    ) -> None:
        serializer = self._serializer_or_create()
        try:
            persisted_payload = serializer.load() or (
                build_default_application_configuration_payload(
                    environment_snapshot_from_os_env()
                )
            )
            payload = dict(persisted_payload)
            if "general" in categories:
                payload["jobs"] = values.general.model_dump(mode="python")
            if "data" in categories:
                payload["ingestion"] = {
                    key: getattr(values.data, key)
                    for key in (
                        "drug_name_min_length",
                        "drug_name_max_length",
                        "drug_name_max_tokens",
                    )
                }
                payload["session_pipeline"] = {
                    key: getattr(values.data, key)
                    for key in (
                        "text_extraction_batch_size",
                        "text_extraction_max_concurrency",
                        "retrieval_batch_size",
                        "retrieval_max_concurrency",
                        "clinical_assessment_batch_size",
                        "clinical_assessment_max_concurrency",
                    )
                }
                payload["clinical_language_detection"] = {
                    key: getattr(values.data, key)
                    for key in (
                        "min_best_score",
                        "high_confidence_min_score",
                        "high_confidence_min_margin",
                        "moderate_confidence_min_score",
                        "moderate_confidence_min_margin",
                    )
                }
            if "integrations" in categories:
                payload.setdefault("runtime", {}).update(
                    values.integrations.model_dump(mode="python")
                )
            if "advanced" in categories:
                payload.setdefault("runtime", {}).update(
                    values.advanced.model_dump(mode="python")
                )
            if "matching" in categories:
                payload["drugs_matcher"] = values.matching.model_dump(mode="python")

            saved_payload = serializer.save(payload)
            get_configuration_manager().reload(saved_payload)
        except (OSError, RuntimeError, ValueError, SQLAlchemyError) as exc:
            raise ServiceError(
                "Runtime settings could not be saved.",
                retryable=True,
            ) from exc

    # -------------------------------------------------------------------------
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
                text_extraction_batch_size=settings.session_pipeline.text_extraction_batch_size,
                text_extraction_max_concurrency=settings.session_pipeline.text_extraction_max_concurrency,
                retrieval_batch_size=settings.session_pipeline.retrieval_batch_size,
                retrieval_max_concurrency=settings.session_pipeline.retrieval_max_concurrency,
                clinical_assessment_batch_size=settings.session_pipeline.clinical_assessment_batch_size,
                clinical_assessment_max_concurrency=settings.session_pipeline.clinical_assessment_max_concurrency,
                min_best_score=settings.clinical_language_detection.min_best_score,
                high_confidence_min_score=settings.clinical_language_detection.high_confidence_min_score,
                high_confidence_min_margin=settings.clinical_language_detection.high_confidence_min_margin,
                moderate_confidence_min_score=settings.clinical_language_detection.moderate_confidence_min_score,
                moderate_confidence_min_margin=settings.clinical_language_detection.moderate_confidence_min_margin,
            ),
            integrations=IntegrationRuntimeSettings(
                ncbi_contact_email=settings.runtime.ncbi_contact_email,
                livertox_download_timeout=settings.runtime.livertox_download_timeout,
                livertox_archive=settings.runtime.livertox_archive,
                livertox_yield_interval=settings.runtime.livertox_yield_interval,
                livertox_skip_deterministic_ratio=settings.runtime.livertox_skip_deterministic_ratio,
                livertox_monograph_max_workers=settings.runtime.livertox_monograph_max_workers,
                rxnav_request_timeout=settings.runtime.rxnav_request_timeout,
                rxnav_max_concurrency=settings.runtime.rxnav_max_concurrency,
            ),
            matching=MatchingRuntimeSettings(
                **settings.drugs_matcher.model_dump(mode="python")
            ),
            advanced=AdvancedRuntimeSettings(
                default_llm_timeout=settings.runtime.default_llm_timeout,
                parser_llm_timeout=settings.runtime.parser_llm_timeout,
                disease_llm_timeout=settings.runtime.disease_llm_timeout,
                clinical_llm_timeout=settings.runtime.clinical_llm_timeout,
                livertox_llm_timeout=settings.runtime.livertox_llm_timeout,
                minimum_llm_timeout=settings.runtime.minimum_llm_timeout,
                cloud_llm_timeout_cap=settings.runtime.cloud_llm_timeout_cap,
                local_llm_timeout_cap=settings.runtime.local_llm_timeout_cap,
                ollama_server_start_timeout=settings.runtime.ollama_server_start_timeout,
                max_excerpt_length=settings.runtime.max_excerpt_length,
            ),
        )

    # -------------------------------------------------------------------------
    def _updated_at(self) -> Any:
        try:
            _, timestamp = self._serializer_or_create().load_with_metadata()
            return timestamp
        except (OSError, RuntimeError, SQLAlchemyError):
            return None
