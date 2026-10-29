# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from threading import RLock
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

_RUNTIME_SETTINGS_WRITE_LOCK = RLock()

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
        with _RUNTIME_SETTINGS_WRITE_LOCK:
            current = self.get_state().values
            updates = payload.model_dump(exclude_unset=True, exclude_none=True)
            candidate_payload = current.model_dump(mode="python")
            for category, category_updates in updates.items():
                candidate_payload[category].update(category_updates)

            try:
                RuntimeSettingsValues.model_validate(candidate_payload)
            except ValueError as exc:
                raise ServiceValidationError(str(exc)) from exc

            self._persist_values(updates)
            return self.get_state()

    # -------------------------------------------------------------------------
    def reset_category(
        self,
        category: RuntimeSettingsCategory,
    ) -> RuntimeSettingsStateResponse:
        with _RUNTIME_SETTINGS_WRITE_LOCK:
            current = self.get_state().values.model_dump(mode="python")
            defaults = getattr(RUNTIME_SETTINGS_DEFAULTS, category).model_dump(
                mode="python"
            )
            current[category] = defaults
            RuntimeSettingsValues.model_validate(current)
            self._persist_values({category: defaults})
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
        updates: dict[str, dict[str, Any]],
    ) -> None:
        serializer = self._serializer_or_create()
        try:
            def update_payload(payload: dict[str, Any]) -> None:
                if not payload:
                    payload.update(
                        build_default_application_configuration_payload(
                            environment_snapshot_from_os_env()
                        )
                    )
                if "general" in updates:
                    jobs = payload.get("jobs")
                    jobs = dict(jobs) if isinstance(jobs, dict) else {}
                    jobs.update(updates["general"])
                    payload["jobs"] = jobs
                if "data" in updates:
                    data_blocks = {
                        "ingestion": {
                            "drug_name_min_length",
                            "drug_name_max_length",
                            "drug_name_max_tokens",
                        },
                        "session_pipeline": {
                            "text_extraction_batch_size",
                            "text_extraction_max_concurrency",
                            "retrieval_batch_size",
                            "retrieval_max_concurrency",
                            "clinical_assessment_batch_size",
                            "clinical_assessment_max_concurrency",
                        },
                        "clinical_language_detection": {
                            "min_best_score",
                            "high_confidence_min_score",
                            "high_confidence_min_margin",
                            "moderate_confidence_min_score",
                            "moderate_confidence_min_margin",
                        },
                    }
                    for block_name, block_fields in data_blocks.items():
                        block = payload.get(block_name)
                        block = dict(block) if isinstance(block, dict) else {}
                        block.update(
                            {
                                key: value
                                for key, value in updates["data"].items()
                                if key in block_fields
                            }
                        )
                        payload[block_name] = block
                if "integrations" in updates or "advanced" in updates:
                    runtime = payload.get("runtime")
                    runtime = dict(runtime) if isinstance(runtime, dict) else {}
                    runtime.update(updates.get("integrations", {}))
                    runtime.update(updates.get("advanced", {}))
                    payload["runtime"] = runtime
                if "matching" in updates:
                    matching = payload.get("drugs_matcher")
                    matching = dict(matching) if isinstance(matching, dict) else {}
                    matching.update(updates["matching"])
                    payload["drugs_matcher"] = matching

            with _RUNTIME_SETTINGS_WRITE_LOCK:
                saved_payload = serializer.update_payload(update_payload)
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
                terminology_download_timeout=settings.runtime.terminology_download_timeout,
                drugcentral_enabled=settings.runtime.drugcentral_enabled,
                drugcentral_host=settings.runtime.drugcentral_host,
                drugcentral_port=settings.runtime.drugcentral_port,
                drugcentral_database=settings.runtime.drugcentral_database,
                drugcentral_user=settings.runtime.drugcentral_user,
                chembl_enabled=settings.runtime.chembl_enabled,
                chembl_max_phase=settings.runtime.chembl_max_phase,
                swissmedic_enabled=settings.runtime.swissmedic_enabled,
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
