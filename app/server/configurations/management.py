# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import os
from threading import RLock
from typing import Any

from pydantic import ValidationError

from configurations.database_config import database_settings, load_database_configuration
from common.constants import (
    DEFAULT_DRUG_MATCH_CATALOG_INDEX_LIMIT,
    DEFAULT_DRUG_MATCH_SPELLING_CONFIDENCE,
    DEFAULT_DRUG_MATCH_SPELLING_LONG_MAX_DISTANCE,
    DEFAULT_DRUG_MATCH_SPELLING_MIN_QUERY_LENGTH,
    DEFAULT_DRUG_MATCH_SPELLING_SHORT_MAX_DISTANCE,
    DEFAULT_DRUG_MATCH_TOKEN_MIN_LENGTH,
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_PORT,
    OLLAMA_DEFAULT_SCHEME,
)
from common.catalogs.model_choices import (
    get_cloud_model_choices,
    get_clinical_model_choices,
    get_text_extraction_model_choices,
)
from common.version import resolve_application_version
from common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_positive_int,
    coerce_str,
    coerce_str_or_none,
)
from domain.model_configs import ReasoningLevel
from domain.settings.configuration import (
    ClinicalLanguageDetectionSettings,
    DEFAULT_RAG_SETTINGS,
    DrugsMatcherSettings,
    FastAPISettings,
    IngestionSettings,
    JobsSettings,
    LLMRuntimeDefaults,
    RagSettings,
    RuntimeSettings,
    ServerSettings,
    SessionPipelineSettings,
)
from domain.settings.environment import EnvironmentSnapshot

###############################################################################
def ensure_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}

###############################################################################
class ConfigurationManager:

    # -------------------------------------------------------------------------
    def __init__(self, persisted_payload: dict[str, Any] | None = None) -> None:
        self._lock = RLock()
        self._persisted_payload = ensure_mapping(persisted_payload).copy()
        self._raw_data: dict[str, Any] = {}
        self._settings: ServerSettings | None = None
        self.reload()

    # -------------------------------------------------------------------------
    @property
    def server_settings(self) -> ServerSettings:
        with self._lock:
            if self._settings is None:
                self.reload()
            if self._settings is None:
                raise RuntimeError("Settings are not available.")
            return self._settings

    # -------------------------------------------------------------------------
    def reload(self, persisted_payload: dict[str, Any] | None = None) -> ServerSettings:
        with self._lock:
            if persisted_payload is not None:
                self._persisted_payload = ensure_mapping(persisted_payload).copy()
            loaded = self._persisted_payload
            payload = build_settings_payload(
                loaded,
                environment_snapshot_from_os_env(),
            )
            try:
                settings = ServerSettings.model_validate(payload)
            except ValidationError as exc:
                raise RuntimeError(f"Invalid configuration settings: {exc}") from exc
            self._raw_data = {
                **loaded,
                "rag": payload["rag"],
                "clinical_language_detection": payload["clinical_language_detection"],
            }
            self._settings = settings
            return settings

    # -------------------------------------------------------------------------
    def reload_from_database(self) -> ServerSettings:
        from repositories.serialization.application_configuration import (
            ApplicationConfigurationSerializer,
        )

        payload = ApplicationConfigurationSerializer().load() or {}
        return self.reload(payload)

    # -------------------------------------------------------------------------
    def get_block(self, block_name: str) -> dict[str, Any]:
        with self._lock:
            if self._settings is not None and hasattr(self._settings, block_name):
                value = getattr(self._settings, block_name)
                if hasattr(value, "model_dump"):
                    return value.model_dump(mode="python")
            return ensure_mapping(self._raw_data.get(block_name)).copy()

    # -------------------------------------------------------------------------
    def get_value(self, block_name: str, key: str, default: Any = None) -> Any:
        block = self.get_block(block_name)
        return block.get(key, default)

###############################################################################
def _resolve_ollama_url_with_scheme(
    normalized_host: str,
    *,
    port_value: int | None,
) -> str:
    scheme, host_port = normalized_host.split("://", maxsplit=1)
    if ":" in host_port:
        host_only, parsed_port = host_port.split(":", maxsplit=1)
        host_only = _normalize_ollama_host(host_only)
        resolved_port = (
            port_value
            if port_value is not None
            else coerce_int(parsed_port, OLLAMA_DEFAULT_PORT, minimum=1, maximum=65535)
        )
        return f"{scheme}://{host_only}:{resolved_port}"
    host_port = _normalize_ollama_host(host_port)
    resolved_port = port_value if port_value is not None else OLLAMA_DEFAULT_PORT
    return f"{scheme}://{host_port}:{resolved_port}"

###############################################################################
def _normalize_ollama_host(host: str) -> str:
    normalized = host.strip()
    if normalized.lower() in {"localhost", "::1", "[::1]"}:
        return "127.0.0.1"
    return normalized

###############################################################################
def resolve_ollama_base_url(
    *,
    ollama_url: str | None,
    ollama_host: str | None,
    ollama_port: int | None,
    fallback: str = (
        f"{OLLAMA_DEFAULT_SCHEME}://{OLLAMA_DEFAULT_HOST}:{OLLAMA_DEFAULT_PORT}"
    ),
) -> str:
    if ollama_url:
        return ollama_url.rstrip("/")
    host_value = coerce_str_or_none(ollama_host)
    port_value = ollama_port
    if host_value:
        normalized_host = _normalize_ollama_host(host_value.strip().rstrip("/"))
        if "://" in normalized_host:
            return _resolve_ollama_url_with_scheme(
                normalized_host, port_value=port_value
            )
        resolved_port = port_value if port_value is not None else OLLAMA_DEFAULT_PORT
        return f"{OLLAMA_DEFAULT_SCHEME}://{normalized_host}:{resolved_port}"
    if port_value is not None:
        return f"{OLLAMA_DEFAULT_SCHEME}://{OLLAMA_DEFAULT_HOST}:{port_value}"
    return fallback.rstrip("/")

###############################################################################
def environment_snapshot_from_os_env() -> EnvironmentSnapshot:
    raw_port = os.getenv("OLLAMA_PORT")
    port = (
        coerce_int(raw_port, OLLAMA_DEFAULT_PORT, minimum=1, maximum=65535)
        if raw_port
        else None
    )
    return EnvironmentSnapshot(
        ollama_url=coerce_str_or_none(os.getenv("OLLAMA_URL")),
        ollama_host=coerce_str_or_none(os.getenv("OLLAMA_HOST")),
        ollama_port=port,
    )

###############################################################################
def _default_llm_runtime_defaults(
    environment: EnvironmentSnapshot,
) -> LLMRuntimeDefaults:
    text_choices = get_text_extraction_model_choices()
    clinical_choices = get_clinical_model_choices()
    cloud_choices = get_cloud_model_choices()
    text_extraction_default = text_choices[0] if text_choices else ""
    clinical_default = clinical_choices[0] if clinical_choices else ""
    provider_default = "openai"
    provider_models = cloud_choices.get(provider_default, [])
    cloud_default = provider_models[0] if provider_models else ""
    return LLMRuntimeDefaults(
        text_extraction_model=text_extraction_default,
        clinical_model=clinical_default,
        llm_provider=provider_default,
        cloud_model=cloud_default,
        use_cloud_services=False,
        reasoning_level=ReasoningLevel.OFF,
        ollama_host_default=resolve_ollama_base_url(
            ollama_url=environment.ollama_url,
            ollama_host=environment.ollama_host,
            ollama_port=environment.ollama_port,
        ),
    )

###############################################################################
def _build_fastapi_settings() -> FastAPISettings:
    return FastAPISettings(
        title=FASTAPI_TITLE,
        version=resolve_application_version(),
        description=FASTAPI_DESCRIPTION,
    )

###############################################################################
def _build_jobs_settings(data: dict[str, Any]) -> JobsSettings:
    payload = ensure_mapping(data)
    polling_interval = coerce_float(payload.get("polling_interval"), 1.0)
    if polling_interval <= 0:
        polling_interval = 1.0
    return JobsSettings(polling_interval=polling_interval)

###############################################################################
def _build_drugs_matcher_settings(data: dict[str, Any]) -> DrugsMatcherSettings:
    return DrugsMatcherSettings(
        direct_confidence=coerce_float(data.get("direct_confidence"), 1.0),
        master_confidence=coerce_float(data.get("master_confidence"), 0.92),
        synonym_confidence=coerce_float(data.get("synonym_confidence"), 0.90),
        normalization_cache_limit=coerce_positive_int(
            data.get("normalization_cache_limit"), 10000
        ),
        match_cache_limit=coerce_positive_int(data.get("match_cache_limit"), 5000),
        alias_cache_limit=coerce_positive_int(data.get("alias_cache_limit"), 2000),
        min_confidence=coerce_float(data.get("min_confidence"), 0.90),
        token_min_length=coerce_positive_int(
            data.get("token_min_length"),
            DEFAULT_DRUG_MATCH_TOKEN_MIN_LENGTH,
        ),
        catalog_index_limit=coerce_positive_int(
            data.get("catalog_index_limit"),
            DEFAULT_DRUG_MATCH_CATALOG_INDEX_LIMIT,
        ),
        spelling_confidence=coerce_float(
            data.get("spelling_confidence"),
            DEFAULT_DRUG_MATCH_SPELLING_CONFIDENCE,
        ),
        spelling_min_query_length=coerce_positive_int(
            data.get("spelling_min_query_length"),
            DEFAULT_DRUG_MATCH_SPELLING_MIN_QUERY_LENGTH,
        ),
        spelling_short_name_length=coerce_positive_int(
            data.get("spelling_short_name_length"),
            8,
        ),
        spelling_short_max_distance=coerce_positive_int(
            data.get("spelling_short_max_distance"),
            DEFAULT_DRUG_MATCH_SPELLING_SHORT_MAX_DISTANCE,
        ),
        spelling_long_max_distance=coerce_positive_int(
            data.get("spelling_long_max_distance"),
            DEFAULT_DRUG_MATCH_SPELLING_LONG_MAX_DISTANCE,
        ),
    )

###############################################################################
def _build_rag_settings(
    data: dict[str, Any], defaults: LLMRuntimeDefaults
) -> RagSettings:
    selected_count = coerce_positive_int(
        data.get("retrieval_selected_count"),
        DEFAULT_RAG_SETTINGS["retrieval_selected_count"],
    )
    candidate_count = coerce_positive_int(
        data.get("retrieval_candidate_count"),
        DEFAULT_RAG_SETTINGS["retrieval_candidate_count"],
    )
    if candidate_count < selected_count:
        candidate_count = selected_count
    return RagSettings(
        allow_local_filesystem_access=coerce_bool(
            data.get("allow_local_filesystem_access"),
            DEFAULT_RAG_SETTINGS["allow_local_filesystem_access"],
        ),
        vector_collection_name=coerce_str(
            data.get("vector_collection_name"),
            DEFAULT_RAG_SETTINGS["vector_collection_name"],
        ),
        chunk_size=coerce_positive_int(
            data.get("chunk_size"), DEFAULT_RAG_SETTINGS["chunk_size"]
        ),
        chunk_overlap=coerce_positive_int(
            data.get("chunk_overlap"), DEFAULT_RAG_SETTINGS["chunk_overlap"]
        ),
        embedding_batch_size=coerce_positive_int(
            data.get("embedding_batch_size"),
            DEFAULT_RAG_SETTINGS["embedding_batch_size"],
        ),
        use_hybrid_search=coerce_bool(
            data.get("use_hybrid_search"), DEFAULT_RAG_SETTINGS["use_hybrid_search"]
        ),
        use_reranking=coerce_bool(
            data.get("use_reranking"), DEFAULT_RAG_SETTINGS["use_reranking"]
        ),
        retrieval_candidate_count=candidate_count,
        retrieval_selected_count=selected_count,
        reranker_model=coerce_str(
            data.get("reranker_model"), DEFAULT_RAG_SETTINGS["reranker_model"]
        ),
        hybrid_vector_weight=max(
            coerce_float(
                data.get("hybrid_vector_weight"),
                DEFAULT_RAG_SETTINGS["hybrid_vector_weight"],
            ),
            0.0,
        ),
        hybrid_text_weight=max(
            coerce_float(
                data.get("hybrid_text_weight"),
                DEFAULT_RAG_SETTINGS["hybrid_text_weight"],
            ),
            0.0,
        ),
        vector_index_metric=coerce_str(
            data.get("vector_index_metric"), DEFAULT_RAG_SETTINGS["vector_index_metric"]
        ),
        vector_index_type=coerce_str(
            data.get("vector_index_type"), DEFAULT_RAG_SETTINGS["vector_index_type"]
        ),
        vector_stream_batch_size=coerce_positive_int(
            data.get("vector_stream_batch_size"),
            DEFAULT_RAG_SETTINGS["vector_stream_batch_size"],
        ),
        embedding_offline_mode=coerce_bool(
            data.get("embedding_offline_mode"),
            DEFAULT_RAG_SETTINGS["embedding_offline_mode"],
        ),
    )

###############################################################################
def _build_runtime_settings(
    data: dict[str, Any], *, fallback_timeout: float
) -> RuntimeSettings:
    unified_llm_timeout = max(
        coerce_float(data.get("default_llm_timeout"), fallback_timeout),
        1.0,
    )
    parser_timeout = max(
        coerce_float(data.get("parser_llm_timeout"), unified_llm_timeout),
        1.0,
    )
    disease_timeout = max(
        coerce_float(data.get("disease_llm_timeout"), unified_llm_timeout),
        1.0,
    )
    clinical_timeout = max(
        coerce_float(data.get("clinical_llm_timeout"), unified_llm_timeout),
        1.0,
    )
    livertox_timeout = max(
        coerce_float(data.get("livertox_llm_timeout"), unified_llm_timeout), 1.0
    )
    minimum_llm_timeout = max(
        coerce_float(data.get("minimum_llm_timeout"), 5.0),
        1.0,
    )
    cloud_llm_timeout_cap = max(
        coerce_float(data.get("cloud_llm_timeout_cap"), 1800.0),
        minimum_llm_timeout,
    )
    local_llm_timeout_cap = max(
        coerce_float(data.get("local_llm_timeout_cap"), 45.0),
        minimum_llm_timeout,
    )
    return RuntimeSettings(
        default_llm_timeout=unified_llm_timeout,
        parser_llm_timeout=parser_timeout,
        disease_llm_timeout=disease_timeout,
        clinical_llm_timeout=clinical_timeout,
        livertox_llm_timeout=livertox_timeout,
        minimum_llm_timeout=minimum_llm_timeout,
        cloud_llm_timeout_cap=cloud_llm_timeout_cap,
        local_llm_timeout_cap=local_llm_timeout_cap,
        ollama_server_start_timeout=max(
            coerce_float(data.get("ollama_server_start_timeout"), 15.0), 1.0
        ),
        ncbi_contact_email=data.get("ncbi_contact_email", ""),
        livertox_download_timeout=max(
            coerce_float(data.get("livertox_download_timeout"), 30.0),
            1.0,
        ),
        livertox_archive=coerce_str(
            data.get("livertox_archive"), "livertox_NBK547852.tar.gz"
        ),
        livertox_yield_interval=coerce_positive_int(
            data.get("livertox_yield_interval"), 25
        ),
        livertox_skip_deterministic_ratio=coerce_float(
            data.get("livertox_skip_deterministic_ratio"), 0.80
        ),
        livertox_monograph_max_workers=coerce_positive_int(
            data.get("livertox_monograph_max_workers"), 4
        ),
        max_excerpt_length=coerce_positive_int(data.get("max_excerpt_length"), 8000),
        rxnav_request_timeout=coerce_float(data.get("rxnav_request_timeout"), 12.0),
        rxnav_max_concurrency=coerce_positive_int(
            data.get("rxnav_max_concurrency"), 10
        ),
    )

###############################################################################
def _build_ingestion_settings(data: dict[str, Any]) -> IngestionSettings:
    min_length = coerce_positive_int(data.get("drug_name_min_length"), 3)
    max_length = coerce_positive_int(data.get("drug_name_max_length"), 200)
    if max_length < min_length:
        max_length = min_length
    return IngestionSettings(
        drug_name_min_length=min_length,
        drug_name_max_length=max_length,
        drug_name_max_tokens=coerce_positive_int(data.get("drug_name_max_tokens"), 8),
    )

###############################################################################
def _build_session_pipeline_settings(data: dict[str, Any]) -> SessionPipelineSettings:
    payload = ensure_mapping(data)
    return SessionPipelineSettings(
        text_extraction_batch_size=coerce_positive_int(
            payload.get("text_extraction_batch_size"), 4
        ),
        text_extraction_max_concurrency=min(
            coerce_positive_int(payload.get("text_extraction_max_concurrency"), 2),
            4,
        ),
        retrieval_batch_size=coerce_positive_int(
            payload.get("retrieval_batch_size"), 8
        ),
        retrieval_max_concurrency=min(
            coerce_positive_int(payload.get("retrieval_max_concurrency"), 4),
            8,
        ),
        clinical_assessment_batch_size=coerce_positive_int(
            payload.get("clinical_assessment_batch_size"), 2
        ),
        clinical_assessment_max_concurrency=min(
            coerce_positive_int(payload.get("clinical_assessment_max_concurrency"), 2),
            4,
        ),
    )

###############################################################################
def _build_clinical_language_detection_settings(
    data: dict[str, Any],
) -> ClinicalLanguageDetectionSettings:
    return ClinicalLanguageDetectionSettings(
        min_best_score=coerce_float(data.get("min_best_score"), 2.0),
        high_confidence_min_score=coerce_float(
            data.get("high_confidence_min_score"), 8.0
        ),
        high_confidence_min_margin=coerce_float(
            data.get("high_confidence_min_margin"), 3.0
        ),
        moderate_confidence_min_score=coerce_float(
            data.get("moderate_confidence_min_score"), 4.0
        ),
        moderate_confidence_min_margin=coerce_float(
            data.get("moderate_confidence_min_margin"), 1.0
        ),
    )

###############################################################################
def build_settings_payload(
    config: dict[str, Any], env: EnvironmentSnapshot
) -> dict[str, Any]:
    payload = ensure_mapping(config)
    llm_defaults = _default_llm_runtime_defaults(env)
    jobs_payload = ensure_mapping(payload.get("jobs"))
    drugs_matcher_payload = ensure_mapping(payload.get("drugs_matcher"))
    rag_payload = ensure_mapping(payload.get("rag_settings"))
    runtime_payload = ensure_mapping(payload.get("runtime"))
    ingestion_payload = ensure_mapping(payload.get("ingestion"))
    session_pipeline_payload = ensure_mapping(payload.get("session_pipeline"))
    language_payload = ensure_mapping(payload.get("clinical_language_detection"))
    return {
        "fastapi": _build_fastapi_settings().model_dump(),
        "jobs": _build_jobs_settings(jobs_payload).model_dump(),
        "database": database_settings(load_database_configuration()).model_dump(),
        "drugs_matcher": _build_drugs_matcher_settings(
            drugs_matcher_payload
        ).model_dump(),
        "rag": _build_rag_settings(rag_payload, llm_defaults).model_dump(),
        "runtime": _build_runtime_settings(
            runtime_payload,
            fallback_timeout=3600.0,
        ).model_dump(),
        "ingestion": _build_ingestion_settings(ingestion_payload).model_dump(),
        "session_pipeline": _build_session_pipeline_settings(
            session_pipeline_payload
        ).model_dump(),
        "clinical_language_detection": _build_clinical_language_detection_settings(
            language_payload
        ).model_dump(),
        "llm_defaults": llm_defaults.model_dump(),
    }

###############################################################################
def build_default_application_configuration_payload(
    env: EnvironmentSnapshot,
) -> dict[str, Any]:
    """Return the complete database seed without reading legacy file state."""
    llm_defaults = _default_llm_runtime_defaults(env)
    payload = build_settings_payload({}, env)
    return {
        "clinical_model": llm_defaults.clinical_model,
        "text_extraction_model": llm_defaults.text_extraction_model,
        "revision_model": (
            llm_defaults.cloud_model
            if llm_defaults.use_cloud_services
            else llm_defaults.clinical_model
        ),
        "timeline_model": (
            llm_defaults.cloud_model
            if llm_defaults.use_cloud_services
            else llm_defaults.text_extraction_model
        ),
        "use_cloud_models": llm_defaults.use_cloud_services,
        "cloud_provider": llm_defaults.llm_provider,
        "cloud_model": llm_defaults.cloud_model,
        "reasoning_level": llm_defaults.reasoning_level.value,
        "ollama_seed": 42,
        "rag_settings": payload["rag"],
        "jobs": payload["jobs"],
        "runtime": payload["runtime"],
        "ingestion": payload["ingestion"],
        "session_pipeline": payload["session_pipeline"],
        "clinical_language_detection": payload["clinical_language_detection"],
        "drugs_matcher": payload["drugs_matcher"],
    }
