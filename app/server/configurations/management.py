from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import ValidationError

from common.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    CONFIGURATIONS_FILE,
    DEFAULT_DRUG_MATCH_CATALOG_EXCLUDED_TERM_SUFFIXES,
    DEFAULT_DRUG_MATCH_CATALOG_INDEX_LIMIT,
    DEFAULT_DRUG_MATCH_SPELLING_CONFIDENCE,
    DEFAULT_DRUG_MATCH_SPELLING_LONG_MAX_DISTANCE,
    DEFAULT_DRUG_MATCH_SPELLING_MIN_QUERY_LENGTH,
    DEFAULT_DRUG_MATCH_SPELLING_SHORT_MAX_DISTANCE,
    DEFAULT_DRUG_MATCH_SPELLING_SHORT_NAME_LENGTH,
    DEFAULT_DRUG_MATCH_TOKEN_MIN_LENGTH,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_PORT,
    OLLAMA_DEFAULT_SCHEME,
    TEXT_EXTRACTION_MODEL_CHOICES,
)
from common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_positive_int,
    coerce_str,
    coerce_str_or_none,
)
from domain.settings.configuration import (
    DatabaseSettings,
    DrugsMatcherSettings,
    ExternalDataSettings,
    FastAPISettings,
    IngestionSettings,
    JobsSettings,
    LLMRuntimeDefaults,
    RagSettings,
    SessionPipelineSettings,
    ServerSettings,
)


def ensure_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def load_configuration_data(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"Configuration file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load configuration from {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Configuration must be a JSON object.")
    return data


###############################################################################
###############################################################################
class EnvironmentSnapshot:
    def __init__(
        self,
        *,
        ollama_url: str | None,
        ollama_host: str | None,
        ollama_port: int | None,
    ) -> None:
        self.ollama_url = ollama_url
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port


###############################################################################
class ConfigurationManager:
    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = Path(config_path or CONFIGURATIONS_FILE)
        self._lock = RLock()
        self._raw_data: dict[str, Any] = {}
        self._settings: ServerSettings | None = None
        self.reload()

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def server_settings(self) -> ServerSettings:
        with self._lock:
            if self._settings is None:
                self.reload()
            if self._settings is None:
                raise RuntimeError("Settings are not available.")
            return self._settings

    def reload(self) -> ServerSettings:
        with self._lock:
            loaded = load_configuration_data(str(self._config_path))
            payload = build_settings_payload_from_json(
                loaded,
                environment_snapshot_from_os_env(),
            )
            try:
                settings = ServerSettings.model_validate(payload)
            except ValidationError as exc:
                raise RuntimeError(f"Invalid configuration settings: {exc}") from exc
            self._raw_data = loaded
            self._settings = settings
            return settings

    def get_block(self, block_name: str) -> dict[str, Any]:
        with self._lock:
            return ensure_mapping(self._raw_data.get(block_name)).copy()

    def get_value(self, block_name: str, key: str, default: Any = None) -> Any:
        block = self.get_block(block_name)
        return block.get(key, default)


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


def _normalize_ollama_host(host: str) -> str:
    normalized = host.strip()
    if normalized.lower() in {"localhost", "::1", "[::1]"}:
        return "127.0.0.1"
    return normalized


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


def _default_llm_runtime_defaults(
    environment: EnvironmentSnapshot,
) -> LLMRuntimeDefaults:
    text_extraction_default = (
        TEXT_EXTRACTION_MODEL_CHOICES[0] if TEXT_EXTRACTION_MODEL_CHOICES else ""
    )
    clinical_default = CLINICAL_MODEL_CHOICES[0] if CLINICAL_MODEL_CHOICES else ""
    provider_default = "openai"
    provider_models = CLOUD_MODEL_CHOICES.get(provider_default, [])
    cloud_default = provider_models[0] if provider_models else ""
    return LLMRuntimeDefaults(
        text_extraction_model=text_extraction_default,
        clinical_model=clinical_default,
        llm_provider=provider_default,
        cloud_model=cloud_default,
        use_cloud_services=False,
        ollama_temperature=0.7,
        cloud_temperature=0.7,
        ollama_reasoning=False,
        ollama_host_default=resolve_ollama_base_url(
            ollama_url=environment.ollama_url,
            ollama_host=environment.ollama_host,
            ollama_port=environment.ollama_port,
        ),
    )


def _build_fastapi_settings() -> FastAPISettings:
    return FastAPISettings(
        title=FASTAPI_TITLE,
        version=FASTAPI_VERSION,
        description=FASTAPI_DESCRIPTION,
    )


def _build_jobs_settings(data: dict[str, Any]) -> JobsSettings:
    payload = ensure_mapping(data)
    polling_interval = coerce_float(payload.get("polling_interval"), 1.0)
    if polling_interval <= 0:
        polling_interval = 1.0
    return JobsSettings(polling_interval=polling_interval)


def _build_database_settings(payload: dict[str, Any]) -> DatabaseSettings:
    embedded = coerce_bool(
        payload.get("embedded_database", payload.get("embedded")), True
    )
    insert_batch_size = coerce_int(payload.get("insert_batch_size"), 1000, minimum=1)
    commit_interval = coerce_int(payload.get("insert_commit_interval"), 5, minimum=1)
    select_page_size = coerce_int(payload.get("select_page_size"), 2000, minimum=100)
    if embedded:
        return DatabaseSettings(
            embedded_database=True,
            engine=None,
            host=None,
            port=None,
            database_name=None,
            username=None,
            password=None,
            ssl=False,
            ssl_ca=None,
            connect_timeout=coerce_int(payload.get("connect_timeout"), 10, minimum=1),
            insert_batch_size=insert_batch_size,
            insert_commit_interval=commit_interval,
            select_page_size=select_page_size,
        )
    engine_value = coerce_str_or_none(payload.get("engine")) or "postgres"
    return DatabaseSettings(
        embedded_database=False,
        engine=engine_value.lower(),
        host=coerce_str_or_none(payload.get("host")),
        port=coerce_int(payload.get("port"), 5432, minimum=1, maximum=65535),
        database_name=coerce_str_or_none(payload.get("database_name")),
        username=coerce_str_or_none(payload.get("username")),
        password=coerce_str_or_none(payload.get("password")),
        ssl=coerce_bool(payload.get("ssl", False), False),
        ssl_ca=coerce_str_or_none(payload.get("ssl_ca")),
        connect_timeout=coerce_int(payload.get("connect_timeout"), 10, minimum=1),
        insert_batch_size=insert_batch_size,
        insert_commit_interval=commit_interval,
        select_page_size=select_page_size,
    )


def _build_drugs_matcher_settings(data: dict[str, Any]) -> DrugsMatcherSettings:
    suffixes_value = data.get(
        "catalog_excluded_term_suffixes",
        DEFAULT_DRUG_MATCH_CATALOG_EXCLUDED_TERM_SUFFIXES,
    )
    suffix_candidates = (
        list(suffixes_value)
        if isinstance(suffixes_value, (list, tuple, set))
        else [suffixes_value]
    )
    suffixes = tuple(
        text
        for text in (coerce_str(entry, "").upper() for entry in suffix_candidates)
        if text
    )
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
        catalog_excluded_term_suffixes=(
            suffixes or DEFAULT_DRUG_MATCH_CATALOG_EXCLUDED_TERM_SUFFIXES
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
            DEFAULT_DRUG_MATCH_SPELLING_SHORT_NAME_LENGTH,
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


def _build_rag_settings(
    data: dict[str, Any], defaults: LLMRuntimeDefaults
) -> RagSettings:
    rerank_top_n = coerce_positive_int(data.get("rerank_top_n"), 10)
    rerank_candidate_k = coerce_positive_int(data.get("rerank_candidate_k"), 100)
    if rerank_candidate_k < rerank_top_n:
        rerank_candidate_k = rerank_top_n
    return RagSettings(
        vector_collection_name=coerce_str(
            data.get("vector_collection_name"), "documents"
        ),
        chunk_size=coerce_positive_int(data.get("chunk_size"), 1024),
        chunk_overlap=coerce_positive_int(data.get("chunk_overlap"), 128),
        embedding_batch_size=coerce_positive_int(
            data.get("embedding_batch_size"),
            DEFAULT_EMBEDDING_BATCH_SIZE,
        ),
        use_reranking=coerce_bool(data.get("use_reranking"), True),
        rerank_candidate_k=rerank_candidate_k,
        rerank_top_n=rerank_top_n,
        embedding_backend=coerce_str(data.get("embedding_backend"), "ollama"),
        ollama_base_url=coerce_str(
            data.get("ollama_base_url"), defaults.ollama_host_default
        ),
        ollama_embedding_model=coerce_str(data.get("ollama_embedding_model"), ""),
        hf_embedding_model=coerce_str(data.get("hf_embedding_model"), ""),
        vector_index_metric=coerce_str(data.get("vector_index_metric"), "cosine"),
        vector_index_type=coerce_str(data.get("vector_index_type"), "IVF_FLAT"),
        reset_vector_collection=coerce_bool(data.get("reset_vector_collection"), True),
        cloud_provider=coerce_str(data.get("cloud_provider"), defaults.llm_provider),
        cloud_model=coerce_str(data.get("cloud_model"), defaults.cloud_model),
        cloud_embedding_model=coerce_str(data.get("cloud_embedding_model"), ""),
        use_cloud_embeddings=coerce_bool(data.get("use_cloud_embeddings"), False),
        vector_stream_batch_size=coerce_positive_int(
            data.get("vector_stream_batch_size"), 1024
        ),
        embedding_max_workers=coerce_positive_int(data.get("embedding_max_workers"), 4),
    )


def _build_external_data_settings(
    data: dict[str, Any], *, fallback_timeout: float
) -> ExternalDataSettings:
    unified_llm_timeout = max(
        coerce_float(
            data.get("llm_timeout"),
            coerce_float(data.get("default_llm_timeout"), fallback_timeout),
        ),
        1.0,
    )
    # Parser and clinical flows share one timeout budget by policy.
    parser_timeout = unified_llm_timeout
    disease_timeout = unified_llm_timeout
    clinical_timeout = unified_llm_timeout
    livertox_timeout = max(
        coerce_float(data.get("livertox_llm_timeout"), unified_llm_timeout), 1.0
    )
    brave_fast_max_results = coerce_positive_int(data.get("brave_fast_max_results"), 5)
    brave_thorough_max_results = coerce_positive_int(
        data.get("brave_thorough_max_results"), 10
    )
    if brave_thorough_max_results < brave_fast_max_results:
        brave_thorough_max_results = brave_fast_max_results
    return ExternalDataSettings(
        default_llm_timeout=unified_llm_timeout,
        parser_llm_timeout=parser_timeout,
        disease_llm_timeout=disease_timeout,
        clinical_llm_timeout=clinical_timeout,
        livertox_llm_timeout=livertox_timeout,
        ollama_server_start_timeout=max(
            coerce_float(data.get("ollama_server_start_timeout"), 15.0), 1.0
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
            data.get("rxnav_max_concurrency"), 16
        ),
        brave_request_timeout_s=max(
            coerce_float(data.get("brave_request_timeout_s"), 20.0), 1.0
        ),
        brave_search_cache_ttl_s=coerce_positive_int(
            data.get("brave_search_cache_ttl_s"), 21600
        ),
        brave_rate_limit_per_minute=coerce_positive_int(
            data.get("brave_rate_limit_per_minute"), 30
        ),
        brave_fast_max_results=brave_fast_max_results,
        brave_thorough_max_results=brave_thorough_max_results,
    )


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


def build_settings_payload_from_json(
    config: dict[str, Any], env: EnvironmentSnapshot
) -> dict[str, Any]:
    payload = ensure_mapping(config)
    llm_defaults = _default_llm_runtime_defaults(env)
    jobs_payload = ensure_mapping(payload.get("jobs"))
    database_payload = ensure_mapping(payload.get("database"))
    drugs_matcher_payload = ensure_mapping(payload.get("drugs_matcher"))
    rag_payload = ensure_mapping(payload.get("rag"))
    external_data_payload = ensure_mapping(payload.get("external_data"))
    ingestion_payload = ensure_mapping(payload.get("ingestion"))
    session_pipeline_payload = ensure_mapping(payload.get("session_pipeline"))
    return {
        "fastapi": _build_fastapi_settings().model_dump(),
        "jobs": _build_jobs_settings(jobs_payload).model_dump(),
        "database": _build_database_settings(database_payload).model_dump(),
        "drugs_matcher": _build_drugs_matcher_settings(
            drugs_matcher_payload
        ).model_dump(),
        "rag": _build_rag_settings(rag_payload, llm_defaults).model_dump(),
        "external_data": _build_external_data_settings(
            external_data_payload,
            fallback_timeout=30.0,
        ).model_dump(),
        "ingestion": _build_ingestion_settings(ingestion_payload).model_dump(),
        "session_pipeline": _build_session_pipeline_settings(
            session_pipeline_payload
        ).model_dump(),
        "llm_defaults": llm_defaults.model_dump(),
    }

