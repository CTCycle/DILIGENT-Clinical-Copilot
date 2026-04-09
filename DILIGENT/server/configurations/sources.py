from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic_settings import PydanticBaseSettingsSource

from DILIGENT.server.common import constants
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_positive_int,
    coerce_str,
    coerce_str_or_none,
)
from DILIGENT.server.configurations.base import ensure_mapping, load_configuration_data
from DILIGENT.server.configurations.ollama import resolve_ollama_base_url
from DILIGENT.server.domain.settings.configuration import (
    DatabaseSettings,
    DrugsMatcherSettings,
    ExternalDataSettings,
    FastAPISettings,
    IngestionSettings,
    JobsSettings,
    LLMRuntimeDefaults,
    RagSettings,
)

FASTAPI_TITLE = "DILI Backend"
FASTAPI_VERSION = "1.0.0"
FASTAPI_DESCRIPTION = "FastAPI backend"

UI_OWNED_ENV_KEYS = {
    "PARSING_MODEL",
    "TEXT_EXTRACTION_MODEL",
    "CLINICAL_MODEL",
    "LLM_PROVIDER",
    "CLOUD_MODEL",
    "USE_CLOUD_SERVICES",
    "OLLAMA_TEMPERATURE",
    "CLOUD_TEMPERATURE",
    "OLLAMA_REASONING",
}

UI_OWNED_JSON_KEYS = {
    "parsing_model",
    "clinical_model",
    "llm_provider",
    "cloud_model",
    "use_cloud_services",
    "ollama_temperature",
    "cloud_temperature",
    "ollama_reasoning",
}

CURATED_ENV_KEYS: dict[str, tuple[str, str, str]] = {
    "DB_EMBEDDED": ("database", "embedded_database", "bool"),
    "DB_ENGINE": ("database", "engine", "str"),
    "DB_HOST": ("database", "host", "str"),
    "DB_PORT": ("database", "port", "int"),
    "DB_NAME": ("database", "database_name", "str"),
    "DB_USER": ("database", "username", "str"),
    "DB_PASSWORD": ("database", "password", "str"),
    "DB_SSL": ("database", "ssl", "bool"),
    "DB_SSL_CA": ("database", "ssl_ca", "str"),
    "DB_CONNECT_TIMEOUT": ("database", "connect_timeout", "int"),
    "DB_INSERT_BATCH_SIZE": ("database", "insert_batch_size", "int"),
    "DB_INSERT_COMMIT_INTERVAL": ("database", "insert_commit_interval", "int"),
    "DB_SELECT_PAGE_SIZE": ("database", "select_page_size", "int"),
    "JOBS_POLLING_INTERVAL": ("jobs", "polling_interval", "float"),
    "RAG_EMBEDDING_BATCH_SIZE": ("rag", "embedding_batch_size", "int"),
    "RAG_VECTOR_STREAM_BATCH_SIZE": ("rag", "vector_stream_batch_size", "int"),
    "RAG_EMBEDDING_MAX_WORKERS": ("rag", "embedding_max_workers", "int"),
    "RAG_EMBEDDING_BACKEND": ("rag", "embedding_backend", "str"),
    "RAG_OLLAMA_BASE_URL": ("rag", "ollama_base_url", "str"),
    "RAG_OLLAMA_EMBEDDING_MODEL": ("rag", "ollama_embedding_model", "str"),
    "RAG_HF_EMBEDDING_MODEL": ("rag", "hf_embedding_model", "str"),
    "RAG_USE_CLOUD_EMBEDDINGS": ("rag", "use_cloud_embeddings", "bool"),
    "RAG_CLOUD_PROVIDER": ("rag", "cloud_provider", "str"),
    "RAG_CLOUD_EMBEDDING_MODEL": ("rag", "cloud_embedding_model", "str"),
    "EXTERNAL_DEFAULT_LLM_TIMEOUT": ("external_data", "default_llm_timeout", "float"),
    "EXTERNAL_PARSER_LLM_TIMEOUT": ("external_data", "parser_llm_timeout", "float"),
    "EXTERNAL_DISEASE_LLM_TIMEOUT": ("external_data", "disease_llm_timeout", "float"),
    "EXTERNAL_CLINICAL_LLM_TIMEOUT": ("external_data", "clinical_llm_timeout", "float"),
    "EXTERNAL_LIVERTOX_LLM_TIMEOUT": ("external_data", "livertox_llm_timeout", "float"),
    "EXTERNAL_OLLAMA_SERVER_START_TIMEOUT": (
        "external_data",
        "ollama_server_start_timeout",
        "float",
    ),
    "EXTERNAL_RXNAV_REQUEST_TIMEOUT": ("external_data", "rxnav_request_timeout", "float"),
    "EXTERNAL_RXNAV_MAX_CONCURRENCY": ("external_data", "rxnav_max_concurrency", "int"),
    "EXTERNAL_DAILYMED_REQUEST_TIMEOUT": ("external_data", "dailymed_request_timeout", "float"),
    "EXTERNAL_DAILYMED_MAX_CONCURRENCY": ("external_data", "dailymed_max_concurrency", "int"),
}

JSON_ENV_OVERRIDE_PREFIXES = (
    "DB_",
    "DATABASE_",
    "JOBS_",
    "RAG_",
    "EXTERNAL_",
    "INGESTION_",
    "DRUGS_MATCHER_",
)

DYNAMIC_ENV_KEYS = {"OLLAMA_AVAILABLE_VRAM_BYTES"}


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

def _parse_with_kind(value: str, kind: str) -> Any:
    if kind == "bool":
        return coerce_bool(value, False)
    if kind == "int":
        return coerce_int(value, 0)
    if kind == "float":
        return coerce_float(value, 0.0)
    return coerce_str(value, "")


def _set_nested(payload: dict[str, Any], section: str, key: str, value: Any) -> None:
    section_payload = payload.setdefault(section, {})
    if not isinstance(section_payload, dict):
        section_payload = {}
        payload[section] = section_payload
    section_payload[key] = value


def _warn_ignored_env_keys(source_name: str, source: dict[str, str]) -> None:
    ignored: list[str] = []
    for key in source:
        if key in CURATED_ENV_KEYS or key in UI_OWNED_ENV_KEYS or key in DYNAMIC_ENV_KEYS:
            continue
        if key in {"FASTAPI_HOST", "FASTAPI_PORT", "UI_HOST", "UI_PORT", "RELOAD"}:
            continue
        if any(key.startswith(prefix) for prefix in JSON_ENV_OVERRIDE_PREFIXES):
            ignored.append(key)
    if ignored:
        logger.warning(
            "Ignoring non-curated environment overrides from %s: %s",
            source_name,
            ", ".join(sorted(set(ignored))),
        )


def _raise_if_ui_env_keys_present(source_name: str, source: dict[str, str]) -> None:
    overlap = sorted(key for key in UI_OWNED_ENV_KEYS if source.get(key))
    if overlap:
        joined = ", ".join(overlap)
        raise RuntimeError(
            "UI-owned runtime keys must not be provided via environment "
            f"({source_name}): {joined}"
        )


def _extract_curated_source_payload(source_name: str, source: dict[str, str]) -> dict[str, Any]:
    _raise_if_ui_env_keys_present(source_name, source)
    _warn_ignored_env_keys(source_name, source)
    payload: dict[str, Any] = {}
    for env_key, (section, target_key, kind) in CURATED_ENV_KEYS.items():
        raw_value = source.get(env_key)
        if raw_value is None:
            continue
        parsed = _parse_with_kind(raw_value, kind)
        _set_nested(payload, section, target_key, parsed)
    return payload


def _read_dotenv() -> dict[str, str]:
    dotenv_path = Path(constants.ENV_FILE_PATH)
    if not dotenv_path.exists():
        return {}
    values = dotenv_values(dotenv_path)
    payload: dict[str, str] = {}
    for key, value in values.items():
        if not key or value is None:
            continue
        payload[str(key)] = str(value)
    return payload


def _resolve_env_snapshot() -> EnvironmentSnapshot:
    dotenv_values_map = _read_dotenv()
    env_ollama_url = os.getenv("OLLAMA_URL")
    env_ollama_host = os.getenv("OLLAMA_HOST")
    env_ollama_port = os.getenv("OLLAMA_PORT")
    dot_ollama_url = dotenv_values_map.get("OLLAMA_URL")
    dot_ollama_host = dotenv_values_map.get("OLLAMA_HOST")
    dot_ollama_port = dotenv_values_map.get("OLLAMA_PORT")
    ollama_url = env_ollama_url if env_ollama_url is not None else dot_ollama_url
    ollama_host = env_ollama_host if env_ollama_host is not None else dot_ollama_host
    raw_port = env_ollama_port if env_ollama_port is not None else dot_ollama_port
    ollama_port = coerce_int(raw_port, None, minimum=1, maximum=65535) if raw_port else None
    return EnvironmentSnapshot(
        ollama_url=coerce_str_or_none(ollama_url),
        ollama_host=coerce_str_or_none(ollama_host),
        ollama_port=ollama_port,
    )


def _default_llm_runtime_defaults(environment: EnvironmentSnapshot) -> LLMRuntimeDefaults:
    parsing_default = constants.PARSING_MODEL_CHOICES[0] if constants.PARSING_MODEL_CHOICES else ""
    clinical_default = constants.CLINICAL_MODEL_CHOICES[0] if constants.CLINICAL_MODEL_CHOICES else ""
    provider_default = "openai"
    provider_models = constants.CLOUD_MODEL_CHOICES.get(provider_default, [])
    cloud_default = provider_models[0] if provider_models else ""
    return LLMRuntimeDefaults(
        parsing_model=parsing_default,
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


def _build_fastapi_settings(data: dict[str, Any]) -> FastAPISettings:
    payload = ensure_mapping(data)
    return FastAPISettings(
        title=coerce_str(payload.get("title"), FASTAPI_TITLE),
        version=coerce_str(payload.get("version"), FASTAPI_VERSION),
        description=coerce_str(payload.get("description"), FASTAPI_DESCRIPTION),
    )


def _build_jobs_settings(data: dict[str, Any]) -> JobsSettings:
    payload = ensure_mapping(data)
    polling_interval = coerce_float(payload.get("polling_interval"), 1.0)
    if polling_interval <= 0:
        polling_interval = 1.0
    return JobsSettings(polling_interval=polling_interval)


def _build_database_settings(payload: dict[str, Any]) -> DatabaseSettings:
    embedded = coerce_bool(payload.get("embedded_database", payload.get("embedded")), True)
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
    suffixes_value = data.get("catalog_excluded_term_suffixes", ["PCK"])
    candidates = list(suffixes_value) if isinstance(suffixes_value, (list, tuple, set)) else [suffixes_value]
    suffixes: list[str] = []
    for entry in candidates:
        text = coerce_str(entry, "").upper()
        if text:
            suffixes.append(text)
    suffix_tuple = tuple(suffixes) if suffixes else ("PCK",)
    return DrugsMatcherSettings(
        direct_confidence=coerce_float(data.get("direct_confidence"), 1.0),
        master_confidence=coerce_float(data.get("master_confidence"), 0.92),
        synonym_confidence=coerce_float(data.get("synonym_confidence"), 0.90),
        partial_confidence=coerce_float(data.get("partial_confidence"), 0.86),
        fuzzy_confidence=coerce_float(data.get("fuzzy_confidence"), 0.84),
        fuzzy_threshold=coerce_float(data.get("fuzzy_threshold"), 0.85),
        token_max_frequency=coerce_positive_int(data.get("token_max_frequency"), 3),
        token_min_length=coerce_positive_int(data.get("token_min_length"), 4),
        normalization_cache_limit=coerce_positive_int(data.get("normalization_cache_limit"), 10000),
        variant_cache_limit=coerce_positive_int(data.get("variant_cache_limit"), 10000),
        min_confidence=coerce_float(data.get("min_confidence"), 0.40),
        catalog_excluded_term_suffixes=suffix_tuple,
        catalog_token_ratio_threshold=coerce_float(data.get("catalog_token_ratio_threshold"), 0.93),
        catalog_overall_ratio_threshold=coerce_float(data.get("catalog_overall_ratio_threshold"), 0.93),
        fuzzy_early_exit_ratio=coerce_float(data.get("fuzzy_early_exit_ratio"), 0.95),
        match_cache_limit=coerce_positive_int(data.get("match_cache_limit"), 5000),
        alias_cache_limit=coerce_positive_int(data.get("alias_cache_limit"), 2000),
        catalog_index_limit=coerce_positive_int(data.get("catalog_index_limit"), 75000),
        catalog_candidate_limit=coerce_positive_int(data.get("catalog_candidate_limit"), 750),
    )


def _build_rag_settings(data: dict[str, Any], defaults: LLMRuntimeDefaults) -> RagSettings:
    rerank_top_n = coerce_positive_int(data.get("rerank_top_n"), 10)
    rerank_candidate_k = coerce_positive_int(data.get("rerank_candidate_k"), 100)
    if rerank_candidate_k < rerank_top_n:
        rerank_candidate_k = rerank_top_n
    return RagSettings(
        vector_collection_name=coerce_str(data.get("vector_collection_name"), "documents"),
        chunk_size=coerce_positive_int(data.get("chunk_size"), 1024),
        chunk_overlap=coerce_positive_int(data.get("chunk_overlap"), 128),
        embedding_batch_size=coerce_positive_int(
            data.get("embedding_batch_size"),
            constants.DEFAULT_EMBEDDING_BATCH_SIZE,
        ),
        use_reranking=coerce_bool(data.get("use_reranking"), True),
        rerank_candidate_k=rerank_candidate_k,
        rerank_top_n=rerank_top_n,
        embedding_backend=coerce_str(data.get("embedding_backend"), "ollama"),
        ollama_base_url=coerce_str(data.get("ollama_base_url"), defaults.ollama_host_default),
        ollama_embedding_model=coerce_str(data.get("ollama_embedding_model"), ""),
        hf_embedding_model=coerce_str(data.get("hf_embedding_model"), ""),
        vector_index_metric=coerce_str(data.get("vector_index_metric"), "cosine"),
        vector_index_type=coerce_str(data.get("vector_index_type"), "IVF_FLAT"),
        reset_vector_collection=coerce_bool(data.get("reset_vector_collection"), True),
        cloud_provider=coerce_str(data.get("cloud_provider"), defaults.llm_provider),
        cloud_model=coerce_str(data.get("cloud_model"), defaults.cloud_model),
        cloud_embedding_model=coerce_str(data.get("cloud_embedding_model"), ""),
        use_cloud_embeddings=coerce_bool(data.get("use_cloud_embeddings"), False),
        vector_stream_batch_size=coerce_positive_int(data.get("vector_stream_batch_size"), 1024),
        embedding_max_workers=coerce_positive_int(data.get("embedding_max_workers"), 4),
    )


def _build_external_data_settings(data: dict[str, Any], *, fallback_timeout: float) -> ExternalDataSettings:
    default_llm_timeout = max(coerce_float(data.get("default_llm_timeout"), fallback_timeout), 1.0)
    parser_timeout = max(coerce_float(data.get("parser_llm_timeout"), default_llm_timeout), 1.0)
    disease_timeout = max(coerce_float(data.get("disease_llm_timeout"), parser_timeout), 1.0)
    clinical_timeout = max(coerce_float(data.get("clinical_llm_timeout"), default_llm_timeout), 1.0)
    livertox_timeout = max(coerce_float(data.get("livertox_llm_timeout"), default_llm_timeout), 1.0)
    tavily_fast_max_results = coerce_positive_int(data.get("tavily_fast_max_results"), 5)
    tavily_thorough_max_results = coerce_positive_int(data.get("tavily_thorough_max_results"), 10)
    if tavily_thorough_max_results < tavily_fast_max_results:
        tavily_thorough_max_results = tavily_fast_max_results
    return ExternalDataSettings(
        default_llm_timeout=default_llm_timeout,
        parser_llm_timeout=parser_timeout,
        disease_llm_timeout=disease_timeout,
        clinical_llm_timeout=clinical_timeout,
        livertox_llm_timeout=livertox_timeout,
        ollama_server_start_timeout=max(coerce_float(data.get("ollama_server_start_timeout"), 15.0), 1.0),
        livertox_archive=coerce_str(data.get("livertox_archive"), "livertox_NBK547852.tar.gz"),
        livertox_yield_interval=coerce_positive_int(data.get("livertox_yield_interval"), 25),
        livertox_skip_deterministic_ratio=coerce_float(data.get("livertox_skip_deterministic_ratio"), 0.80),
        livertox_monograph_max_workers=coerce_positive_int(data.get("livertox_monograph_max_workers"), 4),
        max_excerpt_length=coerce_positive_int(data.get("max_excerpt_length"), 8000),
        rxnav_request_timeout=coerce_float(data.get("rxnav_request_timeout"), 12.0),
        rxnav_max_concurrency=coerce_positive_int(data.get("rxnav_max_concurrency"), 16),
        dili_priors_request_timeout=max(coerce_float(data.get("dili_priors_request_timeout"), 20.0), 1.0),
        dailymed_request_timeout=max(coerce_float(data.get("dailymed_request_timeout"), 20.0), 1.0),
        dailymed_max_concurrency=coerce_positive_int(data.get("dailymed_max_concurrency"), 8),
        dailymed_section_max_length=coerce_positive_int(data.get("dailymed_section_max_length"), 4000),
        dailymed_max_sections_per_drug=coerce_positive_int(data.get("dailymed_max_sections_per_drug"), 5),
        tavily_request_timeout_s=max(coerce_float(data.get("tavily_request_timeout_s"), 20.0), 1.0),
        tavily_search_cache_ttl_s=coerce_positive_int(data.get("tavily_search_cache_ttl_s"), 21600),
        tavily_extract_cache_ttl_s=coerce_positive_int(data.get("tavily_extract_cache_ttl_s"), 259200),
        tavily_rate_limit_per_minute=coerce_positive_int(data.get("tavily_rate_limit_per_minute"), 30),
        tavily_fast_max_results=tavily_fast_max_results,
        tavily_thorough_max_results=tavily_thorough_max_results,
        tavily_extract_top_urls=min(coerce_positive_int(data.get("tavily_extract_top_urls"), 3), 5),
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


def build_settings_payload_from_json(config: dict[str, Any], env: EnvironmentSnapshot) -> dict[str, Any]:
    payload = ensure_mapping(config)
    llm_from_json = payload.get("llm_defaults") or payload.get("llm_runtime_defaults")
    if isinstance(llm_from_json, dict):
        overlap = sorted(key for key in UI_OWNED_JSON_KEYS if key in llm_from_json)
        if overlap:
            joined = ", ".join(overlap)
            raise RuntimeError(
                "UI-owned runtime keys must not be provided via settings/configurations.json "
                f"(llm_defaults): {joined}"
            )
    llm_defaults = _default_llm_runtime_defaults(env)
    fastapi_payload = ensure_mapping(payload.get("fastapi"))
    jobs_payload = ensure_mapping(payload.get("jobs"))
    database_payload = ensure_mapping(payload.get("database"))
    drugs_matcher_payload = ensure_mapping(payload.get("drugs_matcher"))
    rag_payload = ensure_mapping(payload.get("rag"))
    external_data_payload = ensure_mapping(payload.get("external_data"))
    ingestion_payload = ensure_mapping(payload.get("ingestion"))
    return {
        "fastapi": _build_fastapi_settings(fastapi_payload).model_dump(),
        "jobs": _build_jobs_settings(jobs_payload).model_dump(),
        "database": _build_database_settings(database_payload).model_dump(),
        "drugs_matcher": _build_drugs_matcher_settings(drugs_matcher_payload).model_dump(),
        "rag": _build_rag_settings(rag_payload, llm_defaults).model_dump(),
        "external_data": _build_external_data_settings(
            external_data_payload,
            fallback_timeout=30.0,
        ).model_dump(),
        "ingestion": _build_ingestion_settings(ingestion_payload).model_dump(),
        "llm_defaults": llm_defaults.model_dump(),
    }


class CuratedEnvironmentSource(PydanticBaseSettingsSource):
    def __call__(self) -> dict[str, Any]:
        source: dict[str, str] = {}
        for key, value in os.environ.items():
            source[str(key)] = str(value)
        return _extract_curated_source_payload("OS environment", source)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError


class CuratedDotenvSource(PydanticBaseSettingsSource):
    def __call__(self) -> dict[str, Any]:
        source = _read_dotenv()
        return _extract_curated_source_payload(".env", source)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError


class JsonConfigurationSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[Any]) -> None:
        super().__init__(settings_cls)
        raw_path = getattr(settings_cls, "_configuration_file", None) or constants.CONFIGURATIONS_FILE
        self.configuration_file = str(raw_path)

    def __call__(self) -> dict[str, Any]:
        env_snapshot = _resolve_env_snapshot()
        payload = load_configuration_data(self.configuration_file)
        return build_settings_payload_from_json(payload, env_snapshot)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError
