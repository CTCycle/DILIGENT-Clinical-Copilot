from __future__ import annotations

from typing import Any

from DILIGENT.server.common.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    CONFIGURATIONS_FILE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    PARSING_MODEL_CHOICES,
)
from DILIGENT.server.common.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_positive_int,
    coerce_str,
    coerce_str_or_none,
)
from DILIGENT.server.configurations.env_loader import load_environment
from DILIGENT.server.configurations.json_loader import ensure_mapping, load_configuration_data
from DILIGENT.server.configurations.runtime_state import LLMRuntimeConfig
from DILIGENT.server.domain.settings.configuration import (
    DatabaseSettings,
    DrugsMatcherSettings,
    ExternalDataSettings,
    FastAPISettings,
    IngestionSettings,
    JobsSettings,
    LLMRuntimeDefaults,
    RagSettings,
    ServerSettings,
)
from DILIGENT.server.domain.settings.environment import EnvironmentSettings

OLLAMA_DEFAULT_HOST = "localhost"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_SCHEME = "http"
FASTAPI_TITLE = "DILI Backend"
FASTAPI_VERSION = "1.0.0"
FASTAPI_DESCRIPTION = "FastAPI backend"


def resolve_ollama_base_url(
    environment: EnvironmentSettings,
    fallback: str = f"{OLLAMA_DEFAULT_SCHEME}://{OLLAMA_DEFAULT_HOST}:{OLLAMA_DEFAULT_PORT}",
) -> str:
    if environment.ollama_url:
        return environment.ollama_url.rstrip("/")
    host_value = coerce_str_or_none(environment.ollama_host)
    port_value = environment.ollama_port
    if host_value:
        normalized_host = host_value.strip().rstrip("/")
        if "://" in normalized_host:
            scheme, host_port = normalized_host.split("://", maxsplit=1)
            if ":" in host_port:
                host_only, parsed_port = host_port.split(":", maxsplit=1)
                resolved_port = (
                    port_value if port_value is not None else coerce_int(parsed_port, OLLAMA_DEFAULT_PORT, minimum=1, maximum=65535)
                )
                return f"{scheme}://{host_only}:{resolved_port}"
            resolved_port = port_value if port_value is not None else OLLAMA_DEFAULT_PORT
            return f"{scheme}://{host_port}:{resolved_port}"
        resolved_port = port_value if port_value is not None else OLLAMA_DEFAULT_PORT
        return f"{OLLAMA_DEFAULT_SCHEME}://{normalized_host}:{resolved_port}"
    if port_value is not None:
        return f"{OLLAMA_DEFAULT_SCHEME}://{OLLAMA_DEFAULT_HOST}:{port_value}"
    return fallback.rstrip("/")


def build_fastapi_settings(data: dict[str, Any]) -> FastAPISettings:
    payload = ensure_mapping(data)
    return FastAPISettings(
        title=coerce_str(payload.get("title"), FASTAPI_TITLE),
        version=coerce_str(payload.get("version"), FASTAPI_VERSION),
        description=coerce_str(payload.get("description"), FASTAPI_DESCRIPTION),
    )


def build_jobs_settings(data: dict[str, Any]) -> JobsSettings:
    payload = ensure_mapping(data)
    polling_interval = coerce_float(payload.get("polling_interval"), 1.0)
    if polling_interval <= 0:
        polling_interval = 1.0
    return JobsSettings(polling_interval=polling_interval)


def build_database_settings(payload: dict[str, Any], environment: EnvironmentSettings) -> DatabaseSettings:
    embedded = coerce_bool(environment.db_embedded, True)
    insert_batch_size = coerce_int(
        environment.db_insert_batch_size if environment.db_insert_batch_size is not None else payload.get("insert_batch_size"),
        1000,
        minimum=1,
    )
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
            connect_timeout=10,
            insert_batch_size=insert_batch_size,
            insert_commit_interval=commit_interval,
            select_page_size=select_page_size,
        )
    engine_value = coerce_str_or_none(environment.db_engine) or coerce_str_or_none(payload.get("engine")) or "postgres"
    return DatabaseSettings(
        embedded_database=False,
        engine=engine_value.lower(),
        host=coerce_str_or_none(environment.db_host) or coerce_str_or_none(payload.get("host")),
        port=coerce_int(environment.db_port if environment.db_port is not None else payload.get("port"), 5432, minimum=1, maximum=65535),
        database_name=coerce_str_or_none(environment.db_name) or coerce_str_or_none(payload.get("database_name")),
        username=coerce_str_or_none(environment.db_user) or coerce_str_or_none(payload.get("username")),
        password=coerce_str_or_none(environment.db_password),
        ssl=coerce_bool(environment.db_ssl, coerce_bool(payload.get("ssl", False), False)),
        ssl_ca=coerce_str_or_none(environment.db_ssl_ca) or coerce_str_or_none(payload.get("ssl_ca")),
        connect_timeout=coerce_int(environment.db_connect_timeout, 10, minimum=1),
        insert_batch_size=insert_batch_size,
        insert_commit_interval=commit_interval,
        select_page_size=select_page_size,
    )


def build_drugs_matcher_settings(data: dict[str, Any]) -> DrugsMatcherSettings:
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


def build_rag_settings(
    data: dict[str, Any],
    *,
    default_provider: str,
    default_cloud_model: str,
    default_ollama_host: str,
) -> RagSettings:
    rerank_top_n = coerce_positive_int(data.get("rerank_top_n"), 10)
    rerank_candidate_k = coerce_positive_int(data.get("rerank_candidate_k"), 100)
    if rerank_candidate_k < rerank_top_n:
        rerank_candidate_k = rerank_top_n
    return RagSettings(
        vector_collection_name=coerce_str(data.get("vector_collection_name"), "documents"),
        chunk_size=coerce_positive_int(data.get("chunk_size"), 1024),
        chunk_overlap=coerce_positive_int(data.get("chunk_overlap"), 128),
        embedding_batch_size=coerce_positive_int(data.get("embedding_batch_size"), DEFAULT_EMBEDDING_BATCH_SIZE),
        use_reranking=coerce_bool(data.get("use_reranking"), True),
        rerank_candidate_k=rerank_candidate_k,
        rerank_top_n=rerank_top_n,
        embedding_backend=coerce_str(data.get("embedding_backend"), "ollama"),
        ollama_base_url=coerce_str(data.get("ollama_base_url"), default_ollama_host),
        ollama_embedding_model=coerce_str(data.get("ollama_embedding_model"), ""),
        hf_embedding_model=coerce_str(data.get("hf_embedding_model"), ""),
        vector_index_metric=coerce_str(data.get("vector_index_metric"), "cosine"),
        vector_index_type=coerce_str(data.get("vector_index_type"), "IVF_FLAT"),
        reset_vector_collection=coerce_bool(data.get("reset_vector_collection"), True),
        cloud_provider=coerce_str(data.get("cloud_provider"), default_provider),
        cloud_model=coerce_str(data.get("cloud_model"), default_cloud_model),
        cloud_embedding_model=coerce_str(data.get("cloud_embedding_model"), ""),
        use_cloud_embeddings=coerce_bool(data.get("use_cloud_embeddings"), False),
        vector_stream_batch_size=coerce_positive_int(data.get("vector_stream_batch_size"), 1024),
        embedding_max_workers=coerce_positive_int(data.get("embedding_max_workers"), 4),
    )


def build_external_data_settings(data: dict[str, Any], *, fallback_timeout: float) -> ExternalDataSettings:
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
        tavily_request_timeout_s=max(coerce_float(data.get("tavily_request_timeout_s"), 20.0), 1.0),
        tavily_search_cache_ttl_s=coerce_positive_int(data.get("tavily_search_cache_ttl_s"), 21600),
        tavily_extract_cache_ttl_s=coerce_positive_int(data.get("tavily_extract_cache_ttl_s"), 259200),
        tavily_rate_limit_per_minute=coerce_positive_int(data.get("tavily_rate_limit_per_minute"), 30),
        tavily_fast_max_results=tavily_fast_max_results,
        tavily_thorough_max_results=tavily_thorough_max_results,
        tavily_extract_top_urls=min(coerce_positive_int(data.get("tavily_extract_top_urls"), 3), 5),
    )


def build_ingestion_settings(data: dict[str, Any]) -> IngestionSettings:
    min_length = coerce_positive_int(data.get("drug_name_min_length"), 3)
    max_length = coerce_positive_int(data.get("drug_name_max_length"), 200)
    if max_length < min_length:
        max_length = min_length
    return IngestionSettings(
        drug_name_min_length=min_length,
        drug_name_max_length=max_length,
        drug_name_max_tokens=coerce_positive_int(data.get("drug_name_max_tokens"), 8),
    )


def build_llm_runtime_defaults(data: dict[str, Any], environment: EnvironmentSettings) -> LLMRuntimeDefaults:
    parsing_default = PARSING_MODEL_CHOICES[0] if PARSING_MODEL_CHOICES else ""
    clinical_default = CLINICAL_MODEL_CHOICES[0] if CLINICAL_MODEL_CHOICES else ""
    provider_default = coerce_str(data.get("llm_provider"), "openai").lower()
    provider_models = CLOUD_MODEL_CHOICES.get(provider_default, [])
    cloud_default = provider_models[0] if provider_models else ""
    return LLMRuntimeDefaults(
        parsing_model=coerce_str(data.get("parsing_model"), parsing_default),
        clinical_model=coerce_str(data.get("clinical_model"), clinical_default),
        llm_provider=provider_default,
        cloud_model=coerce_str(data.get("cloud_model"), cloud_default),
        use_cloud_services=coerce_bool(data.get("use_cloud_services"), False),
        ollama_temperature=coerce_float(data.get("ollama_temperature"), 0.7),
        cloud_temperature=coerce_float(data.get("cloud_temperature"), 0.7),
        ollama_reasoning=coerce_bool(data.get("ollama_reasoning"), False),
        ollama_host_default=resolve_ollama_base_url(environment),
    )


def build_server_settings(data: dict[str, Any], environment: EnvironmentSettings) -> ServerSettings:
    payload = ensure_mapping(data)
    fastapi_payload = ensure_mapping(payload.get("fastapi"))
    jobs_payload = ensure_mapping(payload.get("jobs"))
    database_payload = ensure_mapping(payload.get("database"))
    drugs_matcher_payload = ensure_mapping(payload.get("drugs_matcher"))
    rag_payload = ensure_mapping(payload.get("rag"))
    external_data_payload = ensure_mapping(payload.get("external_data"))
    ingestion_payload = ensure_mapping(payload.get("ingestion"))
    llm_defaults_payload = ensure_mapping(payload.get("llm_defaults") or payload.get("llm_runtime_defaults"))
    default_ollama_host = resolve_ollama_base_url(environment)
    llm_defaults = build_llm_runtime_defaults(llm_defaults_payload, environment)
    rag_settings = build_rag_settings(
        rag_payload,
        default_provider=llm_defaults.llm_provider,
        default_cloud_model=llm_defaults.cloud_model,
        default_ollama_host=default_ollama_host,
    )
    return ServerSettings(
        fastapi=build_fastapi_settings(fastapi_payload),
        jobs=build_jobs_settings(jobs_payload),
        database=build_database_settings(database_payload, environment),
        drugs_matcher=build_drugs_matcher_settings(drugs_matcher_payload),
        rag=rag_settings,
        external_data=build_external_data_settings(external_data_payload, fallback_timeout=30.0),
        ingestion=build_ingestion_settings(ingestion_payload),
        llm_defaults=llm_defaults,
    )


def get_server_settings(config_path: str | None = None) -> tuple[EnvironmentSettings, ServerSettings]:
    environment = load_environment()
    payload = load_configuration_data(config_path or CONFIGURATIONS_FILE)
    settings = build_server_settings(payload, environment)
    LLMRuntimeConfig.configure(settings.llm_defaults)
    return environment, settings


environment_settings, server_settings = get_server_settings()
