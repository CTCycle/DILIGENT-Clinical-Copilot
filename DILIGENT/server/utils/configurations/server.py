from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from DILIGENT.server.utils.configurations import ensure_mapping, load_configuration_data
from DILIGENT.server.utils.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES, 
    DEFAULT_EMBEDDING_BATCH_SIZE,
    PARSING_MODEL_CHOICES,
    SERVER_CONFIGURATION_FILE,
)

from DILIGENT.server.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_positive_int,
    coerce_str,
    coerce_str_or_none,
    coerce_string_tuple
)


# [LLM RUNTIME CONFIGURATION]
###############################################################################
class LLMRuntimeConfig:
    defaults: LLMRuntimeDefaults | None = None
    parsing_model: str = ""
    clinical_model: str = ""
    llm_provider: str = ""
    cloud_model: str = ""
    use_cloud_services: bool = False
    ollama_temperature: float = 0.0
    ollama_reasoning: bool = False
    revision: int = 0

    # -------------------------------------------------------------------------
    @classmethod
    def configure(cls, defaults: LLMRuntimeDefaults) -> None:
        cls.defaults = defaults
        cls.reset_defaults()

    # -------------------------------------------------------------------------
    @classmethod
    def _get_defaults(cls) -> LLMRuntimeDefaults:
        if cls.defaults is None:
            raise RuntimeError("Client runtime defaults are not configured.")
        return cls.defaults

    # -------------------------------------------------------------------------
    @classmethod
    def touch_revision(cls) -> None:
        cls.revision += 1

    # -------------------------------------------------------------------------
    @classmethod
    def set_parsing_model(cls, model: str) -> str:
        value = model.strip()
        if value and value != cls.parsing_model:
            cls.parsing_model = value
            cls.touch_revision()
        return cls.parsing_model

    # -------------------------------------------------------------------------
    @classmethod
    def set_clinical_model(cls, model: str) -> str:
        value = model.strip()
        if value and value != cls.clinical_model:
            cls.clinical_model = value
            cls.touch_revision()
        return cls.clinical_model

    # -------------------------------------------------------------------------
    @classmethod
    def set_llm_provider(cls, provider: str) -> str:
        defaults = cls._get_defaults()
        value = provider.strip().lower()
        if not value:
            return cls.llm_provider
        if value not in CLOUD_MODEL_CHOICES:
            value = defaults.llm_provider
        if cls.llm_provider != value:
            cls.llm_provider = value
            models = CLOUD_MODEL_CHOICES.get(cls.llm_provider, [])
            if cls.cloud_model not in models:
                cls.cloud_model = models[0] if models else ""
            cls.touch_revision()
        return cls.llm_provider

    # -------------------------------------------------------------------------
    @classmethod
    def set_cloud_model(cls, model: str) -> str:
        value = model.strip()
        if not value:
            if cls.cloud_model:
                cls.cloud_model = ""
                cls.touch_revision()
            return cls.cloud_model
        models = CLOUD_MODEL_CHOICES.get(cls.llm_provider, [])
        if value not in models:
            fallback = models[0] if models else ""
            if fallback and fallback != cls.cloud_model:
                cls.cloud_model = fallback
                cls.touch_revision()
            return cls.cloud_model
        if cls.cloud_model != value:
            cls.cloud_model = value
            cls.touch_revision()
        return cls.cloud_model

    # -------------------------------------------------------------------------
    @classmethod
    def set_use_cloud_services(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        if cls.use_cloud_services != normalized:
            cls.use_cloud_services = normalized
            cls.touch_revision()
        return cls.use_cloud_services

    # -------------------------------------------------------------------------
    @classmethod
    def set_ollama_temperature(cls, value: float | None) -> float:
        defaults = cls._get_defaults()
        try:
            parsed = float(value) if value is not None else cls.ollama_temperature
        except (TypeError, ValueError):
            parsed = defaults.ollama_temperature
        parsed = max(0.0, min(2.0, parsed))
        rounded = round(parsed, 2)
        if cls.ollama_temperature != rounded:
            cls.ollama_temperature = rounded
            cls.touch_revision()
        return cls.ollama_temperature

    # -------------------------------------------------------------------------
    @classmethod
    def set_ollama_reasoning(cls, enabled: bool) -> bool:
        normalized = bool(enabled)
        if cls.ollama_reasoning != normalized:
            cls.ollama_reasoning = normalized
            cls.touch_revision()
        return cls.ollama_reasoning

    # -------------------------------------------------------------------------
    @classmethod
    def get_parsing_model(cls) -> str:
        return cls.parsing_model

    # -------------------------------------------------------------------------
    @classmethod
    def get_clinical_model(cls) -> str:
        return cls.clinical_model

    # -------------------------------------------------------------------------
    @classmethod
    def get_llm_provider(cls) -> str:
        return cls.llm_provider

    # -------------------------------------------------------------------------
    @classmethod
    def get_cloud_model(cls) -> str:
        return cls.cloud_model

    # -------------------------------------------------------------------------
    @classmethod
    def is_cloud_enabled(cls) -> bool:
        return cls.use_cloud_services

    # -------------------------------------------------------------------------
    @classmethod
    def get_ollama_temperature(cls) -> float:
        return cls.ollama_temperature

    # -------------------------------------------------------------------------
    @classmethod
    def is_ollama_reasoning_enabled(cls) -> bool:
        return cls.ollama_reasoning

    # -------------------------------------------------------------------------
    @classmethod
    def reset_defaults(cls) -> None:
        defaults = cls._get_defaults()
        cls.parsing_model = defaults.parsing_model
        cls.clinical_model = defaults.clinical_model
        cls.llm_provider = defaults.llm_provider
        cls.cloud_model = defaults.cloud_model
        cls.use_cloud_services = defaults.use_cloud_services
        cls.ollama_temperature = round(
            max(0.0, min(2.0, defaults.ollama_temperature)),
            2,
        )
        cls.ollama_reasoning = defaults.ollama_reasoning
        cls.revision = 0

    # -------------------------------------------------------------------------
    @classmethod
    def get_revision(cls) -> int:
        return cls.revision

    # -------------------------------------------------------------------------
    @classmethod
    def resolve_provider_and_model(
        cls,
        purpose: Literal["clinical", "parser"],
    ) -> tuple[str, str]:
        if cls.is_cloud_enabled():
            provider = cls.get_llm_provider()
            model = cls.get_cloud_model().strip()
            if not model:
                model = cls.get_parsing_model() if purpose == "parser" else cls.get_clinical_model()
        else:
            provider = "ollama"
            model = cls.get_parsing_model() if purpose == "parser" else cls.get_clinical_model()
        return provider, model.strip()


# [SERVER SETTINGS]
###############################################################################
@dataclass(frozen=True)
class FastAPISettings:
    title: str
    description: str
    version: str    

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatabaseSettings:
    embedded_database: bool
    engine: str | None          
    host: str | None            
    port: int | None            
    database_name: str | None
    username: str | None
    password: str | None
    ssl: bool                   
    ssl_ca: str | None         
    connect_timeout: int
    insert_batch_size: int

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DrugsMatcherSettings:
    direct_confidence: float
    master_confidence: float
    synonym_confidence: float
    partial_confidence: float
    fuzzy_confidence: float
    fuzzy_threshold: float
    token_max_frequency: int
    token_min_length: int
    normalization_cache_limit: int
    variant_cache_limit: int
    min_confidence: float
    catalog_excluded_term_suffixes: tuple[str, ...]
    catalog_token_ratio_threshold: float
    catalog_overall_ratio_threshold: float
    fuzzy_early_exit_ratio: float

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RagSettings:
    vector_collection_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    top_k_documents: int
    embedding_backend: str
    ollama_base_url: str
    ollama_embedding_model: str
    hf_embedding_model: str
    vector_index_metric: str
    vector_index_type: str
    reset_vector_collection: bool
    cloud_provider: str
    cloud_model: str
    cloud_embedding_model: str
    use_cloud_embeddings: bool

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExternalDataSettings:
    default_llm_timeout: float
    livertox_llm_timeout: float
    livertox_archive: str
    livertox_yield_interval: int
    livertox_skip_deterministic_ratio: float
    livertox_monograph_max_workers: int
    max_excerpt_length: int    

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class IngestionSettings:
    drug_name_min_length: int
    drug_name_max_length: int
    drug_name_max_tokens: int

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMRuntimeDefaults:
    parsing_model: str
    clinical_model: str
    llm_provider: str
    cloud_model: str
    use_cloud_services: bool
    ollama_temperature: float
    ollama_reasoning: bool
    ollama_host_default: str

# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ServerSettings:
    fastapi: FastAPISettings
    database: DatabaseSettings
    drugs_matcher: DrugsMatcherSettings
    rag: RagSettings
    external_data: ExternalDataSettings
    ingestion: IngestionSettings
    llm_defaults: LLMRuntimeDefaults


# [BUILDER FUNCTIONS]
###############################################################################
def build_fastapi_settings(data: dict[str, Any]) -> FastAPISettings:
    payload = ensure_mapping(data)
    return FastAPISettings(
        title=coerce_str(payload.get("title"), "DILIGENT Geospatial Search Backend"),
        version=coerce_str(payload.get("version"), "0.1.0"),
        description=coerce_str(payload.get("description"), "FastAPI backend"),        
    )

# -----------------------------------------------------------------------------
def build_database_settings(payload: dict[str, Any] | Any) -> DatabaseSettings:
    embedded = bool(payload.get("embedded_database", True))
    if embedded:
        # External fields are ignored entirely when embedded DB is active
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
            insert_batch_size=coerce_int(payload.get("insert_batch_size"), 1000, minimum=1),
        )

    # External DB mode
    engine_value = coerce_str_or_none(payload.get("engine")) or "postgres"
    normalized_engine = engine_value.lower() if engine_value else None
    return DatabaseSettings(
        embedded_database=False,
        engine=normalized_engine,
        host=coerce_str_or_none(payload.get("host")),
        port=coerce_int(payload.get("port"), 5432, minimum=1, maximum=65535),
        database_name=coerce_str_or_none(payload.get("database_name")),
        username=coerce_str_or_none(payload.get("username")),
        password=coerce_str_or_none(payload.get("password")),
        ssl=bool(payload.get("ssl", False)),
        ssl_ca=coerce_str_or_none(payload.get("ssl_ca")),
        connect_timeout=coerce_int(payload.get("connect_timeout"), 10, minimum=1),
        insert_batch_size=coerce_int(payload.get("insert_batch_size"), 1000, minimum=1),
    )

# -----------------------------------------------------------------------------
def build_drugs_matcher_settings(data: dict[str, Any]) -> DrugsMatcherSettings:
    suffixes_value = data.get("catalog_excluded_term_suffixes", ["PCK"])
    if isinstance(suffixes_value, (list, tuple, set)):
        candidates = list(suffixes_value)
    else:
        candidates = [suffixes_value]
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
        normalization_cache_limit=coerce_positive_int(
            data.get("normalization_cache_limit"),
            10000,
        ),
        variant_cache_limit=coerce_positive_int(
            data.get("variant_cache_limit"),
            10000,
        ),
        min_confidence=coerce_float(data.get("min_confidence"), 0.40),
        catalog_excluded_term_suffixes=suffix_tuple,
        catalog_token_ratio_threshold=coerce_float(
            data.get("catalog_token_ratio_threshold"),
            0.93,
        ),
        catalog_overall_ratio_threshold=coerce_float(
            data.get("catalog_overall_ratio_threshold"),
            0.93,
        ),
        fuzzy_early_exit_ratio=coerce_float(
            data.get("fuzzy_early_exit_ratio"),
            0.95,
        ),
    )

# -----------------------------------------------------------------------------
def build_rag_settings(
    data: dict[str, Any],
    *,
    default_provider: str,
    default_cloud_model: str,
    default_ollama_host: str,
) -> RagSettings:
    embedding_backend = coerce_str(data.get("embedding_backend"), "ollama")
    return RagSettings(
        vector_collection_name=coerce_str(data.get("vector_collection_name"), "documents"),
        chunk_size=coerce_positive_int(data.get("chunk_size"), 1_024),
        chunk_overlap=coerce_positive_int(data.get("chunk_overlap"), 128),
        embedding_batch_size=coerce_positive_int(
            data.get("embedding_batch_size"),
            DEFAULT_EMBEDDING_BATCH_SIZE,
        ),
        top_k_documents=coerce_positive_int(data.get("top_k_documents"), 3),
        embedding_backend=embedding_backend,
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
    )

# -----------------------------------------------------------------------------
def build_external_data_settings(
    data: dict[str, Any],
    *,
    fallback_timeout: float,
) -> ExternalDataSettings:
    default_llm_timeout = coerce_float(data.get("default_llm_timeout"), fallback_timeout)
    livertox_timeout = coerce_float(
        data.get("livertox_llm_timeout"),
        default_llm_timeout,
    )
    return ExternalDataSettings(
        default_llm_timeout=default_llm_timeout,
        livertox_llm_timeout=livertox_timeout,
        livertox_archive=coerce_str(data.get("livertox_archive"), "livertox_NBK547852.tar.gz"),
        livertox_yield_interval=coerce_positive_int(data.get("livertox_yield_interval"), 25),
        livertox_skip_deterministic_ratio=coerce_float(
            data.get("livertox_skip_deterministic_ratio"),
            0.80,
        ),
        livertox_monograph_max_workers=coerce_positive_int(
            data.get("livertox_monograph_max_workers"),
            4,
        ),
        max_excerpt_length=coerce_positive_int(data.get("max_excerpt_length"), 8_000),        
    )

# -----------------------------------------------------------------------------
def build_ingestion_settings(data: dict[str, Any]) -> IngestionSettings:
    min_length = coerce_positive_int(data.get("drug_name_min_length"), 3)
    max_length = coerce_positive_int(data.get("drug_name_max_length"), 200)
    tokens = coerce_positive_int(data.get("drug_name_max_tokens"), 8)
    if max_length < min_length:
        max_length = min_length
    return IngestionSettings(
        drug_name_min_length=min_length,
        drug_name_max_length=max_length,
        drug_name_max_tokens=tokens,
    )

# -----------------------------------------------------------------------------
def build_llm_runtime_defaults(data: dict[str, Any]) -> LLMRuntimeDefaults:
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
        ollama_reasoning=coerce_bool(data.get("ollama_reasoning"), False),
        ollama_host_default=coerce_str(data.get("ollama_host_default"), "http://localhost:11434"),
    )

# -----------------------------------------------------------------------------
def build_server_settings(data: dict[str, Any] | Any) -> ServerSettings:
    payload = ensure_mapping(data)
    fastapi_payload = ensure_mapping(payload.get("fastapi"))
    database_payload = ensure_mapping(payload.get("database"))
    drugs_matcher_payload = ensure_mapping(payload.get("drugs_matcher"))
    rag_payload = ensure_mapping(payload.get("rag"))
    external_data_payload = ensure_mapping(payload.get("external_data"))
    ingestion_payload = ensure_mapping(payload.get("ingestion"))
    llm_defaults_payload = ensure_mapping(
        payload.get("llm_defaults") or payload.get("llm_runtime_defaults")
    )
    llm_defaults = build_llm_runtime_defaults(llm_defaults_payload)
    default_provider = llm_defaults.llm_provider
    default_cloud_model = llm_defaults.cloud_model
    default_ollama_host = coerce_str(payload.get("ollama_base_url"), "http://localhost:11434")    

    rag_settings = build_rag_settings(
        rag_payload,
        default_provider=default_provider,
        default_cloud_model=default_cloud_model,
        default_ollama_host=default_ollama_host,
    )

    external_data_settings = build_external_data_settings(
        external_data_payload,
        fallback_timeout=30.0,
    )

    return ServerSettings(
        fastapi=build_fastapi_settings(fastapi_payload),
        database=build_database_settings(database_payload),
        drugs_matcher=build_drugs_matcher_settings(drugs_matcher_payload),
        rag=rag_settings,
        external_data=external_data_settings,
        ingestion=build_ingestion_settings(ingestion_payload),
        llm_defaults=llm_defaults,
    )


# [SERVER CONFIGURATION LOADER]
###############################################################################
def get_server_settings(config_path: str | None = None) -> ServerSettings:
    path = config_path or SERVER_CONFIGURATION_FILE
    payload = load_configuration_data(path)
    settings = build_server_settings(payload)
    LLMRuntimeConfig.configure(settings.llm_defaults)
    return settings


server_settings = get_server_settings()

