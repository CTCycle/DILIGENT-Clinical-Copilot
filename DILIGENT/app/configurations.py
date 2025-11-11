from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from DILIGENT.app.constants import (
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    PARSING_MODEL_CHOICES,
    SETUP_DIR,
)
from DILIGENT.app.utils.types import (
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_positive_int,
    coerce_str,
)

CONFIGURATION_FILE = os.path.join(SETUP_DIR, "configurations.json")


###############################################################################
def load_configuration_file() -> dict[str, Any]:
    if not os.path.exists(CONFIGURATION_FILE):
        return {}
    try:
        with open(CONFIGURATION_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to load configuration from {CONFIGURATION_FILE}"
        ) from exc
    if not isinstance(payload, dict):
        return {}
    return payload


# -----------------------------------------------------------------------------
def get_nested_value(data: dict[str, Any], *keys: str, default: Any | None = None) -> Any:
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# -----------------------------------------------------------------------------
def get_configuration_value(*keys: str, default: Any | None = None) -> Any:
    return get_nested_value(CONFIGURATION_DATA, *keys, default=default)


# -----------------------------------------------------------------------------
def ensure_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


CONFIGURATION_DATA = load_configuration_file()


###############################################################################
@dataclass(frozen=True)
class BackendSettings:
    title: str
    version: str
    description: str


###############################################################################
@dataclass(frozen=True)
class UIRuntimeSettings:
    host: str
    port: int
    title: str
    mount_path: str
    redirect_path: str
    show_welcome_message: bool
    reconnect_timeout: int


###############################################################################
@dataclass(frozen=True)
class APISettings:
    base_url: str


###############################################################################
@dataclass(frozen=True)
class HTTPSettings:
    timeout: float


###############################################################################
@dataclass(frozen=True)
class DatabaseSettings:
    insert_batch_size: int


###############################################################################
@dataclass(frozen=True)
class DrugsMatcherSettings:
    direct_confidence: float
    master_confidence: float
    synonym_confidence: float
    partial_confidence: float
    fuzzy_confidence: float
    fuzzy_threshold: float
    token_max_frequency: int
    min_confidence: float
    catalog_excluded_term_suffixes: tuple[str, ...]


# -----------------------------------------------------------------------------
def build_backend_settings(data: dict[str, Any]) -> BackendSettings:
    return BackendSettings(
        title=coerce_str(data.get("title"), "DILIGENT Clinical Copilot Backend"),
        version=coerce_str(data.get("version"), "0.1.0"),
        description=coerce_str(data.get("description"), "FastAPI backend"),
    )


# -----------------------------------------------------------------------------
def build_ui_runtime_settings(data: dict[str, Any]) -> UIRuntimeSettings:
    return UIRuntimeSettings(
        host=coerce_str(data.get("host"), "0.0.0.0"),
        port=coerce_positive_int(data.get("port"), 7861),
        title=coerce_str(data.get("title"), "DILIGENT Clinical Copilot"),
        mount_path=coerce_str(data.get("mount_path"), "/ui"),
        redirect_path=coerce_str(data.get("redirect_path"), "/ui"),
        show_welcome_message=coerce_bool(data.get("show_welcome_message"), False),
        reconnect_timeout=coerce_positive_int(data.get("reconnect_timeout"), 180),
    )


# -----------------------------------------------------------------------------
def build_api_settings(data: dict[str, Any]) -> APISettings:
    return APISettings(
        base_url=coerce_str(data.get("base_url"), "http://127.0.0.1:8000"),
    )


# -----------------------------------------------------------------------------
def build_http_settings(data: dict[str, Any]) -> HTTPSettings:
    return HTTPSettings(
        timeout=coerce_float(data.get("timeout"), 3_600.0),
    )


# -----------------------------------------------------------------------------
def build_database_settings(data: dict[str, Any]) -> DatabaseSettings:
    return DatabaseSettings(
        insert_batch_size=coerce_positive_int(
            data.get("insert_batch_size"),
            1_000,
        ),
    )


# -----------------------------------------------------------------------------
def build_drugs_matcher_settings(data: dict[str, Any]) -> DrugsMatcherSettings:
    suffixes_value = data.get("catalog_excluded_term_suffixes", ["PCK"])
    suffixes: list[str] = []
    if isinstance(suffixes_value, (list, tuple, set)):
        candidates = list(suffixes_value)
    else:
        candidates = [suffixes_value]
    for entry in candidates:
        text = coerce_str(entry, "")
        if text:
            suffixes.append(text.upper())
    suffix_tuple = tuple(suffixes) if suffixes else ("PCK",)
    return DrugsMatcherSettings(
        direct_confidence=coerce_float(data.get("direct_confidence"), 1.0),
        master_confidence=coerce_float(data.get("master_confidence"), 0.92),
        synonym_confidence=coerce_float(data.get("synonym_confidence"), 0.90),
        partial_confidence=coerce_float(data.get("partial_confidence"), 0.86),
        fuzzy_confidence=coerce_float(data.get("fuzzy_confidence"), 0.84),
        fuzzy_threshold=coerce_float(data.get("fuzzy_threshold"), 0.85),
        token_max_frequency=coerce_positive_int(data.get("token_max_frequency"), 3),
        min_confidence=coerce_float(data.get("min_confidence"), 0.40),
        catalog_excluded_term_suffixes=suffix_tuple,
    )


BACKEND_SETTINGS = build_backend_settings(
    ensure_mapping(get_configuration_value("backend", default={}))
)
UI_RUNTIME_SETTINGS = build_ui_runtime_settings(
    ensure_mapping(get_configuration_value("ui_runtime", default={}))
)
API_SETTINGS = build_api_settings(
    ensure_mapping(get_configuration_value("api", default={}))
)
HTTP_SETTINGS = build_http_settings(
    ensure_mapping(get_configuration_value("http", default={}))
)
DATABASE_SETTINGS = build_database_settings(
    ensure_mapping(get_configuration_value("database", default={}))
)
DRUGS_MATCHER_SETTINGS = build_drugs_matcher_settings(
    ensure_mapping(get_configuration_value("drugs_matcher", default={}))
)

CLIENT_RUNTIME_DEFAULTS = ensure_mapping(
    get_configuration_value("client_runtime_defaults", default={})
)

DEFAULT_PARSING_MODEL = coerce_str(
    CLIENT_RUNTIME_DEFAULTS.get("parsing_model"),
    PARSING_MODEL_CHOICES[0] if PARSING_MODEL_CHOICES else "",
)
DEFAULT_CLINICAL_MODEL = coerce_str(
    CLIENT_RUNTIME_DEFAULTS.get("clinical_model"),
    CLINICAL_MODEL_CHOICES[0] if CLINICAL_MODEL_CHOICES else "",
)
DEFAULT_LLM_PROVIDER = coerce_str(
    CLIENT_RUNTIME_DEFAULTS.get("llm_provider"),
    "openai",
)
DEFAULT_CLOUD_PROVIDER = DEFAULT_LLM_PROVIDER
_provider_models = CLOUD_MODEL_CHOICES.get(DEFAULT_CLOUD_PROVIDER, [])
DEFAULT_CLOUD_MODEL = coerce_str(
    CLIENT_RUNTIME_DEFAULTS.get("cloud_model"),
    _provider_models[0] if _provider_models else "",
)
DEFAULT_USE_CLOUD_SERVICES = coerce_bool(
    CLIENT_RUNTIME_DEFAULTS.get("use_cloud_services"),
    False,
)
DEFAULT_OLLAMA_TEMPERATURE = coerce_float(
    CLIENT_RUNTIME_DEFAULTS.get("ollama_temperature"),
    0.7,
)
DEFAULT_OLLAMA_REASONING = coerce_bool(
    CLIENT_RUNTIME_DEFAULTS.get("ollama_reasoning"),
    False,
)

DEFAULT_CLOUD_EMBEDDING_MODEL = ""

OLLAMA_HOST_DEFAULT = coerce_str(
    get_configuration_value("ollama_host_default", default=""),
    "",
)

RAG_CONFIGURATION = ensure_mapping(get_configuration_value("rag", default={}))
VECTOR_COLLECTION_NAME = coerce_str(
    RAG_CONFIGURATION.get("vector_collection_name"),
    "documents",
)
RAG_CHUNK_SIZE = coerce_positive_int(RAG_CONFIGURATION.get("chunk_size"), 1_024)
RAG_CHUNK_OVERLAP = coerce_positive_int(
    RAG_CONFIGURATION.get("chunk_overlap"),
    128,
)
RAG_EMBEDDING_BACKEND = coerce_str(
    RAG_CONFIGURATION.get("embedding_backend"),
    "ollama",
)
RAG_OLLAMA_BASE_URL = coerce_str(
    RAG_CONFIGURATION.get("ollama_base_url"),
    OLLAMA_HOST_DEFAULT,
)
RAG_OLLAMA_EMBEDDING_MODEL = coerce_str(
    RAG_CONFIGURATION.get("ollama_embedding_model"),
    "",
)
RAG_HF_EMBEDDING_MODEL = coerce_str(
    RAG_CONFIGURATION.get("hf_embedding_model"),
    "",
)
RAG_VECTOR_INDEX_METRIC = coerce_str(
    RAG_CONFIGURATION.get("vector_index_metric"),
    "cosine",
)
RAG_VECTOR_INDEX_TYPE = coerce_str(
    RAG_CONFIGURATION.get("vector_index_type"),
    "IVF_FLAT",
)
RAG_RESET_VECTOR_COLLECTION = coerce_bool(
    RAG_CONFIGURATION.get("reset_vector_collection"),
    True,
)
RAG_TOP_K_DOCUMENTS = coerce_positive_int(
    RAG_CONFIGURATION.get("top_k_documents"),
    3,
)
RAG_CLOUD_PROVIDER = coerce_str(
    RAG_CONFIGURATION.get("cloud_provider"),
    DEFAULT_CLOUD_PROVIDER,
)
RAG_CLOUD_EMBEDDING_MODEL = coerce_str(
    RAG_CONFIGURATION.get("cloud_embedding_model"),
    DEFAULT_CLOUD_EMBEDDING_MODEL,
)
RAG_USE_CLOUD_EMBEDDINGS = coerce_bool(
    RAG_CONFIGURATION.get("use_cloud_embeddings"),
    False,
)

EXTERNAL_DATA_CONFIGURATION = ensure_mapping(
    get_configuration_value("external_data", default={})
)
DEFAULT_LLM_TIMEOUT = coerce_float(
    EXTERNAL_DATA_CONFIGURATION.get("default_llm_timeout"),
    HTTP_SETTINGS.timeout,
)
LIVERTOX_LLM_TIMEOUT = coerce_float(
    EXTERNAL_DATA_CONFIGURATION.get("livertox_llm_timeout"),
    DEFAULT_LLM_TIMEOUT,
)
LIVERTOX_ARCHIVE = coerce_str(
    EXTERNAL_DATA_CONFIGURATION.get("livertox_archive"),
    "livertox_NBK547852.tar.gz",
)
LIVERTOX_YIELD_INTERVAL = coerce_positive_int(
    EXTERNAL_DATA_CONFIGURATION.get("livertox_yield_interval"),
    25,
)
LIVERTOX_SKIP_DETERMINISTIC_RATIO = coerce_float(
    EXTERNAL_DATA_CONFIGURATION.get("livertox_skip_deterministic_ratio"),
    0.80,
)
LIVERTOX_MONOGRAPH_MAX_WORKERS = coerce_positive_int(
    EXTERNAL_DATA_CONFIGURATION.get("livertox_monograph_max_workers"),
    4,
)
MAX_EXCERPT_LENGTH = coerce_positive_int(
    EXTERNAL_DATA_CONFIGURATION.get("max_excerpt_length"),
    8_000,
)
LLM_NULL_MATCH_NAMES = EXTERNAL_DATA_CONFIGURATION.get("llm_null_match_names", [])

CLINICAL_ANALYSIS_CONFIGURATION = ensure_mapping(
    get_configuration_value("clinical_analysis", default={})
)

###############################################################################
class ClientRuntimeConfig:
    parsing_model: str = DEFAULT_PARSING_MODEL
    clinical_model: str = DEFAULT_CLINICAL_MODEL
    llm_provider: str = DEFAULT_LLM_PROVIDER
    cloud_model: str = DEFAULT_CLOUD_MODEL
    use_cloud_services: bool = DEFAULT_USE_CLOUD_SERVICES
    ollama_temperature: float = DEFAULT_OLLAMA_TEMPERATURE
    ollama_reasoning: bool = DEFAULT_OLLAMA_REASONING
    revision: int = 0

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
        value = provider.strip()
        if not value:
            return cls.llm_provider
        if value not in CLOUD_MODEL_CHOICES:
            value = DEFAULT_LLM_PROVIDER
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
            if models and cls.cloud_model != models[0]:
                cls.cloud_model = models[0]
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
        try:
            parsed = float(value) if value is not None else cls.ollama_temperature
        except (TypeError, ValueError):
            parsed = cls.ollama_temperature
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
        cls.parsing_model = DEFAULT_PARSING_MODEL
        cls.clinical_model = DEFAULT_CLINICAL_MODEL
        cls.llm_provider = DEFAULT_LLM_PROVIDER
        cls.cloud_model = DEFAULT_CLOUD_MODEL
        cls.use_cloud_services = DEFAULT_USE_CLOUD_SERVICES
        cls.ollama_temperature = DEFAULT_OLLAMA_TEMPERATURE
        cls.ollama_reasoning = DEFAULT_OLLAMA_REASONING
        cls.revision = 0

    # -------------------------------------------------------------------------
    @classmethod
    def get_revision(cls) -> int:
        return cls.revision

    # -------------------------------------------------------------------------
    @classmethod
    def resolve_provider_and_model(
        cls, purpose: Literal["clinical", "parser"]
    ) -> tuple[str, str]:
        if cls.is_cloud_enabled():
            provider = cls.get_llm_provider()
            model = cls.get_cloud_model().strip()
            if not model:
                if purpose == "parser":
                    model = cls.get_parsing_model()
                else:
                    model = cls.get_clinical_model()
        else:
            provider = "ollama"
            if purpose == "parser":
                model = cls.get_parsing_model()
            else:
                model = cls.get_clinical_model()
        return provider, model.strip()


__all__ = [    
    "API_SETTINGS",
    "BACKEND_SETTINGS",
    "ClientRuntimeConfig",
    "DATABASE_SETTINGS",
    "DEFAULT_CLINICAL_MODEL",
    "DEFAULT_LLM_TIMEOUT",
    "HTTP_SETTINGS",
    "LIVERTOX_ARCHIVE",
    "LIVERTOX_LLM_TIMEOUT",
    "LIVERTOX_MONOGRAPH_MAX_WORKERS",
    "LIVERTOX_SKIP_DETERMINISTIC_RATIO",
    "LIVERTOX_YIELD_INTERVAL",
    "MAX_EXCERPT_LENGTH",
    "OLLAMA_HOST_DEFAULT",
    "DRUGS_MATCHER_SETTINGS",
    "RAG_CHUNK_OVERLAP",
    "RAG_CHUNK_SIZE",
    "RAG_CLOUD_EMBEDDING_MODEL",
    "RAG_CLOUD_PROVIDER",
    "RAG_CONFIGURATION",
    "RAG_EMBEDDING_BACKEND",
    "RAG_HF_EMBEDDING_MODEL",
    "RAG_OLLAMA_BASE_URL",
    "RAG_OLLAMA_EMBEDDING_MODEL",
    "RAG_RESET_VECTOR_COLLECTION",
    "RAG_TOP_K_DOCUMENTS",
    "RAG_USE_CLOUD_EMBEDDINGS",
    "RAG_VECTOR_INDEX_METRIC",
    "RAG_VECTOR_INDEX_TYPE",
    "UI_RUNTIME_SETTINGS",
    "VECTOR_COLLECTION_NAME",
]

