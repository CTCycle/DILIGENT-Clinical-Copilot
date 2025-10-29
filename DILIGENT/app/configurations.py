from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from DILIGENT.app.constants import SETUP_DIR, CLOUD_MODEL_CHOICES

CONFIGURATION_CACHE: dict[str, Any] | None = None
CONFIGURATION_FILE = os.path.join(SETUP_DIR, "configurations.json")

DEFAULT_CONFIGURATION: dict[str, Any] = {
    "ollama_host_default": "http://localhost:11434",
    "cloud_providers": ["openai", "gemini"],
    "client_defaults": {
        "default_parsing_model": "qwen3:8b",
        "default_clinical_model": "gpt-oss:20b",
        "default_cloud_provider": "openai",
        "default_cloud_model": "gpt-4.1-mini",
        "default_use_cloud_services": False,
        "default_ollama_temperature": 0.7,
        "default_ollama_reasoning": False,
    },
    "rag": {
        "vector_collection_name": "documents",
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "embedding_backend": "ollama",
        "ollama_base_url": "http://localhost:11434",
        "ollama_embedding_model": "nomic-embed-text",
        "hf_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_index_metric": "cosine",
        "vector_index_type": "IVF_FLAT",
        "reset_vector_collection": True,
    },
    "external_data": {
        "default_llm_timeout_seconds": 3600.0,
        "livertox_llm_timeout_seconds": 3600.0,
        "livertox_archive": "livertox_NBK547852.tar.gz",
        "livertox_yield_interval": 25,
        "livertox_skip_deterministic_ratio": 0.8,
        "livertox_monograph_max_workers": 4,
        "max_excerpt_length": 8000,
        "llm_null_match_names": [
            "",
            "none",
            "no match",
            "no matches",
            "not found",
            "unknown",
            "not applicable",
            "n a",
        ],
    },
    "clinical_analysis": {
        "alt_labels": ["ALT", "ALAT"],
        "alp_labels": ["ALP"],
    },
}


def load_configuration_file() -> dict[str, Any]:
    if os.path.exists(CONFIGURATION_FILE):
        try:
            with open(CONFIGURATION_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to load configuration from {CONFIGURATION_FILE}"
            ) from exc
    return json.loads(json.dumps(DEFAULT_CONFIGURATION))


def get_nested_value(data: dict[str, Any], *keys: str, default: Any | None = None) -> Any:
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


CONFIGURATION_DATA = load_configuration_file()


def get_configuration() -> dict[str, Any]:
    return CONFIGURATION_DATA


def get_configuration_value(*keys: str, default: Any | None = None) -> Any:
    return get_nested_value(CONFIGURATION_DATA, *keys, default=default)


CLOUD_PROVIDERS = list(get_configuration_value("cloud_providers", default=[]))
CLIENT_DEFAULTS = get_configuration_value("client_defaults", default={})
DEFAULT_PARSING_MODEL = CLIENT_DEFAULTS.get("default_parsing_model", "")
DEFAULT_CLINICAL_MODEL = CLIENT_DEFAULTS.get("default_clinical_model", "")
DEFAULT_CLOUD_PROVIDER = CLIENT_DEFAULTS.get("default_cloud_provider", "")
DEFAULT_CLOUD_MODEL = CLIENT_DEFAULTS.get("default_cloud_model", "")
DEFAULT_USE_CLOUD_SERVICES = CLIENT_DEFAULTS.get("default_use_cloud_services", False)
DEFAULT_OLLAMA_TEMPERATURE = CLIENT_DEFAULTS.get("default_ollama_temperature", 0.7)
DEFAULT_OLLAMA_REASONING = CLIENT_DEFAULTS.get("default_ollama_reasoning", False)

if DEFAULT_CLOUD_PROVIDER not in CLOUD_PROVIDERS and CLOUD_PROVIDERS:
    DEFAULT_CLOUD_PROVIDER = CLOUD_PROVIDERS[0]

cloud_models = CLOUD_MODEL_CHOICES.get(DEFAULT_CLOUD_PROVIDER, [])
if DEFAULT_CLOUD_MODEL not in cloud_models:
    DEFAULT_CLOUD_MODEL = cloud_models[0] if cloud_models else ""

OLLAMA_HOST_DEFAULT = get_configuration_value("ollama_host_default", default="")

RAG_CONFIGURATION = get_configuration_value("rag", default={})
VECTOR_COLLECTION_NAME = RAG_CONFIGURATION.get("vector_collection_name", "documents")
RAG_CHUNK_SIZE = RAG_CONFIGURATION.get("chunk_size", 1024)
RAG_CHUNK_OVERLAP = RAG_CONFIGURATION.get("chunk_overlap", 128)
RAG_EMBEDDING_BACKEND = RAG_CONFIGURATION.get("embedding_backend", "ollama")
RAG_OLLAMA_BASE_URL = RAG_CONFIGURATION.get("ollama_base_url", OLLAMA_HOST_DEFAULT)
RAG_OLLAMA_EMBEDDING_MODEL = RAG_CONFIGURATION.get("ollama_embedding_model", "")
RAG_HF_EMBEDDING_MODEL = RAG_CONFIGURATION.get("hf_embedding_model", "")
RAG_VECTOR_INDEX_METRIC = RAG_CONFIGURATION.get("vector_index_metric", "cosine")
RAG_VECTOR_INDEX_TYPE = RAG_CONFIGURATION.get("vector_index_type", "IVF_FLAT")
RAG_RESET_VECTOR_COLLECTION = RAG_CONFIGURATION.get("reset_vector_collection", True)

EXTERNAL_DATA_CONFIGURATION = get_configuration_value("external_data", default={})
DEFAULT_LLM_TIMEOUT_SECONDS = EXTERNAL_DATA_CONFIGURATION.get(
    "default_llm_timeout_seconds", 3_600.0
)
LIVERTOX_LLM_TIMEOUT_SECONDS = EXTERNAL_DATA_CONFIGURATION.get(
    "livertox_llm_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS
)
LIVERTOX_ARCHIVE = EXTERNAL_DATA_CONFIGURATION.get(
    "livertox_archive", "livertox_NBK547852.tar.gz"
)
LIVERTOX_YIELD_INTERVAL = EXTERNAL_DATA_CONFIGURATION.get("livertox_yield_interval", 25)
LIVERTOX_SKIP_DETERMINISTIC_RATIO = EXTERNAL_DATA_CONFIGURATION.get(
    "livertox_skip_deterministic_ratio", 0.80
)
LIVERTOX_MONOGRAPH_MAX_WORKERS = EXTERNAL_DATA_CONFIGURATION.get(
    "livertox_monograph_max_workers", 4
)
MAX_EXCERPT_LENGTH = EXTERNAL_DATA_CONFIGURATION.get("max_excerpt_length", 8000)
LLM_NULL_MATCH_NAMES = set(
    EXTERNAL_DATA_CONFIGURATION.get("llm_null_match_names", [])
)

CLINICAL_ANALYSIS_CONFIGURATION = get_configuration_value("clinical_analysis", default={})
ALT_LABELS = set(CLINICAL_ANALYSIS_CONFIGURATION.get("alt_labels", []))
ALP_LABELS = set(CLINICAL_ANALYSIS_CONFIGURATION.get("alp_labels", []))


###############################################################################
class Configuration:
    def __init__(self) -> None:
        self.configuration = get_configuration()

    # -------------------------------------------------------------------------
    def get_configuration(self) -> dict[str, Any]:
        return self.configuration


###############################################################################
@dataclass
class ClientRuntimeConfig:
    parsing_model: str = DEFAULT_PARSING_MODEL
    clinical_model: str = DEFAULT_CLINICAL_MODEL
    llm_provider: str = DEFAULT_CLOUD_PROVIDER
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
        if value and value != cls.llm_provider:
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
        cls.llm_provider = DEFAULT_CLOUD_PROVIDER
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
