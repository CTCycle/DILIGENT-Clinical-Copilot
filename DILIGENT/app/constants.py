from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, "DILIGENT")
SETUP_DIR = join(PROJECT_DIR, "setup")
RSC_PATH = join(PROJECT_DIR, "resources")
MODELS_PATH = join(RSC_PATH, "models")
DATA_PATH = join(RSC_PATH, "database")
DOCS_PATH = join(DATA_PATH, "documents")
SOURCES_PATH = join(DATA_PATH, "sources")
TASKS_PATH = join(DATA_PATH, "tasks")
LOGS_PATH = join(RSC_PATH, "logs")
VECTOR_DB_PATH = join(DATA_PATH, "vectors")

# [ENDPOINS]
###############################################################################
API_BASE_URL = "http://127.0.0.1:8000"
CLINICAL_API_URL = "/clinical"

# [LLM / PROVIDERS]
###############################################################################
# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

PARSING_MODEL_CHOICES = [
    "qwen3:1.7b",
    "qwen3:8b",
    "qwen3:14b",
    "llama3.1:8b",
    "mistral-nemo:12b",
    "gemma2:9b",
    "phi3.5:mini",
    "phi3:medium",
]
CLINICAL_MODEL_CHOICES = [
    "gpt-oss:20b",
    "llama3.1:8b",
    "llama3.1:70b",
    "phi3.5:mini",
    "phi3.5:moe",
    "deepseek-r1:14b",
    "alibayram/medgemma:4b",
    "alibayram/medgemma:27bgemma3:9b",
    "gemma3:27b",
]
OPENAI_CLOUD_MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
GEMINI_CLOUD_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro",
    "gemini-1.0-pro-vision",
]
CLOUD_MODEL_CHOICES: dict[str, list[str]] = {
    "openai": OPENAI_CLOUD_MODELS,
    "gemini": GEMINI_CLOUD_MODELS,
}

# [EXTERNAL DATA SOURCES - API URLS]
###############################################################################
ATC_BASE_URL = "https://atcddd.fhi.no/atc_ddd_index/"
LIVERTOX_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
OPENFDA_DOWNLOAD_BASE_URL = "https://download.open.fda.gov"
OPENFDA_DOWNLOAD_CATALOG_URL = "https://api.fda.gov/download.json"
OPENFDA_DRUG_EVENT_DATASET = "drug/event"
OPENFDA_DRUG_EVENT_INDEX = "drug-event.json"
