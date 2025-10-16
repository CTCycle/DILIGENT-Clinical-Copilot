from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, "DILIGENT")
RSC_PATH = join(PROJECT_DIR, "resources")
MODELS_PATH = join(RSC_PATH, "models")
DATA_PATH = join(RSC_PATH, "database")
DOCS_PATH = join(DATA_PATH, "documents")
SOURCES_PATH = join(DATA_PATH, "sources")
TASKS_PATH = join(DATA_PATH, "tasks")
LOGS_PATH = join(RSC_PATH, "logs")

# [ENDPOINS]
###############################################################################
API_BASE_URL = "http://127.0.0.1:8000"
CLINICAL_API_URL = "/clinical"
PHARMACOLOGY_LIVERTOX_FETCH_ENDPOINT = "/pharmacology/livertox/fetch"
PHARMACOLOGY_LIVERTOX_STATUS_ENDPOINT = "/pharmacology/livertox/status"

# [LLM / PROVIDERS]
###############################################################################
# Local model server (Ollama)
OLLAMA_HOST_DEFAULT = "http://localhost:11434"

# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Runtime selection defaults
CLOUD_PROVIDERS = ["openai", "gemini"]
PARSING_MODEL_CHOICES = [
    "qwen3:1.7b",    
    "qwen3:8b",
    "qwen3:14b",    
    "llama3.1:8b",
    "mistral-nemo:12b",
    "mixtral:8x7b-instruct",
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
    "gemma3:9b",
    "gemma3:27b",
]
OPENAI_CLOUD_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o"   
]
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
DEFAULT_PARSING_MODEL = "qwen3:8b"
DEFAULT_CLINICAL_MODEL = CLINICAL_MODEL_CHOICES[0]
DEFAULT_CLOUD_PROVIDER = CLOUD_PROVIDERS[0]
DEFAULT_CLOUD_MODEL = CLOUD_MODEL_CHOICES[DEFAULT_CLOUD_PROVIDER][0]

# [EXTERNAL DATA SOURCES]
###############################################################################
ATC_BASE_URL = "https://atcddd.fhi.no/atc_ddd_index/"
LIVERTOX_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
LIVERTOX_ARCHIVE = "livertox_NBK547852.tar.gz"
DEFAULT_LLM_TIMEOUT_SECONDS = 3_600.0
LIVERTOX_LLM_TIMEOUT_SECONDS = DEFAULT_LLM_TIMEOUT_SECONDS
LIVERTOX_YIELD_INTERVAL = 25
LIVERTOX_SKIP_DETERMINISTIC_RATIO = 0.80
LIVERTOX_MONOGRAPH_MAX_WORKERS = 4
MAX_EXCERPT_LENGTH = 8000
LLM_NULL_MATCH_NAMES = {
    "",
    "none",
    "no match",
    "no matches",
    "not found",
    "unknown",
    "not applicable",
    "n a",
}


# [CLINICAL ANALYSIS]
###############################################################################
DRUG_SUSPENSION_EXCLUSION_DAYS = 14
ALT_LABELS = {"ALT", "ALAT"}
ALP_LABELS = {"ALP"}

