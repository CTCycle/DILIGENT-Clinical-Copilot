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
AGENT_API_URL = "/agent"
BATCH_AGENT_API_URL = "/batch-agent"
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
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "mistral-nemo:12b",
    "mixtral:8x7b-instruct",
    "gemma2:9b",
    "phi3.5:mini",
    "phi3:medium",
]
AGENT_MODEL_CHOICES = [
    "gpt-oss:20b",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "llama3:8b",
    "llama3:70b",
    "llama3.2-vision:11b",
    "phi3.5:mini",
    "phi3.5:moe",
    "phi3:mini",
    "phi3:medium",
    "mistral-nemo:12b",
    "mistral:7b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:72b",
    "qwen2:7b",
    "qwen2:72b",
    "gemma2:9b",
    "gemma2:27b",
    "deepseek-coder-v2:16b",
]
OPENAI_CLOUD_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-tts",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-translation",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2024-04-09",
]
GEMINI_CLOUD_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
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
DEFAULT_AGENT_MODEL = AGENT_MODEL_CHOICES[0]
DEFAULT_CLOUD_PROVIDER = CLOUD_PROVIDERS[0]
DEFAULT_CLOUD_MODEL = CLOUD_MODEL_CHOICES[DEFAULT_CLOUD_PROVIDER][0]

# [EXTERNAL DATA SOURCES]
###############################################################################
ATC_BASE_URL = "https://atcddd.fhi.no/atc_ddd_index/"
LIVERTOX_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
LIVERTOX_ARCHIVE = "livertox_NBK547852.tar.gz"
DEFAULT_LLM_TIMEOUT_SECONDS = 1_800.0
LIVERTOX_LLM_TIMEOUT_SECONDS = DEFAULT_LLM_TIMEOUT_SECONDS
LIVERTOX_YIELD_INTERVAL = 25
LIVERTOX_SKIP_DETERMINISTIC_RATIO = 0.80
LIVERTOX_MONOGRAPH_MAX_WORKERS = 4
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


# [NLP / TRANSLATION]
###############################################################################
LANGUAGE_DETECTION_MODEL = "papluca/xlm-roberta-base-language-detection"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"
TRANSLATION_CONFIDENCE_THRESHOLD = 0.90
TRANSLATION_MAX_ATTEMPTS = 5
LANGUAGE_DETECTION_CONFIDENCE = 0.90
LANGUAGE_DETECTION_SAMPLE_LENGTH = 500

# Enable/disable the LLM translator fallback (Ollama or cloud as configured).
TRANSLATION_LLM_ENABLED = True
# Default LLM model to use if enabled and available via providers.initialize_llm_client()
# For local: "llama3.1:8b" (Ollama). For cloud, supply a provider-default in ClientRuntimeConfig.
TRANSLATION_LLM_MODEL = "llama3.1:8b"

# Confidence handling
# Minimum self-reported quality from the LLM (0..1) to accept without further retries.
TRANSLATION_LLM_CONFIDENCE_THRESHOLD = 0.80

# Penalize outputs with repeated sentences/tokens. If the repetition ratio exceeds this,
# we scale confidence by (1 - REPETITION_PENALTY_SCALE).
REPETITION_RATIO_THRESHOLD = 0.20
REPETITION_PENALTY_SCALE = 0.25

NLLB_LANGUAGE_CODES: dict[str, str] = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
}
