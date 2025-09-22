from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, "Pharmagent")
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

# [LLM / PROVIDERS]
###############################################################################
# Local model server (Ollama)
OLLAMA_HOST_DEFAULT = "http://localhost:11434"

# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Runtime selection defaults
CLOUD_PROVIDERS = ["openai", "gemini"]
PARSING_MODEL_CHOICES = ["qwen3:14b", "phi3:mini"]
AGENT_MODEL_CHOICES = ["gpt-oss", "llama3.1:8b"]
DEFAULT_PARSING_MODEL = PARSING_MODEL_CHOICES[0]
DEFAULT_AGENT_MODEL = AGENT_MODEL_CHOICES[0]
DEFAULT_CLOUD_PROVIDER = CLOUD_PROVIDERS[0]

# [EXTERNAL DATA SOURCES]
###############################################################################
ATC_BASE_URL = "https://atcddd.fhi.no/atc_ddd_index/"
LIVERTOX_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
LIVERTOX_ARCHIVE = "livertox_NBK547852.tar.gz"

# [NLP / TRANSLATION]
###############################################################################
LANGUAGE_DETECTION_MODEL = "papluca/xlm-roberta-base-language-detection"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"
TRANSLATION_CONFIDENCE_THRESHOLD = 0.78
TRANSLATION_MAX_ATTEMPTS = 3
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


