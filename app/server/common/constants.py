from __future__ import annotations

# [APP DEFAULTS]
###############################################################################
FASTAPI_ROOT_ENDPOINT = "/"
FASTAPI_API_PREFIX = "/api"
FASTAPI_ASSETS_ENDPOINT = "/assets"
FASTAPI_SPA_FALLBACK_ENDPOINT = "/{full_path:path}"
FASTAPI_TITLE = "DILI Backend"
FASTAPI_DESCRIPTION = "FastAPI backend"
FASTAPI_DOCS_URL = "/docs"
FASTAPI_REDOC_URL = "/redoc"
FASTAPI_OPENAPI_URL = "/openapi.json"
OLLAMA_DEFAULT_HOST = "127.0.0.1"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_SCHEME = "http"

# [ERROR HANDLING]
###############################################################################
REQUEST_ID_HEADER = "X-Request-ID"
GENERIC_FAILURE_MESSAGE = "Request could not be completed. Please retry."
TIMEOUT_FAILURE_MESSAGE = "Request timed out. Please retry."
DEPENDENCY_FAILURE_MESSAGE = "Service dependency unavailable. Please retry shortly."
MISSING_RESOURCE_MESSAGE = "Required resource was not found."

# [EXPORTS]
###############################################################################
REPORT_EXPORT_DIRECTORY_PREFIX = "diligent_report_"
REPORT_EXPORT_FILENAME = "clinical_report.md"

# [LLM / PROVIDERS]
###############################################################################
# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"


# [DATA SERIALIZATION]
###############################################################################
TABLE_CLINICAL_SESSIONS = "clinical_sessions"
TABLE_CLINICAL_SESSION_SECTIONS = "clinical_session_sections"
TABLE_CLINICAL_SESSION_RESULTS = "clinical_session_results"
TABLE_DRUGS = "drugs"
TABLE_DRUG_RXNORM_CODES = "drug_rxnorm_codes"
TABLE_DRUG_ALIASES = "drug_aliases"
TABLE_LIVERTOX_MONOGRAPHS = "livertox_monographs"
TABLE_ACCESS_KEYS = "access_keys"
TABLE_REFERENCE_CATALOG_ENTRIES = "reference_catalog_entries"
TABLE_REFERENCE_CATALOG_SEED_RUNS = "reference_catalog_seed_runs"

DEFAULT_DRUG_MATCH_TOKEN_MIN_LENGTH = 4
DEFAULT_DRUG_MATCH_CATALOG_INDEX_LIMIT = 75000
DEFAULT_DRUG_MATCH_SPELLING_CONFIDENCE = 0.94
DEFAULT_DRUG_MATCH_SPELLING_MIN_QUERY_LENGTH = 6
DEFAULT_DRUG_MATCH_SPELLING_SHORT_NAME_LENGTH = 10
DEFAULT_DRUG_MATCH_SPELLING_SHORT_MAX_DISTANCE = 1
DEFAULT_DRUG_MATCH_SPELLING_LONG_MAX_DISTANCE = 2

RXNORM_CATALOG_COLUMNS = [
    "rxcui",
    "raw_name",
    "term_type",
    "name",
    "brand_names",
    "synonyms",
]

LIVERTOX_REQUIRED_COLUMNS = [
    "nbk_id",
    "drug_name",
    "excerpt",
    "synonyms",
]
LIVERTOX_OPTIONAL_COLUMNS = {"nbk_id", "synonyms"}

LIVERTOX_COLUMNS = [
    "drug_name",
    "nbk_id",
    "ingredient",
    "brand_name",
    "synonyms",
    "excerpt",
    "likelihood_score",
    "last_update",
    "reference_count",
    "year_approved",
    "agent_classification",
    "primary_classification",
    "secondary_classification",
    "include_in_livertox",
    "source_url",
    "source_last_modified",
]

LIVERTOX_MASTER_COLUMNS = [
    "drug_name",
    "likelihood_score",
    "last_update",
    "reference_count",
    "year_approved",
    "agent_classification",
    "primary_classification",
    "secondary_classification",
    "include_in_livertox",
    "source_url",
    "source_last_modified",
]

# [EXTERNAL DATA SOURCES - API URLS]
###############################################################################
ATC_BASE_URL = "https://atcddd.fhi.no/atc_ddd_index/"
LIVERTOX_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
DOCUMENT_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".xml", ".docx", ".doc"}
TEXT_FILE_FALLBACK_ENCODINGS = ("utf-8", "utf-16", "latin-1", "iso-8859-1")
DRUG_NAME_ALLOWED_PATTERN = r"[A-Za-z0-9\s\-/(),'+\.]+"
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# [DILI DEFAULTS]
###############################################################################
DEFAULT_DILI_CLASSIFICATION = "indeterminate"
NO_CLINICAL_CONTEXT_FALLBACK = "No additional clinical context provided."
UNKNOWN_R_SCORE_TOKEN = "R=NA"
R_SCORE_HEPATOCELLULAR_THRESHOLD = 5.0
R_SCORE_CHOLESTATIC_THRESHOLD = 2.0
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
