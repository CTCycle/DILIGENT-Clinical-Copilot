from __future__ import annotations

from pathlib import Path

# [PATHS]
###############################################################################
PROJECT_DIR = str(Path(__file__).resolve().parents[2])
ROOT_DIR = str(Path(PROJECT_DIR).parent)
SETTING_PATH = str(Path(PROJECT_DIR) / "settings")
RESOURCES_PATH = str(Path(PROJECT_DIR) / "resources")
MODELS_PATH = str(Path(RESOURCES_PATH) / "models")
SOURCES_PATH = str(Path(RESOURCES_PATH) / "sources")
ARCHIVES_PATH = str(Path(SOURCES_PATH) / "archives")
DOCS_PATH = str(Path(SOURCES_PATH) / "documents")
LOGS_PATH = str(Path(RESOURCES_PATH) / "logs")
VECTOR_DB_PATH = str(Path(SOURCES_PATH) / "vectors")
ENV_FILE_PATH = str(Path(SETTING_PATH) / ".env")
DATABASE_FILENAME = "database.db"

###############################################################################
CONFIGURATIONS_FILE = str(Path(SETTING_PATH) / "configurations.json")

# [ENDPOINTS]
###############################################################################
CLINICAL_API_URL = "/clinical"

# [EXPORTS]
###############################################################################
REPORT_EXPORT_DIRECTORY_PREFIX = "diligent_report_"
REPORT_EXPORT_FILENAME = "clinical_report.md"

# [LLM / PROVIDERS]
###############################################################################
# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"

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
    "alibayram/medgemma:27b",
    "gemma3:9b",
    "gemma3:27b",
]
OPENAI_CLOUD_MODELS = [
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
]
GEMINI_CLOUD_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
CLOUD_MODEL_CHOICES: dict[str, list[str]] = {
    "openai": OPENAI_CLOUD_MODELS,
    "gemini": GEMINI_CLOUD_MODELS,
}

# [DATA SERIALIZATION]
###############################################################################
TABLE_CLINICAL_SESSIONS = "clinical_sessions"
TABLE_CLINICAL_SESSION_SECTIONS = "clinical_session_sections"
TABLE_CLINICAL_SESSION_LABS = "clinical_session_labs"
TABLE_CLINICAL_SESSION_DRUGS = "clinical_session_drugs"
TABLE_CLINICAL_SESSION_RESULTS = "clinical_session_results"
TABLE_DRUGS = "drugs"
TABLE_DRUG_RXNORM_CODES = "drug_rxnorm_codes"
TABLE_DRUG_ALIASES = "drug_aliases"
TABLE_LIVERTOX_MONOGRAPHS = "livertox_monographs"
TABLE_MODEL_SELECTIONS = "model_selections"
TABLE_ACCESS_KEYS = "access_keys"

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

HEPATOTOXIC_MEDDRA_TERMS = {
    "hepatotoxicity",
    "drug induced liver injury",
    "drug-induced liver injury",
    "liver injury",
    "hepatic failure",
    "acute hepatic failure",
    "hepatitis cholestatic",
    "cholestasis",
    "liver disorder",
    "liver function test increased",
    "alanine aminotransferase increased",
    "aspartate aminotransferase increased",
    "alkaline phosphatase increased",
    "blood bilirubin increased",
}

MATCHING_STOPWORDS = {
    "and",
    "apply",
    "combo",
    "combination",
    "caps",
    "capsule",
    "capsules",
    "chewable",
    "cream",
    "dose",
    "doses",
    "drink",
    "drops",
    "elixir",
    "enteric",
    "extended",
    "foam",
    "for",
    "free",
    "gel",
    "granules",
    "im",
    "inj",
    "injection",
    "intramuscular",
    "intravenous",
    "iv",
    "kit",
    "liquid",
    "lotion",
    "mg",
    "ml",
    "nasal",
    "ointment",
    "of",
    "ophthalmic",
    "or",
    "oral",
    "pack",
    "packet",
    "packets",
    "patch",
    "plus",
    "powder",
    "po",
    "prefilled",
    "release",
    "sc",
    "sol",
    "solution",
    "soln",
    "spray",
    "sterile",
    "subcutaneous",
    "suppository",
    "susp",
    "suspension",
    "sustained",
    "syringe",
    "syrup",
    "tablet",
    "tablets",
    "the",
    "topical",
    "treat",
    "treatment",
    "therapy",
    "vial",
    "use",
    "with",
    "without",
}

CLINICAL_GENERIC_TERMS = {
    "administration",
    "applicator",
    "autoinjector",
    "auto-injector",
    "autoinjectors",
    "injector",
    "injectors",
    "device",
    "devices",
    "dosing",
    "inhaler",
    "inhalers",
    "infusion",
    "injectable",
    "injectables",
    "needle",
    "needles",
    "pen",
    "pens",
    "prefill",
    "pre-filled",
    "pump",
    "syringes",
}

RXNAV_SYNONYM_STOPWORDS = MATCHING_STOPWORDS | CLINICAL_GENERIC_TERMS

# [DILI DEFAULTS]
###############################################################################
DEFAULT_DILI_CLASSIFICATION = "indeterminate"
NO_CLINICAL_CONTEXT_FALLBACK = "No additional clinical context provided."
UNKNOWN_R_SCORE_TOKEN = "R=NA"
R_SCORE_HEPATOCELLULAR_THRESHOLD = 5.0
R_SCORE_CHOLESTATIC_THRESHOLD = 2.0

__all__ = [
    "MATCHING_STOPWORDS",
    "CLINICAL_GENERIC_TERMS",
    "RXNAV_SYNONYM_STOPWORDS",
]
