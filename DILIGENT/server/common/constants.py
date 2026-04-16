from __future__ import annotations

from pathlib import Path

from DILIGENT.server.common.utils.catalog_loader import CatalogLoader

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
RXNAV_CURATED_ALIASES_PATH = str(Path(SOURCES_PATH) / "rxnav_curated_aliases.json")
ENV_FILE_PATH = str(Path(SETTING_PATH) / ".env")
DATABASE_FILENAME = "database.db"

###############################################################################
CONFIGURATIONS_FILE = str(Path(SETTING_PATH) / "configurations.json")

# [ENDPOINTS]
###############################################################################
CLINICAL_API_URL = "/clinical"

# [APP DEFAULTS]
###############################################################################
FASTAPI_TITLE = "DILI Backend"
FASTAPI_VERSION = "1.0.0"
FASTAPI_DESCRIPTION = "FastAPI backend"
OLLAMA_DEFAULT_HOST = "localhost"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_SCHEME = "http"

# [EXPORTS]
###############################################################################
REPORT_EXPORT_DIRECTORY_PREFIX = "diligent_report_"
REPORT_EXPORT_FILENAME = "clinical_report.md"

# [LLM / PROVIDERS]
###############################################################################
# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"

PARSING_MODEL_CHOICES = CatalogLoader.get_string_list("llm_models.json", "parsing_model_choices")
CLINICAL_MODEL_CHOICES = CatalogLoader.get_string_list("llm_models.json", "clinical_model_choices")
OPENAI_CLOUD_MODELS = CatalogLoader.get_string_list("llm_models.json", "openai_cloud_models")
GEMINI_CLOUD_MODELS = CatalogLoader.get_string_list("llm_models.json", "gemini_cloud_models")
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
TABLE_DRUG_DILI_ANNOTATIONS = "drug_dili_annotations"
TABLE_DRUG_LABEL_DOCUMENTS = "drug_label_documents"
TABLE_DRUG_LABEL_SECTIONS = "drug_label_sections"
TABLE_MODEL_SELECTIONS = "model_selections"
TABLE_RUNTIME_SETTINGS = "runtime_settings"
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
DILIRANK_SOURCE_URL = "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset"
DILIST_SOURCE_URL = "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-severity-and-toxicity-dilist-dataset"
DAILYMED_RXNORM_SETID_MAPPING_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/rxnorm_mappings.zip"
DAILYMED_LABEL_XML_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
DOCUMENT_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".xml", ".docx", ".doc"}
TEXT_FILE_FALLBACK_ENCODINGS = ("utf-8", "utf-16", "latin-1", "iso-8859-1")
DRUG_NAME_ALLOWED_PATTERN = r"[A-Za-z0-9\s\-/(),'+\.]+"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DAILYMED_SECTION_WHITELIST = (
    "boxed_warning",
    "warnings_and_precautions",
    "adverse_reactions",
    "contraindications",
    "use_in_specific_populations",
)
HEPATIC_KEYWORDS = {
    "liver",
    "hepatic",
    "hepatotoxic",
    "hepatitis",
    "cholestatic",
    "bilirubin",
    "transaminase",
    "alt",
    "ast",
    "alp",
    "drug induced liver injury",
    "dili",
}

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

MATCHING_STOPWORDS = CatalogLoader.get_string_set("text_normalization.json", "matching_stopwords")

CLINICAL_GENERIC_TERMS = CatalogLoader.get_string_set("text_normalization.json", "clinical_generic_terms")

RXNAV_SYNONYM_STOPWORDS = MATCHING_STOPWORDS | CLINICAL_GENERIC_TERMS

# [DILI DEFAULTS]
###############################################################################
DEFAULT_DILI_CLASSIFICATION = "indeterminate"
NO_CLINICAL_CONTEXT_FALLBACK = "No additional clinical context provided."
UNKNOWN_R_SCORE_TOKEN = "R=NA"
R_SCORE_HEPATOCELLULAR_THRESHOLD = 5.0
R_SCORE_CHOLESTATIC_THRESHOLD = 2.0
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
FALSY_ENV_VALUES = {"0", "false", "no", "off"}

__all__ = [
    "MATCHING_STOPWORDS",
    "CLINICAL_GENERIC_TERMS",
    "RXNAV_SYNONYM_STOPWORDS",
]
