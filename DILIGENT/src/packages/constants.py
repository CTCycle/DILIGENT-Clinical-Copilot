from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "DILIGENT")
SETUP_PATH = join(PROJECT_DIR, "setup")
RSC_PATH = join(PROJECT_DIR, "resources")
MODELS_PATH = join(RSC_PATH, "models")
DATA_PATH = join(RSC_PATH, "database")
DOCS_PATH = join(DATA_PATH, "documents")
SOURCES_PATH = join(DATA_PATH, "sources")
TASKS_PATH = join(DATA_PATH, "tasks")
LOGS_PATH = join(RSC_PATH, "logs")
VECTOR_DB_PATH = join(DATA_PATH, "vectors")
CONFIGURATION_FILE = join(SETUP_PATH, "configurations.json")

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
    "alibayram/medgemma:27b",
    "gemma3:9b",
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

LIVERTOX_COLUMNS = [
    "drug_name",
    "nbk_id",
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
    "ophthalmic",
    "oral",
    "pack",
    "packet",
    "packets",
    "plus",
    "powder",
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

__all__ = [
    "MATCHING_STOPWORDS",
    "CLINICAL_GENERIC_TERMS",
    "RXNAV_SYNONYM_STOPWORDS",
]
