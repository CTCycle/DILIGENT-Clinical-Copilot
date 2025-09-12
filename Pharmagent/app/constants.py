from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, "Pharmagent")
RSC_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RSC_PATH, "database")
DOCS_PATH = join(DATA_PATH, "documents")
SOURCES_PATH = join(DATA_PATH, "sources")
TASKS_PATH = join(DATA_PATH, "tasks")
LOGS_PATH = join(RSC_PATH, "logs")

# [ENDPOINS]
###############################################################################
API_BASE_URL = "http://127.0.0.1:8000"
PARSER_MODEL = "qwen3:14b"

# [LLM / PROVIDERS]
###############################################################################
# Local model server (Ollama)
OLLAMA_HOST_DEFAULT = "http://localhost:11434"

# Cloud provider API bases
OPENAI_API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# [EXTERNAL DATA SOURCES]
###############################################################################
ATC_BASE_URL = "https://atcddd.fhi.no/atc_ddd_index/"
LIVERTOX_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/"
LIVERTOX_ARCHIVE = "livertox_NBK547852.tar.gz"
