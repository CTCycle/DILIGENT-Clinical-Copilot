from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, 'Pharmagent')
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'database')
DOCS_PATH = join(DATA_PATH, 'documents')
SOURCES_PATH = join(DATA_PATH, 'sources')
LOGS_PATH = join(RSC_PATH, 'logs')

# [ENDPOINS]
###############################################################################
API_BASE_URL = "http://127.0.0.1:8000"
PARSER_MODEL = "deepseek-r1:14b"





    


