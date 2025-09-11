import os

from dotenv import load_dotenv

from Pharmagent.app.constants import PROJECT_DIR
from Pharmagent.app.logger import logger


# [IMPORT CUSTOM MODULES]
###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = os.path.join(PROJECT_DIR, "setup", ".env")
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path, override=True)
        else:
            logger.error(f".env file not found at: {self.env_path}")

    # -------------------------------------------------------------------------
    def get_environment_variables(self) -> dict[str, str]:
        return dict(os.environ)
