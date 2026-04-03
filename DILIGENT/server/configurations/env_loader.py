from __future__ import annotations

import os

from dotenv import load_dotenv

from DILIGENT.server.common.constants import ENV_FILE_PATH
from DILIGENT.server.domain.settings.environment import EnvironmentSettings


def load_environment() -> EnvironmentSettings:
    if os.path.exists(ENV_FILE_PATH):
        load_dotenv(dotenv_path=ENV_FILE_PATH, override=False)
    return EnvironmentSettings()
