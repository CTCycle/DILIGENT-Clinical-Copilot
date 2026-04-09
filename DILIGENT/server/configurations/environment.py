from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from DILIGENT.server.common.constants import ENV_FILE_PATH

from DILIGENT.server.configurations.bootstrap import (
    ensure_environment_loaded,
    reset_environment_bootstrap_for_tests,
)

_ENV_BOOTSTRAP_LOCK = Lock()
_ENV_BOOTSTRAPPED = False
_DOTENV_INJECTED_KEYS: set[str] = set()

###############################################################################
def initialize_environment() -> Path | None:
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return Path(ENV_FILE_PATH) if Path(ENV_FILE_PATH).exists() else None
    with _ENV_BOOTSTRAP_LOCK:
        if _ENV_BOOTSTRAPPED:
            return Path(ENV_FILE_PATH) if Path(ENV_FILE_PATH).exists() else None
        existing_keys = set(os.environ.keys())
        dotenv_path = Path(ENV_FILE_PATH)
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=True)
        _DOTENV_INJECTED_KEYS.clear()
        _DOTENV_INJECTED_KEYS.update(set(os.environ.keys()) - existing_keys)
        _ENV_BOOTSTRAPPED = True
    ensure_environment_loaded()
    return dotenv_path if dotenv_path.exists() else None

###############################################################################
def get_dotenv_injected_keys() -> set[str]:
    return set(_DOTENV_INJECTED_KEYS)


__all__ = [
    "ensure_environment_loaded",
    "get_dotenv_injected_keys",
    "initialize_environment",
    "reset_environment_bootstrap_for_tests",
]
