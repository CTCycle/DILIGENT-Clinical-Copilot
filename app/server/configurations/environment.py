from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from common import paths
from common.utils.logger import logger
from domain.bootstrap import EnvironmentBootstrapState
from configurations.runtime_bootstrap import ensure_runtime_data_layout

###############################################################################
@lru_cache(maxsize=1)
def _runtime_state() -> "_EnvironmentRuntimeState":
    return _EnvironmentRuntimeState()

###############################################################################
class _EnvironmentRuntimeState:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.bootstrap = EnvironmentBootstrapState()
        self.dotenv_injected_keys: set[str] = set()

###############################################################################
def ensure_environment_loaded(*, force: bool = False) -> Path | None:
    state = _runtime_state()

    with state.bootstrap.lock:
        env_path = paths.ENV_FILE_PATH
        if state.bootstrap.bootstrapped and not force:
            return env_path if env_path.exists() else None

        previous_keys = set(os.environ.keys())
        ensure_runtime_data_layout()
        if not env_path.exists():
            example_path = paths.ENV_EXAMPLE_PATH
            if not example_path.exists():
                raise FileNotFoundError(
                    f"Environment template not found at: {example_path}"
                )
            shutil.copyfile(example_path, env_path)
            logger.info("Created %s from %s", env_path, example_path)
        if env_path.exists():
            # Deployment and CI environment variables must be able to override
            # machine-local values committed to or supplied by the .env file.
            load_dotenv(dotenv_path=env_path, override=False)
        state.dotenv_injected_keys.clear()
        state.dotenv_injected_keys.update(set(os.environ.keys()) - previous_keys)
        state.bootstrap.bootstrapped = True
        return env_path if env_path.exists() else None

###############################################################################
def get_dotenv_injected_keys() -> set[str]:
    return set(_runtime_state().dotenv_injected_keys)

###############################################################################
def reset_environment_bootstrap_for_tests() -> None:
    state = _runtime_state()
    with state.bootstrap.lock:
        state.bootstrap.bootstrapped = False
    state.dotenv_injected_keys.clear()
