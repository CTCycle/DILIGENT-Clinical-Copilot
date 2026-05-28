from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from common import constants
from common.utils.logger import logger
from domain.bootstrap import EnvironmentBootstrapState


@lru_cache(maxsize=1)
def _runtime_state() -> "_EnvironmentRuntimeState":
    return _EnvironmentRuntimeState()


class _EnvironmentRuntimeState:
    def __init__(self) -> None:
        self.bootstrap = EnvironmentBootstrapState()
        self.dotenv_injected_keys: set[str] = set()


def ensure_environment_loaded(*, force: bool = False) -> Path | None:
    state = _runtime_state()

    with state.bootstrap.lock:
        env_path = Path(constants.ENV_FILE_PATH)
        if state.bootstrap.bootstrapped and not force:
            return env_path if env_path.exists() else None

        previous_keys = set(os.environ.keys())
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            logger.warning(".env file not found at: %s", env_path)

        state.dotenv_injected_keys.clear()
        state.dotenv_injected_keys.update(set(os.environ.keys()) - previous_keys)
        state.bootstrap.bootstrapped = True
        return env_path if env_path.exists() else None


def initialize_environment() -> Path | None:
    return ensure_environment_loaded()


def get_dotenv_injected_keys() -> set[str]:
    return set(_runtime_state().dotenv_injected_keys)


def reset_environment_bootstrap_for_tests() -> None:
    state = _runtime_state()
    with state.bootstrap.lock:
        state.bootstrap.bootstrapped = False
    state.dotenv_injected_keys.clear()
