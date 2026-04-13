from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv

from DILIGENT.server.common import constants
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.domain.environment.bootstrap import EnvironmentBootstrapState


@lru_cache(maxsize=1)
def _bootstrap_state() -> EnvironmentBootstrapState:
    return EnvironmentBootstrapState()


_DOTENV_INJECTED_KEYS: set[str] = set()


def ensure_environment_loaded(*, force: bool = False) -> Path | None:
    state = _bootstrap_state()

    with state.lock:
        env_path = Path(constants.ENV_FILE_PATH)
        if state.bootstrapped and not force:
            return env_path if env_path.exists() else None

        previous_keys = set(os.environ.keys())
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        else:
            logger.warning(".env file not found at: %s", env_path)

        _DOTENV_INJECTED_KEYS.clear()
        _DOTENV_INJECTED_KEYS.update(set(os.environ.keys()) - previous_keys)
        state.bootstrapped = True
        return env_path if env_path.exists() else None


def initialize_environment() -> Path | None:
    return ensure_environment_loaded()


def get_dotenv_injected_keys() -> set[str]:
    return set(_DOTENV_INJECTED_KEYS)


def reset_environment_bootstrap_for_tests() -> None:
    state = _bootstrap_state()
    with state.lock:
        state.bootstrapped = False
    _DOTENV_INJECTED_KEYS.clear()
