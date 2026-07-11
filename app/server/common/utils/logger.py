from __future__ import annotations

import logging
import logging.config
import os
from datetime import datetime
from typing import Any

from common.paths import LOGS_PATH

_logging_configured: bool = False

###############################################################################
LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%d-%m-%Y %H:%M:%S",
        },
        "minimal": {
            "format": "[%(levelname)s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "minimal",
        },
    },
    "loggers": {
        "matplotlib": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
        "httpx": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}


###############################################################################
def configure_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_filename = str(LOGS_PATH / f"DILIGENT_{current_timestamp}_{os.getpid()}.log")

    config = {
        **LOG_CONFIG,
        "handlers": {
            **LOG_CONFIG["handlers"],
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "default",
                "filename": log_filename,
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            name: {
                **cfg,
                "handlers": ["console", "file"],
            }
            for name, cfg in LOG_CONFIG["loggers"].items()
        },
        "root": {
            **LOG_CONFIG["root"],
            "handlers": ["console", "file"],
        },
    }
    try:
        logging.config.dictConfig(config)
    except ValueError:
        logging.config.dictConfig(LOG_CONFIG)


logger = logging.getLogger()
