from __future__ import annotations

import logging
import logging.config
import os
from datetime import datetime
from typing import Any

from common.constants import LOGS_PATH

# Generate timestamp for the log filename
###############################################################################
os.makedirs(LOGS_PATH, exist_ok=True)
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
log_filename = os.path.join(LOGS_PATH, f"DILIGENT_{current_timestamp}_{os.getpid()}.log")

# Define logger configuration
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
        "matplotlib": {
            "level": "WARNING",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "httpx": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}


# override logger configuration and load root logger
###############################################################################
try:
    logging.config.dictConfig(LOG_CONFIG)
except ValueError:
    fallback_config = {
        **LOG_CONFIG,
        "handlers": {
            "console": LOG_CONFIG["handlers"]["console"],
        },
        "loggers": {
            name: {
                **config,
                "handlers": ["console"],
            }
            for name, config in LOG_CONFIG["loggers"].items()
        },
        "root": {
            **LOG_CONFIG["root"],
            "handlers": ["console"],
        },
    }
    logging.config.dictConfig(fallback_config)
logger = logging.getLogger()

