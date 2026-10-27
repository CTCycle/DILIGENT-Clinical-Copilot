# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import logging
import logging.config
import os
from datetime import datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from typing import Any

from common.paths import LOGS_PATH


class HttpxRequestFilter(logging.Filter):
    """Keep unsuccessful HTTP responses visible and successful ones at debug."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg == 'HTTP Request: %s %s "%s %d %s"' and len(record.args) == 5:
            status = record.args[3]
            level = (
                logging.WARNING
                if isinstance(status, int) and status >= 400
                else logging.DEBUG
            )
            record.levelno = level
            record.levelname = logging.getLevelName(level)
        return True


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
    "filters": {"httpx_requests": {"()": HttpxRequestFilter}},
    "loggers": {
        "matplotlib": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
        "httpx": {
            "level": "DEBUG",
            "handlers": ["console"],
            "filters": ["httpx_requests"],
            "propagate": False,
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}


###############################################################################
@lru_cache(maxsize=1)
def configure_logging() -> None:

    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    debug = os.getenv("DILIGENT_LOG_LEVEL", "INFO").strip().upper() == "DEBUG"
    # Application middleware owns request diagnostics, including failed requests.
    logging.getLogger("uvicorn.access").disabled = True
    if os.getenv("DILIGENT_DESKTOP", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=[
                RotatingFileHandler(
                    LOGS_PATH / "desktop-backend.log",
                    maxBytes=5 * 1024 * 1024,
                    backupCount=2,
                    encoding="utf-8",
                )
            ],
            force=True,
        )
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.DEBUG)
        httpx_logger.addFilter(HttpxRequestFilter())
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG if debug else logging.INFO)
        return

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_filename = str(LOGS_PATH / f"DILIGENT_{current_timestamp}_{os.getpid()}.log")

    config = {
        **LOG_CONFIG,
        "handlers": {
            **LOG_CONFIG["handlers"],
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG" if debug else "INFO",
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
    if debug:
        config["handlers"]["console"] = {
            **config["handlers"]["console"],
            "level": "DEBUG",
        }
    try:
        logging.config.dictConfig(config)
    except ValueError:
        logging.config.dictConfig(LOG_CONFIG)
    logging.getLogger("uvicorn.access").disabled = True


logger = logging.getLogger()
