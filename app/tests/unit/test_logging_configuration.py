from __future__ import annotations

import logging

from common.utils.logger import LOG_CONFIG


def test_file_logging_is_utf8_safe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert LOG_CONFIG["handlers"]["file"]["encoding"] == "utf-8"

    log_path = tmp_path / "unicode.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    try:
        logger = logging.getLogger("diligent-unicode-smoke")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.info("Unicode smoke: → ≤")
    finally:
        handler.close()
        logger.handlers = []

    assert "→ ≤" in log_path.read_text(encoding="utf-8")
