from __future__ import annotations

import logging

from common.utils.logger import LOG_CONFIG, configure_logging

###############################################################################
def test_file_logging_is_utf8_safe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    configure_logging()

    file_handler = None
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break
    assert file_handler is not None, "FileHandler should be configured"
    assert file_handler.encoding == "utf-8"

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
