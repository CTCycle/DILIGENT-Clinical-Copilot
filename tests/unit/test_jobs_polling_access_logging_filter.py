from __future__ import annotations

import logging

from DILIGENT.common.utils.logger import SkipJobsPollingAccessFilter


###############################################################################
def make_access_record(
    *,
    method: str,
    path: str,
    logger_name: str = "uvicorn.access",
) -> logging.LogRecord:
    return logging.LogRecord(
        name=logger_name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %s',
        args=("127.0.0.1:12345", method, path, "1.1", 200),
        exc_info=None,
    )


# -----------------------------------------------------------------------------
def test_filter_blocks_clinical_job_poll_status_access_logs() -> None:
    filter_instance = SkipJobsPollingAccessFilter()
    poll_record = make_access_record(method="GET", path="/clinical/jobs/abc123")

    assert filter_instance.filter(poll_record) is False


# -----------------------------------------------------------------------------
def test_filter_blocks_model_job_poll_status_access_logs() -> None:
    filter_instance = SkipJobsPollingAccessFilter()
    poll_record = make_access_record(method="GET", path="/models/jobs/abc123")

    assert filter_instance.filter(poll_record) is False


# -----------------------------------------------------------------------------
def test_filter_keeps_non_polling_access_logs() -> None:
    filter_instance = SkipJobsPollingAccessFilter()
    start_record = make_access_record(method="POST", path="/clinical/jobs")
    other_record = make_access_record(method="GET", path="/clinical")

    assert filter_instance.filter(start_record) is True
    assert filter_instance.filter(other_record) is True


# -----------------------------------------------------------------------------
def test_filter_keeps_non_access_logger_records() -> None:
    filter_instance = SkipJobsPollingAccessFilter()
    application_record = make_access_record(
        method="GET",
        path="/clinical/jobs/abc123",
        logger_name="DILIGENT.server.services.jobs",
    )

    assert filter_instance.filter(application_record) is True
