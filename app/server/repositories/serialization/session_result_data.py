"""Pure helpers for clinical-session payload and value serialization."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Any

import pandas as pd

from common.utils.logger import logger
from repositories import values as repository_values


###############################################################################
def decode_patient_image(value: Any) -> bytes | None:
    normalized = repository_values.normalize_string(value)
    if normalized is None:
        return None
    payload = normalized
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", maxsplit=1)[1].strip()
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error, ValueError:
        logger.warning("Skipping invalid patient image payload during session save")
        return None


###############################################################################
def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed


###############################################################################
def parse_session_result_payload(payload_json: str | None) -> dict[str, Any] | None:
    normalized_payload = repository_values.normalize_string(payload_json)
    if normalized_payload is None:
        return None
    try:
        parsed = json.loads(normalized_payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


###############################################################################
def serialize_json_payload(payload: Any) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        return repository_values.normalize_string(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError, ValueError:
        return repository_values.normalize_string(payload)
