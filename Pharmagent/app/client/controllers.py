from __future__ import annotations

import json
from typing import Any

import httpx

from Pharmagent.app.constants import (
    AGENT_API_URL,
    API_BASE_URL,
    BATCH_AGENT_API_URL,
)


###############################################################################
def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("output", "result", "text", "message", "response"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:
        return str(result)


def _sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


async def _trigger_agent(url: str, payload: dict[str, Any] | None = None) -> str:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            if payload is None:
                resp = await client.post(url)
            else:
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            try:
                return _extract_text(resp.json())
            except ValueError:
                return resp.text

    except httpx.ConnectError as exc:
        return f"[ERROR] Could not connect to backend at {url}.\nDetails: {exc}"
    except httpx.HTTPStatusError as exc:
        body = exc.response.text if exc.response is not None else ""
        code = exc.response.status_code if exc.response else "unknown"
        return (
            f"[ERROR] Backend returned status {code}."
            f"\nURL: {url}\nResponse body:\n{body}"
        )
    except httpx.TimeoutException:
        return f"[ERROR] Request timed out after {120} seconds."
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error: {exc}"


async def run_agent(
    patient_name: str | None,
    anamnesis: str,
    drugs: str,
    exams: str,
    alt: str,
    alt_max: str,
    alp: str,
    alp_max: str,
    flags: list[str],
    process_from_files: bool,
) -> str:
    if process_from_files:
        url = f"{API_BASE_URL}{BATCH_AGENT_API_URL}"
        return await _trigger_agent(url)

    cleaned_payload = {
        "name": _sanitize_field(patient_name),
        "anamnesis": _sanitize_field(anamnesis),
        "drugs": _sanitize_field(drugs),
        "exams": _sanitize_field(exams),
        "alt": _sanitize_field(alt),
        "alt_max": _sanitize_field(alt_max),
        "alp": _sanitize_field(alp),
        "alp_max": _sanitize_field(alp_max),
        "flags": flags or [],
    }

    if not any(cleaned_payload[key] for key in ("anamnesis", "drugs", "exams")):
        return "[ERROR] Please provide at least one clinical section."

    url = f"{API_BASE_URL}{AGENT_API_URL}"
    return await _trigger_agent(url, cleaned_payload)


def reset_agent_fields() -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    list[str],
    bool,
    str,
]:
    return "", "", "", "", "", "", "", "", [], False, ""
