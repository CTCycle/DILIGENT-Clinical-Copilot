from __future__ import annotations

import json
from typing import Any

import httpx

from Pharmagent.app.constants import AGENT_API_URL, API_BASE_URL


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


async def _trigger_agent(payload: dict[str, Any]) -> str:
    url = f"{API_BASE_URL}{AGENT_API_URL}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            try:
                return _extract_text(resp.json())
            except ValueError:
                return resp.text

    except httpx.ConnectError as e:
        return f"[ERROR] Could not connect to backend at {url}.\nDetails: {e}"
    except httpx.HTTPStatusError as e:
        body = e.response.text if e.response is not None else ""
        code = e.response.status_code if e.response else "unknown"
        return f"[ERROR] Backend returned status {code}.\nURL: {url}\nResponse body:\n{body}"
    except httpx.TimeoutException:
        return f"[ERROR] Request timed out after {120} seconds."
    except Exception as e:  # noqa: BLE001
        return f"[ERROR] Unexpected error: {e}"


async def run_agent(
    patient_name: str | None,
    anamnesis: str,
    drugs: str,
    exams: str,
    alt: str,
    alp: str,
    flags: list[str],
) -> str:
    cleaned_payload = {
        "name": _sanitize_field(patient_name),
        "anamnesis": _sanitize_field(anamnesis),
        "drugs": _sanitize_field(drugs),
        "exams": _sanitize_field(exams),
        "alt": _sanitize_field(alt),
        "alp": _sanitize_field(alp),
        "flags": flags or [],
        "from_files": False,
    }

    if not any(cleaned_payload[key] for key in ("anamnesis", "drugs", "exams")):
        return "[ERROR] Please provide at least one clinical section."

    return await _trigger_agent(cleaned_payload)


async def run_agent_from_files(flags: list[str]) -> str:
    payload = {"from_files": True, "flags": flags or []}
    return await _trigger_agent(payload)


def reset_agent_fields() -> tuple[str, str, str, str, str, str, list[str], str]:
    return "", "", "", "", "", "", [], ""
