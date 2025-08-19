import asyncio
from typing import Optional, Any
import httpx


from Pharmagent.app.constants import API_BASE_URL, AGENT_API_URL



###############################################################################
def _extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("output", "result", "text", "message", "response"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val
    try:
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:
        return str(result)


async def run_agent(patient_name: Optional[str], input_text: str) -> str:
    input_text = (input_text or "").strip()
    if not input_text:
        return "[ERROR] Please provide input text."

    payload = {"name": (patient_name or None), "text": input_text}
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
    except asyncio.TimeoutError:
        return f"[ERROR] Request timed out after {120} seconds."
    except Exception as e:
        return f"[ERROR] Unexpected error: {e}"