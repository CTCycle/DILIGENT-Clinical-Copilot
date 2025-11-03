from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from DILIGENT.app.utils.services.payload import (
    normalize_visit_date,
    sanitize_dili_payload,
)
from DILIGENT.app.api.models.providers import (
    OllamaClient,
    OllamaError,
    OllamaTimeout,
)
from DILIGENT.app.configurations import (
    ClientRuntimeConfig,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)
from DILIGENT.app.constants import (
    CLINICAL_API_URL,
    CLOUD_MODEL_CHOICES,
    API_BASE_URL,
)

LLM_REQUEST_TIMEOUT_SECONDS = DEFAULT_LLM_TIMEOUT_SECONDS


###############################################################################
@dataclass
class RuntimeSettings:
    use_cloud_services: bool
    provider: str
    cloud_model: str
    parsing_model: str
    clinical_model: str
    temperature: float | None
    reasoning: bool

# [HELPERS]
###############################################################################
def extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("output", "result", "text", "message", "response"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val
        try:
            formatted = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            return str(result)
        return f"```json\n{formatted}\n```"
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        try:
            formatted = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            return str(result)
        return f"```json\n{formatted}\n```"
    try:
        formatted = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001
        return str(result)
    return f"```json\n{formatted}\n```"

# -----------------------------------------------------------------------------
def create_export_file(content: str) -> str:
    directory = tempfile.mkdtemp(prefix="diligent_report_")
    file_path = os.path.join(directory, "clinical_report.md")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return file_path

# -----------------------------------------------------------------------------
def normalize_visit_date_component(
    value: datetime | date | dict[str, Any] | str | None,
) -> datetime | None:
    normalized = normalize_visit_date(value)
    if normalized is None:
        return None
    return datetime.combine(normalized, datetime.min.time())

# [LLM CLIENT CONTROLLERS]
###############################################################################
def resolve_cloud_selection(
    provider: str | None, cloud_model: str | None
) -> dict[str, Any]:
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider not in CLOUD_MODEL_CHOICES:
        normalized_provider = next(iter(CLOUD_MODEL_CHOICES), "")
    models = CLOUD_MODEL_CHOICES.get(normalized_provider, [])
    normalized_model = (cloud_model or "").strip()
    if normalized_model not in models:
        normalized_model = models[0] if models else ""
    return {
        "provider": normalized_provider,
        "models": models,
        "model": normalized_model or None,
    }

# -----------------------------------------------------------------------------
def get_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        use_cloud_services=ClientRuntimeConfig.is_cloud_enabled(),
        provider=ClientRuntimeConfig.get_llm_provider(),
        cloud_model=ClientRuntimeConfig.get_cloud_model(),
        parsing_model=ClientRuntimeConfig.get_parsing_model(),
        clinical_model=ClientRuntimeConfig.get_clinical_model(),
        temperature=ClientRuntimeConfig.get_ollama_temperature(),
        reasoning=ClientRuntimeConfig.is_ollama_reasoning_enabled(),
    )

# -----------------------------------------------------------------------------
def reset_runtime_settings() -> RuntimeSettings:
    ClientRuntimeConfig.reset_defaults()
    return get_runtime_settings()

# -----------------------------------------------------------------------------
def apply_runtime_settings(settings: RuntimeSettings) -> RuntimeSettings:
    ClientRuntimeConfig.set_use_cloud_services(settings.use_cloud_services)
    provider = ClientRuntimeConfig.set_llm_provider(settings.provider)
    ClientRuntimeConfig.set_cloud_model(settings.cloud_model)
    parsing_model = ClientRuntimeConfig.set_parsing_model(settings.parsing_model)
    clinical_model = ClientRuntimeConfig.set_clinical_model(settings.clinical_model)
    temperature = ClientRuntimeConfig.set_ollama_temperature(settings.temperature)
    reasoning = ClientRuntimeConfig.set_ollama_reasoning(settings.reasoning)
    return RuntimeSettings(
        use_cloud_services=ClientRuntimeConfig.is_cloud_enabled(),
        provider=provider,
        cloud_model=ClientRuntimeConfig.get_cloud_model(),
        parsing_model=parsing_model,
        clinical_model=clinical_model,
        temperature=temperature,
        reasoning=reasoning,
    )

# -----------------------------------------------------------------------------
def sync_cloud_model_options(
    provider: str | None, current_model: str | None
) -> dict[str, Any]:
    return resolve_cloud_selection(provider, current_model)


# EVENTS
###############################################################################
async def pull_selected_models(
    settings: RuntimeSettings,
) -> dict[str, Any]:
    normalized_settings = apply_runtime_settings(settings)
    models: list[str] = []
    for name in (
        normalized_settings.parsing_model,
        normalized_settings.clinical_model,
    ):
        if not name:
            continue
        candidate = name.strip()
        if candidate and candidate not in models:
            models.append(candidate)

    if not models:
        return {"message": "[ERROR] No models selected to pull.", "json": None}

    try:
        async with OllamaClient() as client:
            for model in models:
                await client.pull(model, stream=False)
    except OllamaTimeout as exc:
        return {
            "message": f"[ERROR] Timed out pulling models: {exc}",
            "json": None,
        }
    except OllamaError as exc:
        return {
            "message": f"[ERROR] Failed to pull models: {exc}",
            "json": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "message": f"[ERROR] Unexpected error while pulling models: {exc}",
            "json": None,
        }

    pulled = ", ".join(models)
    return {
        "message": f"[INFO] Models available locally: {pulled}.",
        "json": None,
    }

# -----------------------------------------------------------------------------
def clear_session_fields() -> dict[str, Any]:
    defaults = reset_runtime_settings()
    return {
        "patient_name": "",
        "visit_date": None,
        "anamnesis": "",
        "drugs": "",
        "alt": "",
        "alt_max": "",
        "alp": "",
        "alp_max": "",
        "has_diseases": False,
        "use_rag": False,
        "message": "",
        "json": None,
        "export_path": None,
        "settings": defaults,
    }

###############################################################################
# trigger function to start the agent on button click. 
###############################################################################
async def trigger_session(
    url: str, payload: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any] | list[Any] | None]:
    try:
        async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            json_payload: dict[str, Any] | list[Any] | None = None
            try:
                parsed = resp.json()
            except ValueError:
                message = resp.text
            else:
                message = extract_text(parsed)
                if isinstance(parsed, (dict, list)):
                    json_payload = parsed
            return message, json_payload

    except httpx.ConnectError as exc:
        return (
            f"[ERROR] Could not connect to backend at {url}.\nDetails: {exc}",
            None,
        )
    except httpx.HTTPStatusError as exc:
        response = exc.response
        code = response.status_code if response else "unknown"
        json_payload: dict[str, Any] | list[Any] | None = None
        body_content = ""
        if response is not None:
            try:
                parsed = response.json()
            except ValueError:
                body_content = response.text or ""
            else:
                body_content = extract_text(parsed)
                if isinstance(parsed, (dict, list)):
                    json_payload = parsed
        message = f"[ERROR] Backend returned status {code}."
        if body_content:
            message = f"{message}\n{body_content}"
        elif response is not None and response.text:
            message = f"{message}\nURL: {url}\nResponse body:\n{response.text}"
        return message, json_payload
    except httpx.TimeoutException:
        return (
            f"[ERROR] Request timed out after {LLM_REQUEST_TIMEOUT_SECONDS} seconds.",
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error: {exc}", None

# -----------------------------------------------------------------------------
async def run_DILI_session(
    patient_name: str | None,
    visit_date: datetime | date | dict[str, Any] | str | None,
    anamnesis: str,
    has_hepatic_diseases: bool,
    drugs: str,
    alt: str,
    alt_max: str,
    alp: str,
    alp_max: str,
    use_rag: bool,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    apply_runtime_settings(settings)
    cleaned_payload = sanitize_dili_payload(
        patient_name=patient_name,
        visit_date=visit_date,
        anamnesis=anamnesis,
        has_hepatic_diseases=has_hepatic_diseases,
        drugs=drugs,
        alt=alt,
        alt_max=alt_max,
        alp=alp,
        alp_max=alp_max,
        use_rag=use_rag,
    )

    url = f"{API_BASE_URL}{CLINICAL_API_URL}"
    message, json_payload = await trigger_session(url, cleaned_payload)
    normalized_message = message.strip() if message else ""
    export_path = None
    if normalized_message and not normalized_message.startswith("[ERROR]"):
        export_path = create_export_file(message)
    return {"message": message, "json": json_payload, "export_path": export_path}
