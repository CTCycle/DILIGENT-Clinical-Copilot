from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status
import httpx

from DILIGENT.app.utils.services.payload import (
    normalize_visit_date, 
    sanitize_dili_payload
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
LLM_REQUEST_TIMEOUT_DISPLAY = (
    int(LLM_REQUEST_TIMEOUT_SECONDS)
    if float(LLM_REQUEST_TIMEOUT_SECONDS).is_integer()
    else LLM_REQUEST_TIMEOUT_SECONDS
)


###############################################################################
MISSING = object()


###############################################################################
@dataclass
class ComponentUpdate:
    value: Any = MISSING
    options: list[Any] | None = None
    enabled: bool | None = None
    visible: bool | None = None
    download_path: str | None = None


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
def build_json_output(payload: dict[str, Any] | list[Any] | None) -> ComponentUpdate:
    if payload is None:
        return ComponentUpdate(value=None, visible=False)
    return ComponentUpdate(value=payload, visible=True)

# -----------------------------------------------------------------------------
def create_export_file(content: str) -> str:
    directory = tempfile.mkdtemp(prefix="diligent_report_")
    file_path = os.path.join(directory, "clinical_report.md")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return file_path

# -----------------------------------------------------------------------------
def build_export_update(content: str | None) -> ComponentUpdate:
    if content:
        file_path = create_export_file(content)
        return ComponentUpdate(value=file_path, enabled=True, download_path=file_path)
    return ComponentUpdate(value=None, enabled=False, download_path=None)

# -----------------------------------------------------------------------------
def normalize_visit_date_component(
    value: datetime | date | dict[str, Any] | str | None,
) -> datetime | None:
    normalized = normalize_visit_date(value)
    if normalized is None:
        return None
    return datetime.combine(normalized, datetime.min.time())

# -----------------------------------------------------------------------------
def toggle_cloud_services(enabled: bool) -> dict[str, ComponentUpdate]:
    ClientRuntimeConfig.set_use_cloud_services(enabled)

    provider = ClientRuntimeConfig.get_llm_provider()
    provider_update = ComponentUpdate(value=provider, enabled=enabled)

    models = CLOUD_MODEL_CHOICES.get(provider, [])
    selected_model = ClientRuntimeConfig.get_cloud_model()
    if selected_model not in models:
        selected_model = ClientRuntimeConfig.set_cloud_model(models[0] if models else "")

    model_update = ComponentUpdate(value=selected_model, options=models, enabled=enabled)
    button_update = ComponentUpdate(enabled=not enabled)
    temperature_update = ComponentUpdate(
        value=ClientRuntimeConfig.get_ollama_temperature(),
        enabled=not enabled,
    )
    reasoning_update = ComponentUpdate(
        value=ClientRuntimeConfig.is_ollama_reasoning_enabled(),
        enabled=not enabled,
    )
    clinical_update = ComponentUpdate(
        value=ClientRuntimeConfig.get_clinical_model(),
        enabled=not enabled,
    )

    return {
        "provider": provider_update,
        "model": model_update,
        "button": button_update,
        "temperature": temperature_update,
        "reasoning": reasoning_update,
        "clinical": clinical_update,
    }

# -----------------------------------------------------------------------------
def set_llm_provider(provider: str) -> tuple[str, ComponentUpdate]:
    selected = ClientRuntimeConfig.set_llm_provider(provider)
    models = CLOUD_MODEL_CHOICES.get(selected, [])
    current_model = ClientRuntimeConfig.get_cloud_model()
    if current_model not in models:
        current_model = ClientRuntimeConfig.set_cloud_model(models[0] if models else "")
    model_update = ComponentUpdate(
        value=current_model,
        options=models,
        enabled=ClientRuntimeConfig.is_cloud_enabled(),
    )
    return selected, model_update

# -----------------------------------------------------------------------------
def set_cloud_model(model: str) -> str:
    return ClientRuntimeConfig.set_cloud_model(model)

# -----------------------------------------------------------------------------
def set_parsing_model(model: str) -> str:
    return ClientRuntimeConfig.set_parsing_model(model)

# -----------------------------------------------------------------------------
def set_clinical_model(model: str) -> str:
    return ClientRuntimeConfig.set_clinical_model(model)

# -----------------------------------------------------------------------------
def set_ollama_temperature(value: float | None) -> float:
    return ClientRuntimeConfig.set_ollama_temperature(value)

# -----------------------------------------------------------------------------
def set_ollama_reasoning(enabled: bool) -> bool:
    return ClientRuntimeConfig.set_ollama_reasoning(enabled)

# -----------------------------------------------------------------------------
async def pull_selected_models(
    parsing_model: str, clinical_model: str
) -> tuple[str, ComponentUpdate]:
    models: list[str] = []
    for name in (parsing_model, clinical_model):
        if not name:
            continue
        normalized = name.strip()
        if normalized and normalized not in models:
            models.append(normalized)

    if not models:
        return "[ERROR] No models selected to pull.", build_json_output(None)

    try:
        async with OllamaClient() as client:
            for model in models:
                await client.pull(model, stream=False)
    except OllamaTimeout as exc:
        return f"[ERROR] Timed out pulling models: {exc}", build_json_output(None)
    except OllamaError as exc:
        return f"[ERROR] Failed to pull models: {exc}", build_json_output(None)
    except Exception as exc:  # noqa: BLE001
        return (
            f"[ERROR] Unexpected error while pulling models: {exc}",
            build_json_output(None),
        )

    pulled = ", ".join(models)
    return f"[INFO] Models available locally: {pulled}.", build_json_output(None)

# -----------------------------------------------------------------------------
def clear_session_fields() -> tuple[
    str,
    datetime | None,
    str,
    str,
    str,
    str,
    str,
    str,
    bool,
    bool,
    str,
    ComponentUpdate,
    ComponentUpdate,
]:
    return (
        "",
        None,
        "",
        "",
        "",
        "",
        "",
        "",
        False,
        False,
        "",
        build_json_output(None),
        build_export_update(None),
    )


# trigger function to start the agent on button click. Payload is optional depending
# on the requested endpoint URL (defined through run_DILI_session function)
# -----------------------------------------------------------------------------
async def trigger_session(
    url: str, payload: dict[str, Any] | None = None
) -> tuple[str, ComponentUpdate]:
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
            return message, build_json_output(json_payload)

    except httpx.ConnectError as exc:
        return (
            f"[ERROR] Could not connect to backend at {url}.\nDetails: {exc}",
            build_json_output(None),
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
        return message, build_json_output(json_payload)
    except httpx.TimeoutException:
        return (
            f"[ERROR] Request timed out after {LLM_REQUEST_TIMEOUT_DISPLAY} seconds.",
            build_json_output(None),
        )
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error: {exc}", build_json_output(None)


# [AGENT RUNNING LOGIC]
###############################################################################
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
) -> tuple[str, ComponentUpdate, ComponentUpdate]:
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
    message, json_update = await trigger_session(url, cleaned_payload)
    normalized_message = message.strip() if message else ""
    exportable = (
        message
        if normalized_message and not normalized_message.startswith("[ERROR]")
        else None
    )
    return message, json_update, build_export_update(exportable)
