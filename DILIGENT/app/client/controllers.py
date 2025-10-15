from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import httpx
from gradio import update as gr_update

from DILIGENT.app.api.models.providers import (
    OllamaClient,
    OllamaError,
    OllamaTimeout,
)
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import (
    AGENT_API_URL,
    CLOUD_MODEL_CHOICES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    API_BASE_URL,
)

LLM_REQUEST_TIMEOUT_SECONDS = DEFAULT_LLM_TIMEOUT_SECONDS
LLM_REQUEST_TIMEOUT_DISPLAY = (
    int(LLM_REQUEST_TIMEOUT_SECONDS)
    if float(LLM_REQUEST_TIMEOUT_SECONDS).is_integer()
    else LLM_REQUEST_TIMEOUT_SECONDS
)


# [HELPERS]
###############################################################################
def extract_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("output", "result", "text", "message", "response"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:
        return str(result)


# -----------------------------------------------------------------------------
def sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


# -----------------------------------------------------------------------------
def normalize_visit_date(
    value: datetime | date | dict[str, Any] | str | None,
) -> date | None:
    if value is None:
        return None
    if isinstance(value, dict):
        try:
            day = int(value.get("day"))
            month = int(value.get("month"))
            year = int(value.get("year"))
        except (TypeError, ValueError):
            return None
        try:
            normalized = date(year, month, day)
        except ValueError:
            return None
    elif isinstance(value, datetime):
        normalized = value.date()
    elif isinstance(value, date):
        normalized = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed_datetime = datetime.fromisoformat(stripped)
        except ValueError:
            try:
                normalized = date.fromisoformat(stripped)
            except ValueError:
                return None
        else:
            normalized = parsed_datetime.date()
    else:
        return None

    today = date.today()
    if normalized > today:
        return today
    return normalized


# -----------------------------------------------------------------------------
def normalize_visit_date_component(
    value: datetime | date | dict[str, Any] | str | None,
) -> datetime | None:
    normalized = normalize_visit_date(value)
    if normalized is None:
        return None
    return datetime.combine(normalized, datetime.min.time())


# -----------------------------------------------------------------------------
def toggle_cloud_services(
    enabled: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    ClientRuntimeConfig.set_use_cloud_services(enabled)
    provider = ClientRuntimeConfig.get_llm_provider()
    provider_update = gr_update(value=provider, interactive=enabled)
    models = CLOUD_MODEL_CHOICES.get(provider, [])
    selected_model = ClientRuntimeConfig.get_cloud_model()
    if selected_model not in models:
        selected_model = ClientRuntimeConfig.set_cloud_model(
            models[0] if models else ""
        )
    model_update = gr_update(
        value=selected_model,
        choices=models,
        interactive=enabled,
    )
    button_update = gr_update(interactive=not enabled)
    temperature_update = gr_update(
        value=ClientRuntimeConfig.get_ollama_temperature(),
        interactive=not enabled,
    )
    reasoning_update = gr_update(
        value=ClientRuntimeConfig.is_ollama_reasoning_enabled(),
        interactive=not enabled,
    )
    clinical_update = gr_update(
        value=ClientRuntimeConfig.get_clinical_model(),
        interactive=not enabled,
    )
    return (
        provider_update,
        model_update,
        button_update,
        button_update,
        temperature_update,
        reasoning_update,
        clinical_update,
    )


# -----------------------------------------------------------------------------
def set_llm_provider(provider: str) -> tuple[str, dict[str, Any]]:
    selected = ClientRuntimeConfig.set_llm_provider(provider)
    models = CLOUD_MODEL_CHOICES.get(selected, [])
    current_model = ClientRuntimeConfig.get_cloud_model()
    if current_model not in models:
        current_model = ClientRuntimeConfig.set_cloud_model(models[0] if models else "")
    model_update = gr_update(
        value=current_model,
        choices=models,
        interactive=ClientRuntimeConfig.is_cloud_enabled(),
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
async def pull_selected_models(parsing_model: str, clinical_model: str) -> str:
    models: list[str] = []
    for name in (parsing_model, clinical_model):
        if not name:
            continue
        normalized = name.strip()
        if normalized and normalized not in models:
            models.append(normalized)

    if not models:
        return "[ERROR] No models selected to pull."

    try:
        async with OllamaClient() as client:
            for model in models:
                await client.pull(model, stream=False)
    except OllamaTimeout as exc:
        return f"[ERROR] Timed out pulling models: {exc}"
    except OllamaError as exc:
        return f"[ERROR] Failed to pull models: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error while pulling models: {exc}"

    pulled = ", ".join(models)
    return f"[INFO] Models available locally: {pulled}."


# -----------------------------------------------------------------------------
async def start_ollama_client() -> str:
    if ClientRuntimeConfig.is_cloud_enabled():
        return "[INFO] Cloud provider enabled; Ollama client is disabled."

    try:
        async with OllamaClient() as client:
            status = await client.start_server()
    except OllamaTimeout as exc:
        return f"[ERROR] Timed out starting Ollama server: {exc}"
    except OllamaError as exc:
        return f"[ERROR] Failed to start Ollama server: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error while starting Ollama server: {exc}"

    if status == "already_running":
        return "[INFO] Ollama server is already running."

    return "[INFO] Ollama server started successfully."


# -----------------------------------------------------------------------------
async def preload_selected_models(parsing_model: str, clinical_model: str) -> str:
    if ClientRuntimeConfig.is_cloud_enabled():
        return "[INFO] Cloud provider enabled; skipping Ollama preload."

    parser = parsing_model.strip() if parsing_model else ""
    clinical = clinical_model.strip() if clinical_model else ""
    requested = [name for name in (parser, clinical) if name]

    if not requested:
        return "[ERROR] No models selected to preload."

    try:
        async with OllamaClient() as client:
            if not await client.is_server_online():
                return "[ERROR] Ollama server is not reachable. Start the Ollama client first."
            loaded, skipped = await client.preload_models(parser, clinical)
    except OllamaTimeout as exc:
        return f"[ERROR] Timed out while preloading models: {exc}"
    except OllamaError as exc:
        return f"[ERROR] Failed to preload models: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error while preloading models: {exc}"

    if not loaded:
        return "[ERROR] No models were preloaded."

    message = f"[INFO] Preloaded models: {', '.join(loaded)}."
    if skipped:
        message += f" [WARN] Skipped due to limited memory: {', '.join(skipped)}."
    return message


# -----------------------------------------------------------------------------
def clear_agent_fields() -> tuple[
    str,
    datetime | None,
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
    return (
        "",
        None,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        [],
        False,
        "",
    )


# trigger function to start the agent on button click. Payload is optional depending
# on the requested endpoint URL (defined through run_agent function)
# -----------------------------------------------------------------------------
async def trigger_agent(url: str, payload: dict[str, Any] | None = None) -> str:
    try:
        async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            try:
                return extract_text(resp.json())
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
        return f"[ERROR] Request timed out after {LLM_REQUEST_TIMEOUT_DISPLAY} seconds."
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error: {exc}"


# [AGENT RUNNING LOGIC]
###############################################################################
async def run_agent(
    patient_name: str | None,
    visit_date: datetime | date | dict[str, Any] | str | None,
    anamnesis: str,
    has_hepatic_diseases: bool,
    drugs: str,
    exams: str,
    alt: str,
    alt_max: str,
    alp: str,
    alp_max: str,
    symptoms: list[str],
) -> str:
    normalized_visit_date = normalize_visit_date(visit_date)

    cleaned_payload = {
        "name": sanitize_field(patient_name),
        "visit_date": (
            {
                "day": normalized_visit_date.day,
                "month": normalized_visit_date.month,
                "year": normalized_visit_date.year,
            }
            if normalized_visit_date
            else None
        ),
        "anamnesis": sanitize_field(anamnesis),
        "has_hepatic_diseases": bool(has_hepatic_diseases),
        "drugs": sanitize_field(drugs),
        "exams": sanitize_field(exams),
        "alt": sanitize_field(alt),
        "alt_max": sanitize_field(alt_max),
        "alp": sanitize_field(alp),
        "alp_max": sanitize_field(alp_max),
        "symptoms": symptoms or [],
    }

    if not any(cleaned_payload[key] for key in ("anamnesis", "drugs", "exams")):
        return "[ERROR] Please provide at least one clinical section."

    url = f"{API_BASE_URL}{AGENT_API_URL}"
    return await trigger_agent(url, cleaned_payload)
