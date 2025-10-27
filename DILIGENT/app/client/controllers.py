from __future__ import annotations

import inspect
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from DILIGENT.app.api.models.providers import (
    OllamaClient,
    OllamaError,
    OllamaTimeout,
)
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import (
    CLINICAL_API_URL,
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


###############################################################################
MISSING = object()


###############################################################################
async def invoke_progress_callback(
    callback: Callable[[int, int, str], Awaitable[None] | None] | None,
    step: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    outcome = callback(step, total, message)
    if inspect.isawaitable(outcome):
        await outcome


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
        day_raw = value.get("day")
        month_raw = value.get("month")
        year_raw = value.get("year")

        if not isinstance(day_raw, (str, int)):
            return None
        if not isinstance(month_raw, (str, int)):
            return None
        if not isinstance(year_raw, (str, int)):
            return None

        try:
            day = int(day_raw)
            month = int(month_raw)
            year = int(year_raw)
            normalized = date(year, month, day)
        except (TypeError, ValueError):
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
    ComponentUpdate,
    ComponentUpdate,
    ComponentUpdate,
    ComponentUpdate,
    ComponentUpdate,
    ComponentUpdate,
]:
    ClientRuntimeConfig.set_use_cloud_services(enabled)
    provider = ClientRuntimeConfig.get_llm_provider()
    provider_update = ComponentUpdate(value=provider, enabled=enabled)
    models = CLOUD_MODEL_CHOICES.get(provider, [])
    selected_model = ClientRuntimeConfig.get_cloud_model()
    if selected_model not in models:
        selected_model = ClientRuntimeConfig.set_cloud_model(
            models[0] if models else ""
        )
    model_update = ComponentUpdate(
        value=selected_model,
        options=models,
        enabled=enabled,
    )
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
    return (
        provider_update,
        model_update,
        button_update,
        temperature_update,
        reasoning_update,
        clinical_update,
    )


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
    list[str],
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
        [],
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
    url: str,
    payload: dict[str, Any] | None = None,
    on_progress: Callable[[int, int, str], Awaitable[None] | None] | None = None,
) -> tuple[str, ComponentUpdate]:
    try:
        async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT_SECONDS) as client:
            if on_progress is not None:
                params = {"stream": "true"}
                async with client.stream("POST", url, json=payload, params=params) as response:
                    response.raise_for_status()
                    final_message = ""
                    json_payload: dict[str, Any] | list[Any] | None = None
                    buffered_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            buffered_lines.append(line)
                            continue
                        event_type = data.get("type")
                        if event_type == "progress":
                            step_value = data.get("step", 0)
                            total_value = data.get("total", 0)
                            try:
                                step_int = int(step_value)
                            except (TypeError, ValueError):
                                step_int = 0
                            try:
                                total_int = int(total_value)
                            except (TypeError, ValueError):
                                total_int = 0
                            message_text = str(data.get("message", ""))
                            await invoke_progress_callback(
                                on_progress,
                                step_int,
                                total_int,
                                message_text,
                            )
                        elif event_type == "result":
                            final_message = str(data.get("content") or "")
                            payload_candidate = data.get("json")
                            if isinstance(payload_candidate, (dict, list)):
                                json_payload = payload_candidate
                        elif event_type == "error":
                            json_candidate = data.get("json")
                            json_payload = json_candidate if isinstance(json_candidate, (dict, list)) else None
                            error_message = str(
                                data.get(
                                    "message",
                                    "[ERROR] Unexpected error during analysis.",
                                )
                            )
                            detail_payload = data.get("detail")
                            if detail_payload:
                                if isinstance(detail_payload, str):
                                    error_message = f"{error_message}\nDetails: {detail_payload}"
                                else:
                                    try:
                                        detail_serialized = json.dumps(
                                            detail_payload,
                                            ensure_ascii=False,
                                            indent=2,
                                        )
                                    except (TypeError, ValueError):
                                        detail_serialized = str(detail_payload)
                                    error_message = f"{error_message}\nDetails: {detail_serialized}"
                            return error_message, build_json_output(json_payload)
                    if final_message:
                        return final_message, build_json_output(json_payload)
                    if buffered_lines:
                        fallback_message = "\n".join(buffered_lines).strip()
                        if fallback_message:
                            return fallback_message, build_json_output(None)
                    return "[ERROR] Empty response from backend.", build_json_output(None)
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
    symptoms: list[str],
    use_rag: bool,
    on_progress: Callable[[int, int, str], Awaitable[None] | None] | None = None,
) -> tuple[str, ComponentUpdate, ComponentUpdate]:
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
        "alt": sanitize_field(alt),
        "alt_max": sanitize_field(alt_max),
        "alp": sanitize_field(alp),
        "alp_max": sanitize_field(alp_max),
        "symptoms": symptoms or [],
        "use_rag": bool(use_rag),
    }

    if not any(cleaned_payload[key] for key in ("anamnesis", "drugs")):
        return (
            "[ERROR] Please provide at least one clinical section.",
            build_json_output(None),
            build_export_update(None),
        )

    url = f"{API_BASE_URL}{CLINICAL_API_URL}"
    message, json_update = await trigger_session(
        url,
        cleaned_payload,
        on_progress=on_progress,
    )
    normalized_message = message.strip() if message else ""
    exportable = (
        message
        if normalized_message and not normalized_message.startswith("[ERROR]")
        else None
    )
    return message, json_update, build_export_update(exportable)
