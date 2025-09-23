from __future__ import annotations

import json
from typing import Any

import httpx
from gradio import update as gr_update

from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.constants import (
    AGENT_API_URL,
    API_BASE_URL,
    BATCH_AGENT_API_URL,
)
from Pharmagent.app.api.models.providers import (
    OllamaClient,
    OllamaError,
    OllamaTimeout,
)


_PRELOADED_MODELS: set[str] = set()


# [HELPERS]
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


# -----------------------------------------------------------------------------
def _sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


# -----------------------------------------------------------------------------
async def _release_preloaded_models() -> str:
    if not _PRELOADED_MODELS:
        return ""

    try:
        async with OllamaClient() as client:
            if not await client.is_server_online():
                return "[WARN] Ollama server is not reachable. Unable to free models."
            models = sorted(_PRELOADED_MODELS)
            released, failures = await client.unload_models(models)
    except OllamaTimeout as exc:
        return f"[ERROR] Timed out while releasing models: {exc}"
    except OllamaError as exc:
        return f"[ERROR] Failed to release models: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error while releasing models: {exc}"

    for name in released:
        _PRELOADED_MODELS.discard(name)

    if not released and not failures:
        return ""

    parts: list[str] = []
    if released:
        parts.append(f"[INFO] Released models from memory: {', '.join(released)}.")
    if failures:
        details = ", ".join(f"{name}: {error}" for name, error in failures.items())
        parts.append(f"[WARN] Some models could not be released ({details}).")
    return " ".join(parts)


# -----------------------------------------------------------------------------
def toggle_cloud_services(enabled: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ClientRuntimeConfig.set_use_cloud_services(enabled)
    provider_update = gr_update(
        value=ClientRuntimeConfig.get_llm_provider(),
        interactive=enabled,
    )
    button_update = gr_update(interactive=not enabled)
    return provider_update, button_update, button_update


# -----------------------------------------------------------------------------
def set_llm_provider(provider: str) -> str:
    return ClientRuntimeConfig.set_llm_provider(provider)


# -----------------------------------------------------------------------------
def set_parsing_model(model: str) -> str:
    return ClientRuntimeConfig.set_parsing_model(model)


# -----------------------------------------------------------------------------
def set_agent_model(model: str) -> str:
    return ClientRuntimeConfig.set_agent_model(model)


# -----------------------------------------------------------------------------
async def pull_selected_models(parsing_model: str, agent_model: str) -> str:
    models: list[str] = []
    for name in (parsing_model, agent_model):
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
async def preload_selected_models(parsing_model: str, agent_model: str) -> str:
    if ClientRuntimeConfig.is_cloud_enabled():
        return "[INFO] Cloud provider enabled; skipping Ollama preload."

    parser = parsing_model.strip() if parsing_model else ""
    agent = agent_model.strip() if agent_model else ""
    requested = [name for name in (parser, agent) if name]

    if not requested:
        return "[ERROR] No models selected to preload."

    try:
        async with OllamaClient() as client:
            if not await client.is_server_online():
                return "[ERROR] Ollama server is not reachable. Start the Ollama client first."
            loaded, skipped = await client.preload_models(parser, agent)
    except OllamaTimeout as exc:
        return f"[ERROR] Timed out while preloading models: {exc}"
    except OllamaError as exc:
        return f"[ERROR] Failed to preload models: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] Unexpected error while preloading models: {exc}"

    if not loaded:
        return "[ERROR] No models were preloaded."

    _PRELOADED_MODELS.update(loaded)

    message = f"[INFO] Preloaded models: {', '.join(loaded)}."
    if skipped:
        message += f" [WARN] Skipped due to limited memory: {', '.join(skipped)}."
    return message


# -----------------------------------------------------------------------------
async def clear_agent_fields() -> tuple[
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
    bool,
    bool,
    str,
    str,
]:
    status = await _release_preloaded_models()

    return (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        [],
        False,
        False,
        False,
        status,
    )


# trigger function to start the agent on button click. Payload is optional depending
# on the requested endpoint URL (defined through run_agent function)
# -----------------------------------------------------------------------------
async def _trigger_agent(url: str, payload: dict[str, Any] | None = None) -> str:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                json=payload
            )
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


# [AGENT RUNNING LOGIC]
###############################################################################
async def run_agent(
    patient_name: str | None,
    anamnesis: str,
    drugs: str,
    exams: str,
    alt: str,
    alt_max: str,
    alp: str,
    alp_max: str,
    symptoms: list[str],
    process_from_files: bool,
    translate_to_eng: bool,
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
        "symptoms": symptoms or [],
        "translate_to_eng": translate_to_eng,
    }

    if not any(cleaned_payload[key] for key in ("anamnesis", "drugs", "exams")):
        return "[ERROR] Please provide at least one clinical section."

    url = f"{API_BASE_URL}{AGENT_API_URL}"
    return await _trigger_agent(url, cleaned_payload)
