from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx

from DILIGENT.src.server.models.providers import (
    OllamaClient,
    OllamaError,
    OllamaTimeout,
)
from DILIGENT.src.packages.configurations import (
    AppConfigurations,
    LLMRuntimeConfig,
    configurations
)

from DILIGENT.src.packages.constants import (
    CLINICAL_API_URL,
    CLOUD_MODEL_CHOICES,
    REPORT_EXPORT_DIRECTORY_PREFIX,
    REPORT_EXPORT_FILENAME,
)

from DILIGENT.src.packages.utils.services.payload import (
    normalize_visit_date,
    sanitize_dili_payload,
)


# [RUNTIME SETTINGS DATACLASS]
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


# [SETTINGS]
###############################################################################
class SettingsService:
    def __init__(
        self, 
        runtime_config: type[LLMRuntimeConfig] = LLMRuntimeConfig,
        app_config: AppConfigurations = configurations
    ) -> None:
        self.runtime_config = runtime_config
        self.app_config = app_config

    # -------------------------------------------------------------------------
    def resolve_cloud_selection(
        self, provider: str | None, cloud_model: str | None
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

    # -------------------------------------------------------------------------
    def get_runtime_settings(self) -> RuntimeSettings:        
        return RuntimeSettings(
            use_cloud_services=self.runtime_config.is_cloud_enabled(),
            provider=self.runtime_config.get_llm_provider(),
            cloud_model=self.runtime_config.get_cloud_model(),
            parsing_model=self.runtime_config.get_parsing_model(),
            clinical_model=self.runtime_config.get_clinical_model(),
            temperature=self.runtime_config.get_ollama_temperature(),
            reasoning=self.runtime_config.is_ollama_reasoning_enabled(),
        )

    # -------------------------------------------------------------------------
    def reset_runtime_settings(self) -> RuntimeSettings:
        self.runtime_config.reset_defaults()
        return self.get_runtime_settings()

    # -------------------------------------------------------------------------
    def apply_runtime_settings(self, settings: RuntimeSettings) -> RuntimeSettings:
        self.runtime_config.set_use_cloud_services(settings.use_cloud_services)
        provider = self.runtime_config.set_llm_provider(settings.provider)
        self.runtime_config.set_cloud_model(settings.cloud_model)
        parsing_model = self.runtime_config.set_parsing_model(settings.parsing_model)
        clinical_model = self.runtime_config.set_clinical_model(settings.clinical_model)
        temperature = self.runtime_config.set_ollama_temperature(settings.temperature)
        reasoning = self.runtime_config.set_ollama_reasoning(settings.reasoning)
        return RuntimeSettings(
            use_cloud_services=self.runtime_config.is_cloud_enabled(),
            provider=provider,
            cloud_model=self.runtime_config.get_cloud_model(),
            parsing_model=parsing_model,
            clinical_model=clinical_model,
            temperature=temperature,
            reasoning=reasoning,
        )

    # -------------------------------------------------------------------------
    def normalize_visit_date_component(
        self, value: datetime | date | dict[str, Any] | str | None
    ) -> datetime | None:
        normalized = normalize_visit_date(value)
        if normalized is None:
            return None
        return datetime.combine(normalized, datetime.min.time())

    # -------------------------------------------------------------------------
    def clear_session_fields(self) -> dict[str, Any]:
        defaults = self.reset_runtime_settings()
        return {
            "patient_name": "",
            "visit_date": None,
            "anamnesis": "",
            "drugs": "",
            "alt": "",
            "alt_max": "",
            "alp": "",
            "alp_max": "",
            "use_rag": False,
            "message": "",
            "json": None,
            "export_path": None,
            "settings": defaults,
        }
    

# [MODEL PULL CONTROLLER]
###############################################################################
class ModelPullEndpointService:
    def __init__(self, config: AppConfigurations = configurations) -> None:
        self.config = config
        self.settings_controller = SettingsService()

    # -------------------------------------------------------------------------
    async def pull_selected_models(self, settings: RuntimeSettings) -> dict[str, Any]:
        normalized_settings = self.settings_controller.apply_runtime_settings(settings)
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


# [DILI SESSION CONTROLLER]
###############################################################################
class DILISessionEndpointService:
    def __init__(self, config: AppConfigurations = configurations) -> None:
        self.config = config
        self.settings_controller = SettingsService()
        
    # -------------------------------------------------------------------------
    def extract_text(self, result: Any) -> str:
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

    # -------------------------------------------------------------------------
    def create_export_file(self, content: str) -> str:
        directory = tempfile.mkdtemp(prefix=REPORT_EXPORT_DIRECTORY_PREFIX)
        file_path = os.path.join(directory, REPORT_EXPORT_FILENAME)
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return file_path

    # -------------------------------------------------------------------------
    async def trigger_session(
        self, url: str, payload: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | list[Any] | None]:
        try:
            async with httpx.AsyncClient(timeout=self.config.client.ui.http_timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                json_payload: dict[str, Any] | list[Any] | None = None
                try:
                    parsed = resp.json()
                except ValueError:
                    message = resp.text
                else:
                    message = self.extract_text(parsed)
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
                    body_content = self.extract_text(parsed)
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
                f"[ERROR] Request timed out after {self.config.client.ui.http_timeout} seconds.",
                None,
            )
        except Exception as exc:  # noqa: BLE001
            return f"[ERROR] Unexpected error: {exc}", None

    # -------------------------------------------------------------------------
    async def run_DILI_session(
        self,
        patient_name: str | None,
        visit_date: datetime | date | dict[str, Any] | str | None,
        anamnesis: str,
        drugs: str,
        alt: str,
        alt_max: str,
        alp: str,
        alp_max: str,
        use_rag: bool,
        settings: RuntimeSettings,
    ) -> dict[str, Any]:
        self.settings_controller.apply_runtime_settings(settings)
        cleaned_payload = sanitize_dili_payload(
            patient_name=patient_name,
            visit_date=visit_date,
            anamnesis=anamnesis,
            drugs=drugs,
            alt=alt,
            alt_max=alt_max,
            alp=alp,
            alp_max=alp_max,
            use_rag=use_rag,
        )

        url = f"{self.config.client.ui.api_base_url}{CLINICAL_API_URL}"
        message, json_payload = await self.trigger_session(url, cleaned_payload)
        normalized_message = message.strip() if message else ""
        export_path = None
        if normalized_message and not normalized_message.startswith("[ERROR]"):
            export_path = self.create_export_file(message)
        return {"message": message, "json": json_payload, "export_path": export_path}

