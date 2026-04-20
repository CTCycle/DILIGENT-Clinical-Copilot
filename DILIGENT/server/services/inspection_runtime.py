from __future__ import annotations

from typing import Any

from DILIGENT.server.services.runtime_overrides import RuntimeOverrides


###############################################################################
def coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


###############################################################################
def coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


###############################################################################
def coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


###############################################################################
def runtime_override_context(
    *,
    use_cloud_services: bool | None,
    llm_provider: str | None,
    cloud_model: str | None,
    text_extraction_model: str | None,
    clinical_model: str | None,
    ollama_temperature: float | None,
    cloud_temperature: float | None,
    ollama_reasoning: bool | None,
):
    return RuntimeOverrides.context(
        use_cloud_services=use_cloud_services,
        llm_provider=llm_provider,
        cloud_model=cloud_model,
        text_extraction_model=text_extraction_model,
        clinical_model=clinical_model,
        ollama_temperature=ollama_temperature,
        cloud_temperature=cloud_temperature,
        ollama_reasoning=ollama_reasoning,
    )
