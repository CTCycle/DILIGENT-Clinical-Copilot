from __future__ import annotations

from common.utils.catalog_loader import CatalogLoader
from services.llm.provider_registry import provider_registry


###############################################################################
def get_cloud_model_choices() -> dict[str, list[str]]:
    return {item.provider_id: list(item.models) for item in provider_registry.all()}


###############################################################################
def get_text_extraction_model_choices() -> list[str]:
    return CatalogLoader.get_string_list(
        "llm_models.json", "text_extraction_model_choices"
    )


###############################################################################
def get_clinical_model_choices() -> list[str]:
    return CatalogLoader.get_string_list("llm_models.json", "clinical_model_choices")


###############################################################################
def get_all_cloud_model_names() -> set[str]:
    return {
        model_name
        for values in get_cloud_model_choices().values()
        for model_name in values
    }
