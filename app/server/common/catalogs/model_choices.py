from __future__ import annotations

from common.utils.catalog_loader import CatalogLoader


###############################################################################
def get_cloud_model_choices() -> dict[str, list[str]]:
    return {
        "openai": CatalogLoader.get_string_list("llm_models.json", "openai_cloud_models"),
        "gemini": CatalogLoader.get_string_list("llm_models.json", "gemini_cloud_models"),
    }


###############################################################################
def get_text_extraction_model_choices() -> list[str]:
    return CatalogLoader.get_string_list("llm_models.json", "text_extraction_model_choices")


###############################################################################
def get_clinical_model_choices() -> list[str]:
    return CatalogLoader.get_string_list("llm_models.json", "clinical_model_choices")
