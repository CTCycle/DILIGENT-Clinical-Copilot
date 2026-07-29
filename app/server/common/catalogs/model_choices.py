from __future__ import annotations

import json
from pathlib import Path

from common.paths import CATALOGS_PATH
from common.utils.catalog_loader import CatalogLoader
from domain.llm.providers import CloudProviderDefinition

###############################################################################
def get_cloud_model_choices() -> dict[str, list[str]]:
    payload = json.loads(
        (Path(CATALOGS_PATH) / "cloud_providers.json").read_text(
            encoding="utf-8"
        )
    )
    definitions = (
        CloudProviderDefinition.model_validate(item) for item in payload["providers"]
    )
    return {item.provider_id: list(item.models) for item in definitions}

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
