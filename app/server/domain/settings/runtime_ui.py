# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.settings.configuration import NCBIContactEmail

RuntimeSettingsCategory = Literal[
    "general", "data", "integrations", "matching", "advanced"
]

###############################################################################
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

###############################################################################
class _PatchModel(_StrictModel):

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_patch(self) -> "_PatchModel":
        if not self.model_fields_set:
            raise ValueError("At least one setting must be provided.")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null.")
        return self

###############################################################################
class GeneralRuntimeSettings(_StrictModel):
    polling_interval: float = Field(gt=0.0, le=60.0)

###############################################################################
class GeneralRuntimeSettingsPatch(_PatchModel):
    polling_interval: float | None = Field(default=None, gt=0.0, le=60.0)

###############################################################################
class DataRuntimeSettings(_StrictModel):
    drug_name_min_length: int = Field(ge=1, le=200)
    drug_name_max_length: int = Field(ge=1, le=1000)
    drug_name_max_tokens: int = Field(ge=1, le=64)
    text_extraction_batch_size: int = Field(ge=1, le=128)
    text_extraction_max_concurrency: int = Field(ge=1, le=64)
    retrieval_batch_size: int = Field(ge=1, le=128)
    retrieval_max_concurrency: int = Field(ge=1, le=64)
    clinical_assessment_batch_size: int = Field(ge=1, le=128)
    clinical_assessment_max_concurrency: int = Field(ge=1, le=64)
    min_best_score: float = Field(ge=0.0, le=100.0)
    high_confidence_min_score: float = Field(ge=0.0, le=100.0)
    high_confidence_min_margin: float = Field(ge=0.0, le=100.0)
    moderate_confidence_min_score: float = Field(ge=0.0, le=100.0)
    moderate_confidence_min_margin: float = Field(ge=0.0, le=100.0)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_data(self) -> "DataRuntimeSettings":
        if self.drug_name_max_length < self.drug_name_min_length:
            raise ValueError(
                "Drug-name maximum length cannot be smaller than minimum length."
            )
        if self.high_confidence_min_score < self.moderate_confidence_min_score:
            raise ValueError(
                "High-confidence score cannot be smaller than moderate-confidence score."
            )
        if self.high_confidence_min_margin < self.moderate_confidence_min_margin:
            raise ValueError(
                "High-confidence margin cannot be smaller than moderate-confidence margin."
            )
        return self

###############################################################################
class DataRuntimeSettingsPatch(_PatchModel):
    drug_name_min_length: int | None = Field(default=None, ge=1, le=200)
    drug_name_max_length: int | None = Field(default=None, ge=1, le=1000)
    drug_name_max_tokens: int | None = Field(default=None, ge=1, le=64)
    text_extraction_batch_size: int | None = Field(default=None, ge=1, le=128)
    text_extraction_max_concurrency: int | None = Field(default=None, ge=1, le=64)
    retrieval_batch_size: int | None = Field(default=None, ge=1, le=128)
    retrieval_max_concurrency: int | None = Field(default=None, ge=1, le=64)
    clinical_assessment_batch_size: int | None = Field(default=None, ge=1, le=128)
    clinical_assessment_max_concurrency: int | None = Field(default=None, ge=1, le=64)
    min_best_score: float | None = Field(default=None, ge=0.0, le=100.0)
    high_confidence_min_score: float | None = Field(default=None, ge=0.0, le=100.0)
    high_confidence_min_margin: float | None = Field(default=None, ge=0.0, le=100.0)
    moderate_confidence_min_score: float | None = Field(default=None, ge=0.0, le=100.0)
    moderate_confidence_min_margin: float | None = Field(default=None, ge=0.0, le=100.0)

###############################################################################
class IntegrationRuntimeSettings(_StrictModel):
    ncbi_contact_email: NCBIContactEmail
    livertox_download_timeout: float = Field(gt=0.0, le=3600.0)
    livertox_archive: str = Field(min_length=1, max_length=255)
    livertox_yield_interval: int = Field(ge=1, le=10000)
    livertox_skip_deterministic_ratio: float = Field(ge=0.0, le=1.0)
    livertox_monograph_max_workers: int = Field(ge=1, le=64)
    rxnav_request_timeout: float = Field(gt=0.0, le=120.0)
    terminology_download_timeout: float = Field(ge=10.0, le=3600.0)
    drugcentral_enabled: bool
    drugcentral_host: str = Field(min_length=1, max_length=255)
    drugcentral_port: int = Field(ge=1, le=65535)
    drugcentral_database: str = Field(min_length=1, max_length=128)
    drugcentral_user: str = Field(min_length=1, max_length=128)
    chembl_enabled: bool
    chembl_max_phase: int = Field(ge=1, le=4)
    swissmedic_enabled: bool

###############################################################################
class IntegrationRuntimeSettingsPatch(_PatchModel):
    ncbi_contact_email: NCBIContactEmail | None = None
    livertox_download_timeout: float | None = Field(default=None, gt=0.0, le=3600.0)
    livertox_archive: str | None = Field(default=None, min_length=1, max_length=255)
    livertox_yield_interval: int | None = Field(default=None, ge=1, le=10000)
    livertox_skip_deterministic_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    livertox_monograph_max_workers: int | None = Field(default=None, ge=1, le=64)
    rxnav_request_timeout: float | None = Field(default=None, gt=0.0, le=120.0)
    terminology_download_timeout: float | None = Field(default=None, ge=10.0, le=3600.0)
    drugcentral_enabled: bool | None = None
    drugcentral_host: str | None = Field(default=None, min_length=1, max_length=255)
    drugcentral_port: int | None = Field(default=None, ge=1, le=65535)
    drugcentral_database: str | None = Field(default=None, min_length=1, max_length=128)
    drugcentral_user: str | None = Field(default=None, min_length=1, max_length=128)
    chembl_enabled: bool | None = None
    chembl_max_phase: int | None = Field(default=None, ge=1, le=4)
    swissmedic_enabled: bool | None = None

###############################################################################
class MatchingRuntimeSettings(_StrictModel):
    resolution_cache_limit: int = Field(ge=1, le=2_000_000)
    fuzzy_min_query_length: int = Field(ge=3, le=32)
    fuzzy_shortlist_size: int = Field(ge=5, le=500)
    fuzzy_short_name_length: int = Field(ge=3, le=64)
    fuzzy_short_max_distance: int = Field(ge=0, le=4)
    fuzzy_long_max_distance: int = Field(ge=0, le=4)
    fuzzy_min_margin: int = Field(ge=1, le=4)
    external_fallback_enabled: bool
    external_max_requests_per_name: int = Field(ge=1, le=10)
    external_negative_cache_days: int = Field(ge=1, le=365)

###############################################################################
class MatchingRuntimeSettingsPatch(_PatchModel):
    resolution_cache_limit: int | None = Field(default=None, ge=1, le=2_000_000)
    fuzzy_min_query_length: int | None = Field(default=None, ge=3, le=32)
    fuzzy_shortlist_size: int | None = Field(default=None, ge=5, le=500)
    fuzzy_short_name_length: int | None = Field(default=None, ge=3, le=64)
    fuzzy_short_max_distance: int | None = Field(default=None, ge=0, le=4)
    fuzzy_long_max_distance: int | None = Field(default=None, ge=0, le=4)
    fuzzy_min_margin: int | None = Field(default=None, ge=1, le=4)
    external_fallback_enabled: bool | None = None
    external_max_requests_per_name: int | None = Field(default=None, ge=1, le=10)
    external_negative_cache_days: int | None = Field(default=None, ge=1, le=365)

###############################################################################
class AdvancedRuntimeSettings(_StrictModel):
    default_llm_timeout: float = Field(gt=0.0, le=86400.0)
    parser_llm_timeout: float = Field(gt=0.0, le=86400.0)
    disease_llm_timeout: float = Field(gt=0.0, le=86400.0)
    clinical_llm_timeout: float = Field(gt=0.0, le=86400.0)
    livertox_llm_timeout: float = Field(gt=0.0, le=86400.0)
    minimum_llm_timeout: float = Field(gt=0.0, le=3600.0)
    cloud_llm_timeout_cap: float = Field(gt=0.0, le=86400.0)
    local_llm_timeout_cap: float = Field(gt=0.0, le=86400.0)
    ollama_server_start_timeout: float = Field(gt=0.0, le=3600.0)
    max_excerpt_length: int = Field(ge=500, le=100000)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_timeout_caps(self) -> "AdvancedRuntimeSettings":
        if self.cloud_llm_timeout_cap < self.minimum_llm_timeout:
            raise ValueError(
                "Cloud LLM timeout cap cannot be smaller than the minimum timeout."
            )
        if self.local_llm_timeout_cap < self.minimum_llm_timeout:
            raise ValueError(
                "Local LLM timeout cap cannot be smaller than the minimum timeout."
            )
        return self

###############################################################################
class AdvancedRuntimeSettingsPatch(_PatchModel):
    default_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    parser_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    disease_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    clinical_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    livertox_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    minimum_llm_timeout: float | None = Field(default=None, gt=0.0, le=3600.0)
    cloud_llm_timeout_cap: float | None = Field(default=None, gt=0.0, le=86400.0)
    local_llm_timeout_cap: float | None = Field(default=None, gt=0.0, le=86400.0)
    ollama_server_start_timeout: float | None = Field(default=None, gt=0.0, le=3600.0)
    max_excerpt_length: int | None = Field(default=None, ge=500, le=100000)

###############################################################################
class RuntimeSettingsValues(_StrictModel):
    general: GeneralRuntimeSettings
    data: DataRuntimeSettings
    integrations: IntegrationRuntimeSettings
    matching: MatchingRuntimeSettings
    advanced: AdvancedRuntimeSettings

###############################################################################
class RuntimeSettingsUpdateRequest(_StrictModel):
    general: GeneralRuntimeSettingsPatch | None = None
    data: DataRuntimeSettingsPatch | None = None
    integrations: IntegrationRuntimeSettingsPatch | None = None
    matching: MatchingRuntimeSettingsPatch | None = None
    advanced: AdvancedRuntimeSettingsPatch | None = None

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_update(self) -> "RuntimeSettingsUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("At least one settings category must be provided.")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null.")
        return self

###############################################################################
class RuntimeSettingsStateResponse(_StrictModel):
    values: RuntimeSettingsValues
    defaults: RuntimeSettingsValues
    source: Literal["database"] = "database"
    environment_editable: Literal[False] = False
    updated_at: datetime | None = None


RUNTIME_SETTINGS_DEFAULTS = RuntimeSettingsValues(
    general=GeneralRuntimeSettings(polling_interval=1.0),
    data=DataRuntimeSettings(
        drug_name_min_length=3,
        drug_name_max_length=200,
        drug_name_max_tokens=8,
        text_extraction_batch_size=4,
        text_extraction_max_concurrency=2,
        retrieval_batch_size=8,
        retrieval_max_concurrency=4,
        clinical_assessment_batch_size=2,
        clinical_assessment_max_concurrency=2,
        min_best_score=2.0,
        high_confidence_min_score=8.0,
        high_confidence_min_margin=3.0,
        moderate_confidence_min_score=4.0,
        moderate_confidence_min_margin=1.0,
    ),
    integrations=IntegrationRuntimeSettings(
        ncbi_contact_email="",
        livertox_download_timeout=30.0,
        livertox_archive="livertox_NBK547852.tar.gz",
        livertox_yield_interval=25,
        livertox_skip_deterministic_ratio=0.8,
        livertox_monograph_max_workers=4,
        rxnav_request_timeout=12.0,
        terminology_download_timeout=300.0,
        drugcentral_enabled=True,
        drugcentral_host="unmtid-dbs.net",
        drugcentral_port=5433,
        drugcentral_database="drugcentral",
        drugcentral_user="drugman",
        chembl_enabled=True,
        chembl_max_phase=4,
        swissmedic_enabled=True,
    ),
    matching=MatchingRuntimeSettings(
        resolution_cache_limit=5000,
        fuzzy_min_query_length=5,
        fuzzy_shortlist_size=50,
        fuzzy_short_name_length=8,
        fuzzy_short_max_distance=1,
        fuzzy_long_max_distance=2,
        fuzzy_min_margin=1,
        external_fallback_enabled=True,
        external_max_requests_per_name=4,
        external_negative_cache_days=7,
    ),
    advanced=AdvancedRuntimeSettings(
        default_llm_timeout=3600.0,
        parser_llm_timeout=3600.0,
        disease_llm_timeout=3600.0,
        clinical_llm_timeout=3600.0,
        livertox_llm_timeout=3600.0,
        minimum_llm_timeout=5.0,
        cloud_llm_timeout_cap=1800.0,
        local_llm_timeout_cap=45.0,
        ollama_server_start_timeout=15.0,
        max_excerpt_length=8000,
    ),
)
