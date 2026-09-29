from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.constants import DEFAULT_NCBI_CONTACT_EMAIL
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
    rxnav_max_concurrency: int = Field(ge=1, le=64)


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
    rxnav_max_concurrency: int | None = Field(default=None, ge=1, le=64)


###############################################################################
class MatchingRuntimeSettings(_StrictModel):
    direct_confidence: float = Field(ge=0.0, le=1.0)
    master_confidence: float = Field(ge=0.0, le=1.0)
    synonym_confidence: float = Field(ge=0.0, le=1.0)
    normalization_cache_limit: int = Field(ge=1, le=2_000_000)
    match_cache_limit: int = Field(ge=1, le=2_000_000)
    alias_cache_limit: int = Field(ge=1, le=2_000_000)
    min_confidence: float = Field(ge=0.0, le=1.0)
    token_min_length: int = Field(ge=1, le=128)
    catalog_index_limit: int = Field(ge=1, le=2_000_000)
    spelling_confidence: float = Field(ge=0.0, le=1.0)
    spelling_min_query_length: int = Field(ge=1, le=128)
    spelling_short_name_length: int = Field(ge=1, le=128)
    spelling_short_max_distance: int = Field(ge=0, le=16)
    spelling_long_max_distance: int = Field(ge=0, le=16)


class MatchingRuntimeSettingsPatch(_PatchModel):
    direct_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    master_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    synonym_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    normalization_cache_limit: int | None = Field(default=None, ge=1, le=2_000_000)
    match_cache_limit: int | None = Field(default=None, ge=1, le=2_000_000)
    alias_cache_limit: int | None = Field(default=None, ge=1, le=2_000_000)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    token_min_length: int | None = Field(default=None, ge=1, le=128)
    catalog_index_limit: int | None = Field(default=None, ge=1, le=2_000_000)
    spelling_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    spelling_min_query_length: int | None = Field(default=None, ge=1, le=128)
    spelling_short_name_length: int | None = Field(default=None, ge=1, le=128)
    spelling_short_max_distance: int | None = Field(default=None, ge=0, le=16)
    spelling_long_max_distance: int | None = Field(default=None, ge=0, le=16)


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
        ncbi_contact_email=DEFAULT_NCBI_CONTACT_EMAIL,
        livertox_download_timeout=30.0,
        livertox_archive="livertox_NBK547852.tar.gz",
        livertox_yield_interval=25,
        livertox_skip_deterministic_ratio=0.8,
        livertox_monograph_max_workers=4,
        rxnav_request_timeout=12.0,
        rxnav_max_concurrency=10,
    ),
    matching=MatchingRuntimeSettings(
        direct_confidence=1.0,
        master_confidence=0.92,
        synonym_confidence=0.9,
        normalization_cache_limit=10000,
        match_cache_limit=5000,
        alias_cache_limit=2000,
        min_confidence=0.9,
        token_min_length=4,
        catalog_index_limit=75000,
        spelling_confidence=0.94,
        spelling_min_query_length=6,
        spelling_short_name_length=8,
        spelling_short_max_distance=1,
        spelling_long_max_distance=2,
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
