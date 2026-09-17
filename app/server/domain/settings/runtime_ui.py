from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RuntimeSettingsCategory = Literal["general", "data", "integrations", "advanced"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PatchModel(_StrictModel):
    @model_validator(mode="after")
    def validate_patch(self) -> "_PatchModel":
        if not self.model_fields_set:
            raise ValueError("At least one setting must be provided.")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null.")
        return self


class GeneralRuntimeSettings(_StrictModel):
    polling_interval: float = Field(gt=0.0, le=60.0)


class GeneralRuntimeSettingsPatch(_PatchModel):
    polling_interval: float | None = Field(default=None, gt=0.0, le=60.0)


class DataRuntimeSettings(_StrictModel):
    drug_name_min_length: int = Field(ge=1, le=200)
    drug_name_max_length: int = Field(ge=1, le=1000)
    drug_name_max_tokens: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def validate_lengths(self) -> "DataRuntimeSettings":
        if self.drug_name_max_length < self.drug_name_min_length:
            raise ValueError(
                "Drug-name maximum length cannot be smaller than minimum length."
            )
        return self


class DataRuntimeSettingsPatch(_PatchModel):
    drug_name_min_length: int | None = Field(default=None, ge=1, le=200)
    drug_name_max_length: int | None = Field(default=None, ge=1, le=1000)
    drug_name_max_tokens: int | None = Field(default=None, ge=1, le=64)


class IntegrationRuntimeSettings(_StrictModel):
    livertox_download_timeout: float = Field(gt=0.0, le=3600.0)
    rxnav_request_timeout: float = Field(gt=0.0, le=120.0)
    rxnav_max_concurrency: int = Field(ge=1, le=64)


class IntegrationRuntimeSettingsPatch(_PatchModel):
    livertox_download_timeout: float | None = Field(
        default=None, gt=0.0, le=3600.0
    )
    rxnav_request_timeout: float | None = Field(default=None, gt=0.0, le=120.0)
    rxnav_max_concurrency: int | None = Field(default=None, ge=1, le=64)


class AdvancedRuntimeSettings(_StrictModel):
    default_llm_timeout: float = Field(gt=0.0, le=86400.0)
    clinical_llm_timeout: float = Field(gt=0.0, le=86400.0)
    livertox_llm_timeout: float = Field(gt=0.0, le=86400.0)
    minimum_llm_timeout: float = Field(gt=0.0, le=3600.0)
    cloud_llm_timeout_cap: float = Field(gt=0.0, le=86400.0)
    local_llm_timeout_cap: float = Field(gt=0.0, le=86400.0)
    max_excerpt_length: int = Field(ge=500, le=100000)

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
    clinical_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    livertox_llm_timeout: float | None = Field(default=None, gt=0.0, le=86400.0)
    minimum_llm_timeout: float | None = Field(default=None, gt=0.0, le=3600.0)
    cloud_llm_timeout_cap: float | None = Field(default=None, gt=0.0, le=86400.0)
    local_llm_timeout_cap: float | None = Field(default=None, gt=0.0, le=86400.0)
    max_excerpt_length: int | None = Field(default=None, ge=500, le=100000)


class RuntimeSettingsValues(_StrictModel):
    general: GeneralRuntimeSettings
    data: DataRuntimeSettings
    integrations: IntegrationRuntimeSettings
    advanced: AdvancedRuntimeSettings


class RuntimeSettingsUpdateRequest(_StrictModel):
    general: GeneralRuntimeSettingsPatch | None = None
    data: DataRuntimeSettingsPatch | None = None
    integrations: IntegrationRuntimeSettingsPatch | None = None
    advanced: AdvancedRuntimeSettingsPatch | None = None

    @model_validator(mode="after")
    def validate_update(self) -> "RuntimeSettingsUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("At least one settings category must be provided.")
        for field_name in self.model_fields_set:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} cannot be null.")
        return self


class RuntimeSettingsStateResponse(_StrictModel):
    values: RuntimeSettingsValues
    defaults: RuntimeSettingsValues
    source: Literal["settings/configurations.json"] = "settings/configurations.json"
    environment_editable: Literal[False] = False
    updated_at: datetime | None = None


RUNTIME_SETTINGS_DEFAULTS = RuntimeSettingsValues(
    general=GeneralRuntimeSettings(polling_interval=1.0),
    data=DataRuntimeSettings(
        drug_name_min_length=3,
        drug_name_max_length=200,
        drug_name_max_tokens=8,
    ),
    integrations=IntegrationRuntimeSettings(
        livertox_download_timeout=30.0,
        rxnav_request_timeout=12.0,
        rxnav_max_concurrency=10,
    ),
    advanced=AdvancedRuntimeSettings(
        default_llm_timeout=3600.0,
        clinical_llm_timeout=3600.0,
        livertox_llm_timeout=3600.0,
        minimum_llm_timeout=5.0,
        cloud_llm_timeout_cap=1800.0,
        local_llm_timeout_cap=45.0,
        max_excerpt_length=8000,
    ),
)
