export type RuntimeSettingsCategory = 'general' | 'data' | 'integrations' | 'advanced';

export interface GeneralRuntimeSettings {
  polling_interval: number;
}

export interface DataRuntimeSettings {
  drug_name_min_length: number;
  drug_name_max_length: number;
  drug_name_max_tokens: number;
}

export interface IntegrationRuntimeSettings {
  livertox_download_timeout: number;
  rxnav_request_timeout: number;
  rxnav_max_concurrency: number;
}

export interface AdvancedRuntimeSettings {
  default_llm_timeout: number;
  clinical_llm_timeout: number;
  livertox_llm_timeout: number;
  minimum_llm_timeout: number;
  cloud_llm_timeout_cap: number;
  local_llm_timeout_cap: number;
  max_excerpt_length: number;
}

export interface RuntimeSettingsValues {
  general: GeneralRuntimeSettings;
  data: DataRuntimeSettings;
  integrations: IntegrationRuntimeSettings;
  advanced: AdvancedRuntimeSettings;
}

export interface RuntimeSettingsStateResponse {
  values: RuntimeSettingsValues;
  defaults: RuntimeSettingsValues;
  source: 'settings/configurations.json';
  environment_editable: false;
  updated_at: string | null;
}

export interface RuntimeSettingsUpdateRequest {
  general?: Partial<GeneralRuntimeSettings>;
  data?: Partial<DataRuntimeSettings>;
  integrations?: Partial<IntegrationRuntimeSettings>;
  advanced?: Partial<AdvancedRuntimeSettings>;
}
