export type RuntimeSettingsCategory =
  | 'general'
  | 'data'
  | 'integrations'
  | 'matching'
  | 'advanced';

export interface GeneralRuntimeSettings {
  polling_interval: number;
}

export interface DataRuntimeSettings {
  drug_name_min_length: number;
  drug_name_max_length: number;
  drug_name_max_tokens: number;
  text_extraction_batch_size: number;
  text_extraction_max_concurrency: number;
  retrieval_batch_size: number;
  retrieval_max_concurrency: number;
  clinical_assessment_batch_size: number;
  clinical_assessment_max_concurrency: number;
  min_best_score: number;
  high_confidence_min_score: number;
  high_confidence_min_margin: number;
  moderate_confidence_min_score: number;
  moderate_confidence_min_margin: number;
}

export interface IntegrationRuntimeSettings {
  ncbi_contact_email: string;
  livertox_download_timeout: number;
  livertox_archive: string;
  livertox_yield_interval: number;
  livertox_skip_deterministic_ratio: number;
  livertox_monograph_max_workers: number;
  rxnav_request_timeout: number;
  rxnav_max_concurrency: number;
}

export interface MatchingRuntimeSettings {
  direct_confidence: number;
  master_confidence: number;
  synonym_confidence: number;
  normalization_cache_limit: number;
  match_cache_limit: number;
  alias_cache_limit: number;
  min_confidence: number;
  token_min_length: number;
  catalog_index_limit: number;
  spelling_confidence: number;
  spelling_min_query_length: number;
  spelling_short_name_length: number;
  spelling_short_max_distance: number;
  spelling_long_max_distance: number;
}

export interface AdvancedRuntimeSettings {
  default_llm_timeout: number;
  parser_llm_timeout: number;
  disease_llm_timeout: number;
  clinical_llm_timeout: number;
  livertox_llm_timeout: number;
  minimum_llm_timeout: number;
  cloud_llm_timeout_cap: number;
  local_llm_timeout_cap: number;
  ollama_server_start_timeout: number;
  max_excerpt_length: number;
}

export interface RuntimeSettingsValues {
  general: GeneralRuntimeSettings;
  data: DataRuntimeSettings;
  integrations: IntegrationRuntimeSettings;
  matching: MatchingRuntimeSettings;
  advanced: AdvancedRuntimeSettings;
}

export interface RuntimeSettingsStateResponse {
  values: RuntimeSettingsValues;
  defaults: RuntimeSettingsValues;
  source: 'database';
  environment_editable: false;
  updated_at: string | null;
}

export interface RuntimeSettingsUpdateRequest {
  general?: Partial<GeneralRuntimeSettings>;
  data?: Partial<DataRuntimeSettings>;
  integrations?: Partial<IntegrationRuntimeSettings>;
  matching?: Partial<MatchingRuntimeSettings>;
  advanced?: Partial<AdvancedRuntimeSettings>;
}
