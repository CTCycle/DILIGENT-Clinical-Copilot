export type CloudProvider = "openai" | "gemini";

export type RuntimeSettings = {
  useCloudServices: boolean;
  provider: CloudProvider;
  cloudModel: string | null;
  textExtractionModel: string;
  clinicalModel: string;
  temperature: number;
  reasoning: boolean;
};

export type LocalModelCard = {
  name: string;
  description: string;
  family: string;
  available_in_ollama: boolean;
};

export type ModelConfigStateResponse = {
  status: "success";
  local_models: LocalModelCard[];
  cloud_model_choices: Partial<Record<CloudProvider, string[]>>;
  use_cloud_services: boolean;
  llm_provider: CloudProvider;
  cloud_model: string | null;
  clinical_model: string | null;
  text_extraction_model: string | null;
  ollama_temperature: number;
  cloud_temperature: number;
  ollama_reasoning: boolean;
  updated_at: string | null;
};

export type ModelConfigUpdateRequest = {
  use_cloud_services?: boolean;
  llm_provider?: CloudProvider;
  cloud_model?: string | null;
  clinical_model?: string | null;
  text_extraction_model?: string | null;
  ollama_temperature?: number;
  cloud_temperature?: number;
  ollama_reasoning?: boolean;
};

export type AccessKeyProvider = "openai" | "gemini" | "tavily";

export type AccessKeyRecord = {
  id: number;
  provider: AccessKeyProvider;
  is_active: boolean;
  fingerprint: string;
  created_at: string | null;
  updated_at: string | null;
  last_used_at: string | null;
};

export type ClinicalFormState = {
  patientName: string;
  visitDate: string;
  patientImageDataUrl: string | null;
  anamnesis: string;
  drugs: string;
  laboratoryAnalysis: string;
  useRag: boolean;
  useWebSearch: boolean;
};

export type ClinicalRequestPayload = {
  name: string | null;
  visit_date: { day: number; month: number; year: number } | null;
  anamnesis: string | null;
  drugs: string | null;
  laboratory_analysis: string | null;
  patient_image_base64?: string | null;
  use_rag: boolean;
  use_web_search: boolean;
};

export type ApiResult = {
  message: string;
  json: unknown;
};

export type JobType =
  | "clinical"
  | "ollama_pull"
  | "rxnav_update"
  | "livertox_update"
  | "dili_priors_update"
  | "drug_labels_update"
  | "rag_update";

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type ClinicalJobResult = {
  report?: string;
  progress_stage?: string;
  progress_message?: string;
  [key: string]: unknown;
};

export type OllamaPullJobResult = {
  model?: string;
  pulled?: boolean;
  progress_status?: string;
  progress_message?: string;
  total_bytes?: number;
  completed_bytes?: number;
  [key: string]: unknown;
};

export type JobStartResponse = {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
  message: string;
  poll_interval: number;
};

export type JobStatusResponse<TJobResult extends Record<string, unknown> = ClinicalJobResult> = {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
  progress: number;
  result: TJobResult | null;
  error: string | null;
};

export type InspectionUpdateJobResult = {
  phase?: string;
  step_index?: number;
  step_count?: number;
  progress_message?: string;
  summary?: Record<string, unknown>;
  [key: string]: unknown;
};

export type InspectionUpdateJobStatusResponse = JobStatusResponse<InspectionUpdateJobResult>;

export type JobCancelResponse = {
  job_id: string;
  success: boolean;
  message: string;
};

export type InspectionSessionStatus = "successful" | "failed";
export type InspectionDateFilterMode = "before" | "after" | "exact";

export type InspectionSessionItem = {
  session_id: number;
  patient_name: string | null;
  session_timestamp: string | null;
  status: InspectionSessionStatus;
  total_duration: number | null;
  has_report: boolean;
  has_timeline: boolean;
  can_generate_timeline: boolean;
};

export type InspectionSessionCatalogResponse = {
  items: InspectionSessionItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionSessionReportResponse = {
  session_id: number;
  report: string;
};

export type InspectionTimelineEventType = "therapy" | "disease" | "lab" | "other";

export type InspectionTimelineEvent = {
  event_id: string;
  title: string;
  description: string | null;
  event_type: InspectionTimelineEventType;
  event_date: string | null;
  relative_time: string | null;
  source: string | null;
  confidence: number | null;
  sort_order: number;
};

export type InspectionSessionTimeline = {
  session_id: number;
  generated_at: string;
  events: InspectionTimelineEvent[];
};

export type InspectionSessionTimelineRequest = {
  force_regenerate?: boolean;
};

export type InspectionCatalogQuery = {
  search?: string;
  offset?: number;
  limit?: number;
};

export type InspectionSessionQuery = InspectionCatalogQuery & {
  status?: InspectionSessionStatus;
  date_mode?: InspectionDateFilterMode;
  date?: string;
};

export type InspectionRxNavItem = {
  drug_id: number;
  drug_name: string;
  last_update: string | null;
};

export type InspectionRxNavCatalogResponse = {
  items: InspectionRxNavItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionAliasEntry = {
  alias: string;
  alias_kind: string;
};

export type InspectionAliasGroup = {
  source: string;
  aliases: InspectionAliasEntry[];
};

export type InspectionDrugAliasesResponse = {
  drug_id: number;
  drug_name: string;
  groups: InspectionAliasGroup[];
};

export type InspectionLiverToxItem = {
  drug_id: number;
  drug_name: string;
  last_update: string | null;
};

export type InspectionLiverToxCatalogResponse = {
  items: InspectionLiverToxItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionLiverToxExcerptResponse = {
  drug_id: number;
  drug_name: string;
  excerpt: string;
  last_update: string | null;
};

export type InspectionDeleteResponse = {
  deleted: boolean;
};

export type InspectionUpdateTarget = "rxnav" | "livertox" | "dili_priors" | "drug_labels" | "rag";

export type InspectionUpdateConfigResponse = {
  target: InspectionUpdateTarget;
  defaults: Record<string, unknown>;
  allowed_fields: string[];
};

export type InspectionRxNavOverrideRequest = {
  rxnav_request_timeout?: number;
  rxnav_max_concurrency?: number;
};

export type InspectionLiverToxOverrideRequest = {
  livertox_monograph_max_workers?: number;
  livertox_archive?: string;
  redownload?: boolean;
};

export type InspectionDiliPriorsOverrideRequest = {
  redownload?: boolean;
};

export type InspectionDrugLabelsOverrideRequest = {
  dailymed_request_timeout?: number;
  dailymed_max_concurrency?: number;
  redownload?: boolean;
};

export type InspectionRagOverrideRequest = {
  documents_path?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  embedding_batch_size?: number;
  vector_stream_batch_size?: number;
  embedding_max_workers?: number;
  embedding_backend?: string;
  ollama_embedding_model?: string;
  hf_embedding_model?: string;
  cloud_provider?: CloudProvider;
  cloud_embedding_model?: string;
  use_cloud_embeddings?: boolean;
  reset_vector_collection?: boolean;
};

export type InspectionRagDocumentRow = {
  path: string;
  file_name: string;
  extension: string;
  file_size: number;
  last_modified: string;
  supported_for_ingestion: boolean;
};

export type InspectionRagDocumentsResponse = {
  items: InspectionRagDocumentRow[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionRagVectorStoreSummary = {
  source_documents_path: string;
  vector_db_path: string;
  collection_name: string;
  collection_exists: boolean;
  embedding_count: number;
  distinct_document_count: number;
  embedding_dimension: number | null;
  index_ready: boolean;
  configured_metric: string | null;
  configured_index_type: string | null;
};

export type InspectionDiliPriorItem = {
  drug_id: number;
  drug_name: string;
  dilirank_class: string | null;
  dilist_class: string | null;
  linked_source_count: number;
};

export type InspectionDiliPriorCatalogResponse = {
  items: InspectionDiliPriorItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionDiliPriorDetailResponse = {
  drug_id: number;
  drug_name: string;
  annotations: Record<string, unknown>[];
};

export type InspectionDrugLabelItem = {
  drug_id: number;
  drug_name: string;
  source: string;
  effective_date: string | null;
  retained_section_count: number;
};

export type InspectionDrugLabelCatalogResponse = {
  items: InspectionDrugLabelItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionDrugLabelSectionsResponse = {
  drug_id: number;
  drug_name: string;
  source: string;
  set_id: string;
  spl_version: number;
  effective_date: string | null;
  sections: {
    section_key: string;
    section_title: string | null;
    text: string;
    contains_hepatic_keywords: boolean;
    display_order: number;
  }[];
};

