export type CloudProvider =
  | "openai"
  | "gemini"
  | "deepseek"
  | "anthropic"
  | "opencode_zen"
  | "opencode_go";

export type ReasoningLevel = "off" | "low" | "medium" | "high";

export type RuntimeSettings = {
  useCloudServices: boolean;
  provider: CloudProvider;
  cloudModel: string | null;
  textExtractionModel: string;
  clinicalModel: string;
  revisionModel: string;
  timelineModel: string;
  reasoning: ReasoningLevel;
};

export type LocalModelCard = {
  name: string;
  description: string;
  family: string;
  available_in_ollama: boolean;
  recommended_for_local_extraction: boolean;
  recommended_rank: number | null;
};

export type CatalogProvider = CloudProvider | "ollama";
export type CatalogStatus =
  | "available"
  | "cached"
  | "not_loaded"
  | "unavailable"
  | "authentication_required";
export type LocalCatalogMetadata = {
  status: CatalogStatus;
  updated_at: string | null;
  message: string | null;
};

export type RagSettings = {
  chunk_size: number;
  chunk_overlap: number;
  embedding_batch_size: number;
  use_hybrid_search: boolean;
  use_reranking: boolean;
  retrieval_candidate_count: number;
  retrieval_selected_count: number;
  reranker_model: string;
  hybrid_vector_weight: number;
  hybrid_text_weight: number;
  vector_stream_batch_size: number;
  embedding_device: string;
  embedding_offline_mode: boolean;
};

export type ModelConfigStateResponse = {
  local_models: LocalModelCard[];
  cloud_providers: CloudProviderDescriptor[];
  local_catalog: LocalCatalogMetadata;
  use_cloud_services: boolean;
  llm_provider: string;
  cloud_model: string | null;
  clinical_model: string | null;
  text_extraction_model: string | null;
  revision_model: string | null;
  timeline_model: string | null;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
  rag_settings: RagSettings;
  embedding_runtime: EmbeddingRuntimeStatus;
  embedding_index: EmbeddingIndexStatus;
  updated_at: string | null;
};

export type ModelConfigPersistResponse = {
  use_cloud_services: boolean;
  llm_provider: CloudProvider;
  cloud_model: string | null;
  clinical_model: string | null;
  text_extraction_model: string | null;
  revision_model: string | null;
  timeline_model: string | null;
  reasoning_level: ReasoningLevel;
  ollama_seed: number | null;
  rag_settings: RagSettings;
  updated_at: string | null;
};

export type EmbeddingRuntimeStatus = {
  model_display_name: string;
  model_revision: string;
  device: string;
  cache_status: string;
  loaded: boolean;
};

export type EmbeddingIndexStatus = {
  status: string;
  fingerprint: string | null;
  document_count: number;
  chunk_count: number;
  built_at: string | null;
};

export type EmbeddingStatusResponse = {
  embedding_runtime: EmbeddingRuntimeStatus;
  embedding_index: EmbeddingIndexStatus;
};

export type ModelConfigUpdateRequest = {
  use_cloud_services?: boolean;
  llm_provider?: CloudProvider;
  cloud_model?: string | null;
  clinical_model?: string | null;
  text_extraction_model?: string | null;
  revision_model?: string | null;
  timeline_model?: string | null;
  reasoning_level?: ReasoningLevel;
  ollama_seed?: number | null;
  rag_settings?: Partial<RagSettings>;
};

export type AccessKeyProvider =
  | "openai"
  | "gemini"
  | "deepseek"
  | "anthropic"
  | "opencode"
  | "brave";

export type ProviderCapabilities = {
  chat: boolean;
  structured_output: boolean;
  reasoning: boolean;
  model_listing: boolean;
  embeddings: boolean;
  vision: boolean;
};

export type CloudModelDescriptor = {
  id: string;
  display_name: string;
  endpoint_family: string | null;
  capabilities: ProviderCapabilities | null;
  input_token_limit: number | null;
  output_token_limit: number | null;
  supports_thinking: boolean | null;
  supports_temperature: boolean | null;
};

export type CloudProviderDescriptor = {
  id: CloudProvider;
  display_name: string;
  credential_scope: AccessKeyProvider;
  capabilities: ProviderCapabilities;
  catalog_status: CatalogStatus;
  catalog_updated_at?: string | null;
  catalog_message?: string | null;
  models: CloudModelDescriptor[];
};

export type ModelCatalogOperationResponse = {
  catalog_provider: CatalogProvider;
  outcome: "cached" | "refreshed" | "failed";
  error: string | null;
  state: ModelConfigStateResponse;
};

export type ConnectivityCheckRequest = { provider: CloudProvider; model: string };
export type ConnectivityCheckResponse = {
  provider: CloudProvider;
  model: string;
  ok: boolean;
  response_preview: string | null;
  error: string | null;
};

export type AccessKeyRecord = {
  id: number;
  provider: AccessKeyProvider;
  is_active: boolean;
  fingerprint: string;
  created_at: string | null;
  updated_at: string | null;
  last_used_at: string | null;
};

export type ClinicalActorSource = "authenticated_user" | "local_profile" | "manual_entry" | "system" | "unknown";
export type ClinicalActorConfidence = "verified" | "unverified" | "system";

export type ClinicalFormState = {
  patientName: string;
  visitDate: string;
  patientImageDataUrl: string | null;
  clinicalInput: string;
  useRag: boolean;
};

export type ClinicalRequestPayload = {
  name: string | null;
  visit_date: { day: number; month: number; year: number } | null;
  clinical_input: string | null;
  selected_model_providers: string[];
  patient_image_base64?: string | null;
  use_rag: boolean;
};

export type ClinicalSectionTemplateResponse = {
  headings: Record<string, string[]>;
  template: string;
};

export type ClinicalInputPreflightIssue = {
  severity: "blocking" | "non_blocking";
  code: string;
  message: string;
  field?: string | null;
  title: string;
  description: string;
  affected_section: string;
  consequence: string;
  continuation_allowed: boolean;
};

export type ClinicalInputPreflightResponse = {
  ready: boolean;
  blocking_issues: ClinicalInputPreflightIssue[];
  non_blocking_issues: ClinicalInputPreflightIssue[];
  runtime_settings: Record<string, unknown>;
  extraction_quality: Record<string, unknown>;
  deterministic_diagnostics: Record<string, unknown>;
  rag_readiness?: RagReadiness | null;
};

export type RagReadiness = {
  requested: boolean;
  available: boolean;
  backend: string;
  model?: string | null;
  reason_code?: string | null;
  message?: string | null;
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
  | "rag_update"
  | "session_revision"
  | "session_timeline";

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
  created_at?: number | null;
  completed_at?: number | null;
  version?: number | null;
  stop_requested?: boolean;
};

export type JobCancelResponse = {
  job_id: string;
  success: boolean;
  message: string;
};

export type {
  InspectionUpdateJobResult,
  InspectionUpdateJobStatusResponse,
  InspectionTimelineJobResult,
  InspectionTimelineJobStatusResponse,
  InspectionUpdateJobListResponse,
  InspectionSessionStatus,
  InspectionDateFilterMode,
  InspectionSessionItem,
  InspectionSessionCatalogResponse,
  ClinicalSessionDetail,
  ClinicalSessionUpdateRequest,
  ManualReportEditAudit,
  ManualReportEditRequest,
  ManualReportEditResponse,
  InspectionTimelineEventType,
  InspectionTimelineTimingType,
  InspectionTimelineDatePrecision,
  InspectionTimelineDateCertainty,
  InspectionTimelineEvent,
  InspectionSessionTimeline,
  InspectionSessionTimelineRequest,
  InspectionSessionTimelinePreview,
  InspectionSessionTimelineListResponse,
  InspectionCatalogQuery,
  InspectionSessionQuery,
  InspectionRxNavItem,
  InspectionRxNavUpdateRequest,
  InspectionRxNavCatalogResponse,
  InspectionAliasEntry,
  InspectionAliasGroup,
  InspectionDrugAliasesResponse,
  InspectionLiverToxItem,
  InspectionLiverToxCatalogResponse,
  InspectionLiverToxExcerptResponse,
  InspectionDeleteResponse,
  InspectionUpdateTarget,
  InspectionUpdateOverridesByTarget,
  InspectionUpdateStartRequest,
  InspectionUpdateConfigResponse,
  InspectionRxNavOverrideRequest,
  InspectionLiverToxOverrideRequest,
  InspectionRagVectorizationSummary,
  InspectionRagUpdateRequest,
  InspectionRagDocumentRow,
  InspectionRagDocumentsResponse,
  InspectionRagVectorStoreSummary,
} from "./inspection-types";

export type {
  SessionRevisionRequest,
  RevisionJobResult,
  RevisionJobStatusResponse,
  RevisionPipelineStep,
  RevisionPipelineStepListResponse,
  RevisionArtifact,
  RevisionArtifactListResponse,
  RevisionClinicalReviewStatus,
  RevisionClinicalReviewUpdateRequest,
  SessionVersionSummary,
  RevisionClinicalReviewAction,
  RevisionClinicalReviewUpdateResponse,
} from "./revision-types";
