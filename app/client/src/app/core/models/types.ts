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
  recommended_for_local_extraction: boolean;
  recommended_rank: number | null;
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
  embedding_backend: string;
  ollama_embedding_model: string;
  hf_embedding_model: string;
  cloud_provider: string;
  cloud_embedding_model: string;
  use_cloud_embeddings: boolean;
  reset_vector_collection: boolean;
  vector_stream_batch_size: number;
  embedding_max_workers: number;
};

export type ModelConfigStateResponse = {
  local_models: LocalModelCard[];
  cloud_model_choices: Partial<Record<CloudProvider, string[]>>;
  use_cloud_services: boolean;
  llm_provider: string;
  cloud_model: string | null;
  clinical_model: string | null;
  text_extraction_model: string | null;
  ollama_temperature: number;
  cloud_temperature: number;
    ollama_reasoning: boolean;
    ollama_seed: number | null;
  rag_settings: RagSettings;
  rag_model: string | null;
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
    ollama_seed?: number | null;
  rag_settings?: Partial<RagSettings>;
};

export type AccessKeyProvider = "openai" | "gemini" | "brave";

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
  | "session_revision";

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
  version: number;
  original_session_id: number | null;
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

export type ClinicalSessionDetail = {
  session_id: number;
  patient_name: string | null;
  visit_date: string | null;
  session_timestamp: string | null;
  version: number;
  original_session_id: number | null;
  status: InspectionSessionStatus;
  text_extraction_model: string | null;
  clinical_model: string | null;
  metadata: Record<string, unknown>;
  sections: Record<string, string>;
  session_text: string;
  source_clinical_text: string;
  result_payload: Record<string, unknown>;
  report: string | null;
  official_report_text: string | null;
  manual_edit_history: ManualReportEditAudit[];
};

export type ClinicalSessionUpdateRequest = {
  session_text?: string | null;
  report_text?: string | null;
  edited_fields?: string[];
  reviewer_note?: string | null;
  edited_by?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type ManualReportEditAudit = {
  session_id: number;
  current_version_id: number;
  edited_by: string | null;
  actor_id: string | null;
  actor_display_name: string | null;
  actor_source: "authenticated_user" | "local_profile" | "manual_entry" | "system" | "unknown";
  actor_confidence: "verified" | "unverified" | "system";
  edited_at: string;
  previous_text_hash: string;
  new_text_hash: string;
  edited_fields: string[];
  reviewer_note: string | null;
  metadata: Record<string, unknown>;
};

export type ManualReportEditRequest = {
  report_text: string;
  edited_fields?: string[];
  reviewer_note?: string | null;
  edited_by?: string | null;
  metadata?: Record<string, unknown>;
};

export type ManualReportEditResponse = {
  session: ClinicalSessionDetail;
  audit: ManualReportEditAudit;
};

export type InspectionTimelineEventType = "therapy" | "disease" | "lab" | "other";
export type InspectionTimelineTimingType =
  | "explicit_date"
  | "relative"
  | "duration"
  | "recurring"
  | "uncertain"
  | "ordering";

export type InspectionTimelineEvent = {
  event_id: string;
  title: string;
  description: string | null;
  event_type: InspectionTimelineEventType;
  timing_type: InspectionTimelineTimingType;
  event_date: string | null;
  relative_time: string | null;
  extracted_timing_text: string | null;
  source_evidence: string | null;
  linked_patient_event_ids: string[];
  source: string | null;
  confidence: number | null;
  confidence_rationale: string | null;
  sort_order: number;
};

export type InspectionSessionTimeline = {
  timeline_id?: number | null;
  session_id: number;
  generated_at: string;
  generation_status?: "llm_generated" | "fallback";
  generation_note?: string | null;
  source_model?: string | null;
  source_kind?: "local" | "cloud" | null;
  model_provider?: string | null;
  events: InspectionTimelineEvent[];
};

export type InspectionSessionTimelineRequest = {
  force_regenerate?: boolean;
};

export type InspectionSessionTimelinePreview = {
  timeline_id: number | null;
  session_id: number;
  generated_at: string;
  generation_status?: "llm_generated" | "fallback";
  generation_note?: string | null;
  source_model?: string | null;
  source_kind?: "local" | "cloud" | null;
  model_provider?: string | null;
  event_count: number;
  start_date: string | null;
  end_date: string | null;
  title?: string | null;
};

export type InspectionSessionTimelineListResponse = {
  items: InspectionSessionTimelinePreview[];
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

export type InspectionRxNavUpdateRequest = {
  drug_name: string;
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

export type InspectionUpdateTarget = "rxnav" | "livertox" | "rag";

export type InspectionUpdateConfigResponse = {
  target: InspectionUpdateTarget;
  defaults: Record<string, unknown>;
  allowed_fields: string[];
  summary: Record<string, unknown>;
  read_only: boolean;
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

export type InspectionRagVectorizationSummary = {
  chunk_size: number;
  chunk_overlap: number;
  embedding_batch_size: number;
  vector_stream_batch_size: number;
  embedding_max_workers: number;
  embedding_backend: string;
  ollama_embedding_model: string;
  hf_embedding_model: string;
  cloud_provider: CloudProvider;
  cloud_embedding_model: string;
  use_cloud_embeddings: boolean;
  reset_vector_collection: boolean;
};

export type InspectionRagUpdateRequest = {
  documents_path?: string;
};

export type InspectionRagDocumentRow = {
  path: string;
  file_name: string;
  extension: string;
  file_size: number;
  last_modified: string;
  supported_for_ingestion: boolean;
  vector_model?: string | null;
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




export type SessionRevisionRequest = {
  selected_text?: string | null;
  revision_instruction?: string | null;
  model_overrides?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  max_tasks?: number;
  max_tool_iterations?: number;
  allowed_tools?: string[] | null;
  revision_goal?: "full_report_revision" | "selected_text_revision" | "metadata_review";
  dry_run?: boolean;
};

export type RevisionJobResult = Record<string, unknown> & {
  pipeline_run_id?: string;
  revision_version_id?: number;
  revised_session_id?: number | null;
  revision_status?: string;
  task_count?: number;
  tool_call_count?: number;
  blocking_issue_count?: number;
  manual_review_required?: boolean;
};

export type RevisionJobStatusResponse = JobStatusResponse<RevisionJobResult>;

export type RevisionPipelineStep = {
  step_name: string;
  step_index: number;
  step_count: number;
  status: string;
  output_summary?: Record<string, unknown> | null;
  output_payload?: Record<string, unknown> | null;
  error?: Record<string, unknown> | null;
};

export type RevisionPipelineStepListResponse = { items: RevisionPipelineStep[] };

export type RevisionArtifact = {
  artifact_key: string | null;
  status: string | null;
  payload: Record<string, unknown> | null;
};

export type RevisionArtifactListResponse = { items: RevisionArtifact[] };

export type RevisionClinicalReviewUpdateRequest = {
  clinical_review_status: "approved" | "rejected";
  reviewer_note?: string | null;
  reviewed_by?: string | null;
  metadata?: Record<string, unknown>;
};
