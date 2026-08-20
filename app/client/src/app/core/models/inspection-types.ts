import type {
  ClinicalActorConfidence,
  ClinicalActorSource,
  JobStatusResponse,
} from "./types";

export type InspectionUpdateJobResult = {
  phase?: string;
  step_index?: number;
  step_count?: number;
  progress_message?: string;
  summary?: Record<string, unknown>;
  [key: string]: unknown;
};

export type InspectionUpdateJobStatusResponse = JobStatusResponse<InspectionUpdateJobResult>;

export type InspectionTimelineJobResult = {
  session_id?: number;
  timeline_id?: number | null;
  progress_message?: string;
  [key: string]: unknown;
};

export type InspectionTimelineJobStatusResponse = JobStatusResponse<InspectionTimelineJobResult>;

export type InspectionUpdateJobListResponse = {
  jobs: InspectionUpdateJobStatusResponse[];
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
  actor_source: ClinicalActorSource;
  actor_confidence: ClinicalActorConfidence;
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
export type InspectionTimelineDatePrecision = "day" | "month" | "year";
export type InspectionTimelineDateCertainty = "explicit" | "inferred" | "uncertain";

export type InspectionTimelineEvent = {
  event_id: string;
  title: string;
  description: string | null;
  event_type: InspectionTimelineEventType;
  timing_type: InspectionTimelineTimingType;
  event_date: string | null;
  event_date_end?: string | null;
  date_precision?: InspectionTimelineDatePrecision | null;
  date_certainty?: InspectionTimelineDateCertainty;
  uncertainty_reason?: string | null;
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
  generation_error_code?:
    | "network_unavailable"
    | "timeout"
    | "authentication"
    | "rate_limited"
    | "upstream_error"
    | "invalid_response"
    | "configuration"
    | "provider_error"
    | "unknown"
    | null;
  source_model?: string | null;
  source_kind?: "local" | "cloud" | null;
  model_provider?: string | null;
  events: InspectionTimelineEvent[];
};

export type InspectionSessionTimelineRequest = {
  force_regenerate?: boolean;
  model_overrides?: InspectionSessionTimelineModelOverrides | null;
};

export type InspectionSessionTimelineModelOverrides = {
  use_cloud_services: boolean;
  llm_provider?: string | null;
  cloud_model?: string | null;
  text_extraction_model?: string | null;
};

export type InspectionSessionTimelinePreview = {
  timeline_id: number | null;
  session_id: number;
  generated_at: string;
  generation_status?: "llm_generated" | "fallback";
  generation_note?: string | null;
  generation_error_code?:
    | "network_unavailable"
    | "timeout"
    | "authentication"
    | "rate_limited"
    | "upstream_error"
    | "invalid_response"
    | "configuration"
    | "provider_error"
    | "unknown"
    | null;
  source_model?: string | null;
  source_kind?: "local" | "cloud" | null;
  model_provider?: string | null;
  event_count: number;
  start_date: string | null;
  end_date: string | null;
  title?: string | null;
  source_evidence_event_count: number;
  missing_evidence_event_count: number;
  uncertain_event_count: number;
  undated_event_count: number;
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

export type InspectionUpdateOverridesByTarget = {
  rxnav: InspectionRxNavOverrideRequest;
  livertox: InspectionLiverToxOverrideRequest;
  rag: InspectionRagUpdateRequest;
};

export type InspectionUpdateStartRequest =
  | { target: "rxnav"; payload: InspectionUpdateOverridesByTarget["rxnav"] }
  | { target: "livertox"; payload: InspectionUpdateOverridesByTarget["livertox"] }
  | { target: "rag"; payload: InspectionUpdateOverridesByTarget["rag"] };

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
  embedding_device: string;
  embedding_offline_mode: boolean;
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
  embedding_model: string;
  embedding_revision: string;
  index_status: string;
  embedding_fingerprint: string | null;
  built_at: string | null;
};
