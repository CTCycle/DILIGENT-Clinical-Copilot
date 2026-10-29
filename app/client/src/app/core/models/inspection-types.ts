// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import type {
  ClinicalActorConfidence,
  ClinicalActorSource,
  JobStatus,
  JobStatusResponse,
} from "./types";

export type InspectionUpdateJobResult = {
  phase?: string;
  step_index?: number;
  step_count?: number;
  progress_message?: string;
  summary?: Record<string, unknown>;
  sources?: Partial<Record<InspectionUpdateAllTarget, InspectionStructuredSourceJobState>>;
  summaries?: Partial<Record<InspectionUpdateAllTarget, Record<string, unknown>>>;
  [key: string]: unknown;
};

export type InspectionStructuredSourceJobState = {
  status: JobStatus;
  progress: number;
  message: string;
  error: string | null;
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
  has_excerpt: boolean;
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

export type InspectionDiliRankItem = {
  drug_id: number | null;
  drug_name: string | null;
  ltkb_id: string;
  compound_name: string;
  severity_class: number | null;
  label_section: string | null;
  dili_concern: string;
  comment: string | null;
  source_url: string | null;
  source_last_modified: string | null;
};

export type InspectionDiliRankCatalogResponse = {
  items: InspectionDiliRankItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionDeleteResponse = {
  deleted: boolean;
};

export type InspectionUpdateTarget = "rxnav" | "livertox" | "terminology" | "dilirank" | "rag";

export const INSPECTION_UPDATE_ALL_TARGETS = [
  "livertox",
  "rxnav",
  "terminology",
  "dilirank",
] as const;

export type InspectionUpdateAllTarget = (typeof INSPECTION_UPDATE_ALL_TARGETS)[number];

export type InspectionUpdateOverridesByTarget = {
  rxnav: InspectionRxNavOverrideRequest;
  livertox: InspectionLiverToxOverrideRequest;
  terminology: InspectionTerminologyOverrideRequest;
  dilirank: InspectionDiliRankUpdateRequest;
  rag: InspectionRagUpdateRequest;
};

export type InspectionUpdateStartRequest =
  | { target: "rxnav"; payload: InspectionUpdateOverridesByTarget["rxnav"] }
  | { target: "livertox"; payload: InspectionUpdateOverridesByTarget["livertox"] }
  | { target: "terminology"; payload: InspectionUpdateOverridesByTarget["terminology"] }
  | { target: "dilirank"; payload: InspectionUpdateOverridesByTarget["dilirank"] }
  | { target: "rag"; payload: InspectionUpdateOverridesByTarget["rag"] };

export type InspectionStructuredSourcesUpdateRequest = {
  rxnav: InspectionRxNavOverrideRequest;
  livertox: InspectionLiverToxOverrideRequest;
  terminology: InspectionTerminologyOverrideRequest;
  dilirank: InspectionDiliRankUpdateRequest;
};

export type InspectionUpdateAllStartRequestMap = {
  [TTarget in InspectionUpdateAllTarget]: Extract<
    InspectionUpdateStartRequest,
    { target: TTarget }
  >;
};

export type InspectionUpdateAllConfigMap = Record<
  InspectionUpdateAllTarget,
  Record<string, unknown> | null
>;

export type InspectionUpdateAllConfigLoadingMap = Record<InspectionUpdateAllTarget, boolean>;

export type InspectionUpdateAllConfigErrorMap = Record<InspectionUpdateAllTarget, string | null>;

export type InspectionUpdateAllPhase =
  | "idle"
  | "starting"
  | "running"
  | "completed"
  | "partial_failure"
  | "cancelled";

export type InspectionUpdateAllTargetState = {
  started: boolean;
  jobId: string | null;
  status: JobStatus | null;
  progress: number;
  message: string;
  error: string | null;
};

export type InspectionUpdateAllState = {
  runId: number | null;
  phase: InspectionUpdateAllPhase;
  progress: number;
  message: string;
  cancelRequested: boolean;
  targets: Record<InspectionUpdateAllTarget, InspectionUpdateAllTargetState>;
};

export type InspectionUpdateConfigResponse = {
  target: InspectionUpdateTarget;
  defaults: Record<string, unknown>;
  allowed_fields: string[];
  summary: Record<string, unknown>;
  read_only: boolean;
};

export type InspectionRxNavOverrideRequest = {
  force_full?: boolean;
};

export type InspectionTerminologyOverrideRequest = {
  force_full?: boolean;
  drugcentral_enabled?: boolean;
  chembl_enabled?: boolean;
  swissmedic_enabled?: boolean;
};

export type TerminologyResolutionStatus = "verified" | "probable" | "ambiguous" | "unresolved";

export type TerminologyIngredient = {
  drug_id: number;
  name: string;
  concept_type: string;
  parent_drug_id: number | null;
  parent_name: string | null;
};

export type TerminologyCandidate = {
  drug_id: number;
  name: string;
  concept_type: string;
  matched_alias: string | null;
  alias_kind: string | null;
  source: string | null;
  country: string | null;
  edit_distance: number | null;
  similarity: number | null;
};

export type TerminologyResolution = {
  original: string;
  normalized: string;
  strengths: string[];
  qualifiers: string[];
  status: TerminologyResolutionStatus;
  accepted: boolean;
  method: string;
  drug_id: number | null;
  preferred_name: string | null;
  concept_type: string | null;
  ingredients: TerminologyIngredient[];
  precise_forms: TerminologyIngredient[];
  matched_alias: string | null;
  alias_kind: string | null;
  source: string | null;
  source_version: string | null;
  country: string | null;
  strength_supported: boolean | null;
  candidates: TerminologyCandidate[];
  review_reasons: string[];
  external_calls: number;
  terminology_revision: number | null;
};

export type TerminologySourceSummary = {
  source: string;
  enabled: boolean;
  license: string;
  attribution: string;
  last_status: string | null;
  last_version: string | null;
  last_finished_at: string | null;
  last_completed_version: string | null;
  last_completed_at: string | null;
  aliases: number;
  products: number;
  open_conflicts: number;
  last_error: string | null;
};

export type TerminologySummaryResponse = {
  revision: number | null;
  concepts: number;
  aliases: number;
  resolvable_aliases: number;
  provisional_aliases: number;
  products: number;
  verified_products: number;
  livertox_links: number;
  linked_monographs: number;
  total_monographs: number;
  open_conflicts: number;
  index: Record<string, number | null>;
  sources: TerminologySourceSummary[];
};

export type TerminologyImportRunItem = {
  id: number;
  source: string;
  source_version: string | null;
  mode: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  stats: Record<string, unknown> | null;
  error: string | null;
};

export type TerminologyConflictItem = {
  id: number;
  source: string;
  conflict_type: string;
  subject: string;
  existing_drug: string | null;
  proposed_drug: string | null;
  detail: string | null;
  status: string;
  detected_at: string;
};

export type TerminologyConflictsResponse = {
  items: TerminologyConflictItem[];
  total: number;
  offset: number;
  limit: number;
};

export type TerminologyAliasDecision = "confirm" | "reject";

export type TerminologyAliasDecisionResponse = {
  alias: string;
  drug_id: number;
  validation_status: "confirmed" | "rejected";
};

export type InspectionLiverToxOverrideRequest = {
  livertox_monograph_max_workers?: number;
  livertox_archive?: string;
  redownload?: boolean;
};

export type InspectionDiliRankUpdateRequest = {
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
  source_relative_path: string;
  file_name: string;
  extension: string;
  file_size: number;
  last_modified: string;
  supported_for_ingestion: boolean;
  content_fingerprint?: string | null;
  canonical_source_relative_path?: string | null;
  is_canonical_source: boolean;
  is_duplicate: boolean;
  duplicate_of?: string | null;
  duplicate_source_paths: string[];
  duplicate_alias_paths: string[];
  vector_model?: string | null;
};

export type InspectionRagDocumentsResponse = {
  items: InspectionRagDocumentRow[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionRagDirectoryItem = {
  name: string;
  path: string;
  is_dir: boolean;
};

export type InspectionRagDirectoryBrowseResponse = {
  current_path: string;
  parent_path: string | null;
  items: InspectionRagDirectoryItem[];
  drives: string[];
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
  physical_supported_file_count: number;
  unique_ingested_document_count: number;
  duplicate_file_count: number;
};
