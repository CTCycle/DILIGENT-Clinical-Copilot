import type {
  ClinicalActorConfidence,
  ClinicalActorSource,
  JobStatusResponse,
} from "./types";

export type SessionRevisionRequest = {
  selected_text?: string | null;
  revision_instruction?: string | null;
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

export type RevisionClinicalReviewStatus = "under_review" | "approved_by_human" | "rejected_by_human";

export type RevisionClinicalReviewUpdateRequest = {
  clinical_review_status: RevisionClinicalReviewStatus;
  reviewer_note?: string | null;
  reviewed_by?: string | null;
  metadata?: Record<string, unknown>;
};

export type SessionVersionSummary = {
  version_id: number;
  session_id: number | null;
  root_session_id: number;
  source_version_id: number | null;
  revision_version_id: number;
  version_number: number;
  version_status:
    | "current"
    | "superseded"
    | "draft_revision"
    | "pending_qa"
    | "qa_failed"
    | "requires_human_review"
    | "llm_qa_passed"
    | "human_approved"
    | "human_rejected";
  revision_kind: "original" | "manual_edit" | "llm_assisted_revision";
  llm_qa_status:
    | "not_run"
    | "pending"
    | "passed"
    | "passed_with_warnings"
    | "failed"
    | "requires_human_review";
  clinical_review_status: "not_reviewed" | RevisionClinicalReviewStatus;
  pipeline_run_id: string | null;
  model_configuration: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

export type SessionVersionDetailResponse = {
  version: SessionVersionSummary;
  session: Record<string, unknown> | null;
};

export type RevisionClinicalReviewAction = {
  revision_version_id: number;
  session_id: number | null;
  clinical_review_status: RevisionClinicalReviewStatus;
  reviewer_note: string | null;
  reviewed_by: string | null;
  actor_id: string | null;
  actor_display_name: string | null;
  actor_source: ClinicalActorSource;
  actor_confidence: ClinicalActorConfidence;
  metadata: Record<string, unknown>;
  reviewed_at: string;
  created_at: string;
  updated_at: string;
};

export type RevisionClinicalReviewUpdateResponse = {
  version: SessionVersionSummary;
  review_action: RevisionClinicalReviewAction;
};
