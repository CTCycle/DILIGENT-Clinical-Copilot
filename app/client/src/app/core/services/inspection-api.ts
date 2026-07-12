import { API_BASE_URL } from "../constants";
import {
  InspectionCatalogQuery,
  InspectionDeleteResponse,
  InspectionDrugAliasesResponse,
  InspectionLiverToxCatalogResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxOverrideRequest,
  InspectionRagDocumentsResponse,
  InspectionRagUpdateRequest,
  InspectionRagVectorStoreSummary,
  InspectionRxNavCatalogResponse,
  InspectionRxNavOverrideRequest,
  InspectionRxNavUpdateRequest,
  InspectionSessionCatalogResponse,
  InspectionSessionQuery,
  InspectionSessionTimeline,
  InspectionSessionTimelineListResponse,
  InspectionSessionTimelineRequest,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
  JobCancelResponse,
  JobStartResponse,
  ClinicalSessionDetail,
  ManualReportEditRequest,
  ManualReportEditResponse,
  ClinicalSessionUpdateRequest,
  RevisionArtifactListResponse,
  RevisionJobStatusResponse,
  RevisionPipelineStepListResponse,
  SessionRevisionRequest,
  RevisionClinicalReviewUpdateRequest,
} from "../models/types";
import { buildQueryString, requestJson } from "./http-api";

const TIMELINE_REQUEST_TIMEOUT_SECONDS = 360;
const INSPECTION_JOB_STATUS_TIMEOUT_SECONDS = 20;

export async function fetchInspectionSessions(
  query: InspectionSessionQuery,
): Promise<InspectionSessionCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    status: query.status,
    date_mode: query.date_mode,
    date: query.date,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionSessionCatalogResponse>(
    `${API_BASE_URL}/inspection/sessions${queryString}`,
    { method: "GET" },
  );
}

export async function fetchClinicalSessionDetail(
  sessionId: number,
): Promise<ClinicalSessionDetail> {
  return requestJson<ClinicalSessionDetail>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}`,
    { method: "GET" },
  );
}

export async function updateClinicalSession(
  sessionId: number,
  payload: ClinicalSessionUpdateRequest,
): Promise<ClinicalSessionDetail> {
  return requestJson<ClinicalSessionDetail>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchInspectionSessionTimelineList(
  sessionId: number,
): Promise<InspectionSessionTimelineListResponse> {
  return requestJson<InspectionSessionTimelineListResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timelines`,
    { method: "GET" },
    TIMELINE_REQUEST_TIMEOUT_SECONDS,
  );
}

export async function fetchInspectionSessionTimelineById(
  sessionId: number,
  timelineId: number,
): Promise<InspectionSessionTimeline> {
  return requestJson<InspectionSessionTimeline>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timelines/${encodeURIComponent(String(timelineId))}`,
    { method: "GET" },
    TIMELINE_REQUEST_TIMEOUT_SECONDS,
  );
}

export async function generateInspectionSessionTimeline(
  sessionId: number,
  payload: InspectionSessionTimelineRequest = {},
): Promise<InspectionSessionTimeline> {
  return requestJson<InspectionSessionTimeline>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timelines`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    TIMELINE_REQUEST_TIMEOUT_SECONDS,
  );
}

export async function deleteInspectionSession(
  sessionId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionRxNavCatalog(
  query: InspectionCatalogQuery,
): Promise<InspectionRxNavCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionRxNavCatalogResponse>(
    `${API_BASE_URL}/inspection/rxnav${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionRxNavAliases(
  drugId: number,
): Promise<InspectionDrugAliasesResponse> {
  return requestJson<InspectionDrugAliasesResponse>(
    `${API_BASE_URL}/inspection/rxnav/${encodeURIComponent(String(drugId))}/aliases`,
    { method: "GET" },
  );
}

export async function deleteInspectionRxNavDrug(
  drugId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/rxnav/${encodeURIComponent(String(drugId))}`,
    { method: "DELETE" },
  );
}

export async function updateInspectionRxNavDrug(
  drugId: number,
  payload: InspectionRxNavUpdateRequest,
): Promise<InspectionRxNavCatalogResponse["items"][number]> {
  return requestJson<InspectionRxNavCatalogResponse["items"][number]>(
    `${API_BASE_URL}/inspection/rxnav/${encodeURIComponent(String(drugId))}`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchInspectionRxNavUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/rxnav/update-config`, {
    method: "GET",
  });
}

export async function startInspectionRxNavUpdateJob(
  payload: InspectionRxNavOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rxnav/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionRxNavUpdateJobStatus(
  jobId: string,
  timeoutSeconds: number = INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelInspectionRxNavUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionLiverToxCatalog(
  query: InspectionCatalogQuery,
): Promise<InspectionLiverToxCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionLiverToxCatalogResponse>(
    `${API_BASE_URL}/inspection/livertox${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionLiverToxExcerpt(
  drugId: number,
): Promise<InspectionLiverToxExcerptResponse> {
  return requestJson<InspectionLiverToxExcerptResponse>(
    `${API_BASE_URL}/inspection/livertox/${encodeURIComponent(String(drugId))}/excerpt`,
    { method: "GET" },
  );
}

export async function deleteInspectionLiverToxDrug(
  drugId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/livertox/${encodeURIComponent(String(drugId))}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionLiverToxUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/livertox/update-config`, {
    method: "GET",
  });
}

export async function startInspectionLiverToxUpdateJob(
  payload: InspectionLiverToxOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/livertox/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionLiverToxUpdateJobStatus(
  jobId: string,
  timeoutSeconds: number = INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelInspectionLiverToxUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionRagUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/rag/update-config`, {
    method: "GET",
  });
}

export async function fetchInspectionRagDocuments(
  query: InspectionCatalogQuery,
): Promise<InspectionRagDocumentsResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionRagDocumentsResponse>(`${API_BASE_URL}/inspection/rag/documents${queryString}`, {
    method: "GET" },
  );
}

export async function fetchInspectionRagVectorStore(): Promise<InspectionRagVectorStoreSummary> {
  return requestJson<InspectionRagVectorStoreSummary>(`${API_BASE_URL}/inspection/rag/vector-store`, {
    method: "GET" },
  );
}

export async function startInspectionRagUpdateJob(
  payload: InspectionRagUpdateRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rag/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionRagUpdateJobStatus(
  jobId: string,
  timeoutSeconds: number = INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelInspectionRagUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function manualEditClinicalSessionReport(
  sessionId: number,
  payload: ManualReportEditRequest,
): Promise<ManualReportEditResponse> {
  return requestJson<ManualReportEditResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/report`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function startSessionRevisionJob(sessionId: number, payload: SessionRevisionRequest): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/revision/jobs`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
}

export async function fetchSessionRevisionJobStatus(jobId: string): Promise<RevisionJobStatusResponse> {
  return requestJson<RevisionJobStatusResponse>(`${API_BASE_URL}/inspection/sessions/revision/jobs/${encodeURIComponent(jobId)}`, { method: "GET" });
}

export async function cancelSessionRevisionJob(jobId: string): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(`${API_BASE_URL}/inspection/sessions/revision/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
}

export async function fetchRevisionPipelineSteps(pipelineRunId: string): Promise<RevisionPipelineStepListResponse> {
  return requestJson<RevisionPipelineStepListResponse>(`${API_BASE_URL}/inspection/sessions/revision/pipeline-runs/${encodeURIComponent(pipelineRunId)}/steps`, { method: "GET" });
}

export async function fetchRevisionArtifacts(sessionId: number, versionId: number): Promise<RevisionArtifactListResponse> {
  return requestJson<RevisionArtifactListResponse>(`${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/versions/${encodeURIComponent(String(versionId))}/artifacts`, { method: "GET" });
}
export async function updateRevisionClinicalReview(sessionId: number, versionId: number, payload: RevisionClinicalReviewUpdateRequest): Promise<unknown> {
  return requestJson<unknown>(`${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/versions/${encodeURIComponent(String(versionId))}/clinical-review`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
}
