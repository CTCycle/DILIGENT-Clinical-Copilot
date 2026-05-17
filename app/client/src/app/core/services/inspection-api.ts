import { API_BASE_URL } from "../constants";
import {
  InspectionCatalogQuery,
  InspectionDeleteResponse,
  InspectionDrugAliasesResponse,
  InspectionLiverToxCatalogResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxOverrideRequest,
  InspectionRagDocumentsResponse,
  InspectionRagOverrideRequest,
  InspectionRagVectorStoreSummary,
  InspectionRxNavCatalogResponse,
  InspectionRxNavOverrideRequest,
  InspectionSessionCatalogResponse,
  InspectionSessionQuery,
  InspectionSessionTimeline,
  InspectionSessionTimelineRequest,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
  JobCancelResponse,
  JobStartResponse,
  ClinicalSessionDetail,
  ClinicalSessionRevisionRequest,
  ClinicalSessionUpdateRequest,
} from "../models/types";
import { buildQueryString, requestJson } from "./http-api";

const TIMELINE_REQUEST_TIMEOUT_SECONDS = 120;

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

export async function startClinicalSessionRevisionJob(
  sessionId: number,
  payload: ClinicalSessionRevisionRequest,
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/revision/jobs`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchClinicalSessionRevisionJobStatus(
  jobId: string,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/sessions/revision/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelClinicalSessionRevisionJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/sessions/revision/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionSessionTimeline(
  sessionId: number,
): Promise<InspectionSessionTimeline> {
  return requestJson<InspectionSessionTimeline>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timeline`,
    { method: "GET" },
    TIMELINE_REQUEST_TIMEOUT_SECONDS,
  );
}

export async function generateInspectionSessionTimeline(
  sessionId: number,
  payload: InspectionSessionTimelineRequest = {},
): Promise<InspectionSessionTimeline> {
  return requestJson<InspectionSessionTimeline>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timeline`,
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
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
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
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
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
  payload: InspectionRagOverrideRequest = {},
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
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelInspectionRagUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rag/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST" },
  );
}
