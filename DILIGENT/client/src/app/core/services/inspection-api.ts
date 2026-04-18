import { API_BASE_URL } from "../constants";
import {
  InspectionCatalogQuery,
  InspectionDeleteResponse,
  InspectionDiliPriorCatalogResponse,
  InspectionDiliPriorDetailResponse,
  InspectionDiliPriorsOverrideRequest,
  InspectionDrugAliasesResponse,
  InspectionDrugLabelCatalogResponse,
  InspectionDrugLabelSectionsResponse,
  InspectionDrugLabelsOverrideRequest,
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
  InspectionSessionReportResponse,
  InspectionSessionTimeline,
  InspectionSessionTimelineRequest,
  InspectionUpdateConfigResponse,
  InspectionUpdateJobStatusResponse,
  JobCancelResponse,
  JobStartResponse,
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

export async function fetchInspectionSessionReport(
  sessionId: number,
): Promise<InspectionSessionReportResponse> {
  return requestJson<InspectionSessionReportResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/report`,
    { method: "GET" },
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

export async function fetchInspectionDiliPriorsCatalog(
  query: InspectionCatalogQuery,
): Promise<InspectionDiliPriorCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionDiliPriorCatalogResponse>(
    `${API_BASE_URL}/inspection/dili-priors${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionDiliPriorDetails(
  drugId: number,
): Promise<InspectionDiliPriorDetailResponse> {
  return requestJson<InspectionDiliPriorDetailResponse>(
    `${API_BASE_URL}/inspection/dili-priors/${encodeURIComponent(String(drugId))}`,
    { method: "GET" },
  );
}

export async function fetchInspectionDiliPriorsUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/dili-priors/update-config`, {
    method: "GET",
  });
}

export async function startInspectionDiliPriorsUpdateJob(
  payload: InspectionDiliPriorsOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/dili-priors/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionDiliPriorsUpdateJobStatus(
  jobId: string,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/dili-priors/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelInspectionDiliPriorsUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/dili-priors/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchInspectionDrugLabelsCatalog(
  query: InspectionCatalogQuery,
): Promise<InspectionDrugLabelCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionDrugLabelCatalogResponse>(
    `${API_BASE_URL}/inspection/drug-labels${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionDrugLabelSections(
  drugId: number,
): Promise<InspectionDrugLabelSectionsResponse> {
  return requestJson<InspectionDrugLabelSectionsResponse>(
    `${API_BASE_URL}/inspection/drug-labels/${encodeURIComponent(String(drugId))}/sections`,
    { method: "GET" },
  );
}

export async function fetchInspectionDrugLabelsUpdateConfig(): Promise<InspectionUpdateConfigResponse> {
  return requestJson<InspectionUpdateConfigResponse>(`${API_BASE_URL}/inspection/drug-labels/update-config`, {
    method: "GET",
  });
}

export async function startInspectionDrugLabelsUpdateJob(
  payload: InspectionDrugLabelsOverrideRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/drug-labels/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchInspectionDrugLabelsUpdateJobStatus(
  jobId: string,
): Promise<InspectionUpdateJobStatusResponse> {
  return requestJson<InspectionUpdateJobStatusResponse>(
    `${API_BASE_URL}/inspection/drug-labels/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelInspectionDrugLabelsUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/drug-labels/jobs/${encodeURIComponent(jobId)}`,
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
