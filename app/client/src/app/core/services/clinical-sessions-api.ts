// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from "../constants";
import {
  ClinicalSessionAttachment,
  ClinicalSessionDetail,
  ClinicalSessionUpdateRequest,
  InspectionDeleteResponse,
  InspectionSessionCatalogResponse,
  InspectionSessionQuery,
  ManualReportEditRequest,
  ManualReportEditResponse,
} from "../models/inspection-types";
import {
  ApiHttpError,
  GENERIC_REQUEST_ERROR,
  buildQueryString,
  fetchWithTimeout,
  parseApiResponse,
  requestJson,
} from "./http-api";

export async function fetchInspectionSessions(
  query: InspectionSessionQuery,
): Promise<InspectionSessionCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    status: query.statuses,
    sort: query.sort,
    date_field: query.date_field,
    date_from: query.date_from,
    date_to: query.date_to,
    created_before: query.created_before,
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

export async function deleteInspectionSession(
  sessionId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}`,
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

function sessionAttachmentsUrl(sessionId: number): string {
  return `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/attachments`;
}

export async function fetchSessionAttachmentBlob(
  sessionId: number,
  attachmentId: number,
): Promise<Blob> {
  const response = await fetchWithTimeout(
    `${sessionAttachmentsUrl(sessionId)}/${encodeURIComponent(String(attachmentId))}`,
    { method: "GET" },
  );
  if (!response.ok) {
    const result = await parseApiResponse(response);
    throw new ApiHttpError(result.message || GENERIC_REQUEST_ERROR, response.status);
  }
  return response.blob();
}

export async function fetchSessionAttachments(
  sessionId: number,
): Promise<ClinicalSessionAttachment[]> {
  const response = await requestJson<{ items: ClinicalSessionAttachment[] }>(
    sessionAttachmentsUrl(sessionId),
    { method: "GET" },
  );
  return response.items;
}

export async function uploadSessionAttachment(
  sessionId: number,
  file: File,
): Promise<ClinicalSessionAttachment> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return requestJson<ClinicalSessionAttachment>(sessionAttachmentsUrl(sessionId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_name: file.name,
      media_type: file.type || "application/octet-stream",
      content_base64: btoa(binary),
    }),
  });
}

export async function deleteSessionAttachment(
  sessionId: number,
  attachmentId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${sessionAttachmentsUrl(sessionId)}/${encodeURIComponent(String(attachmentId))}`,
    { method: "DELETE" },
  );
}
