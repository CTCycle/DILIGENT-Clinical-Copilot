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
import { buildQueryString, requestJson } from "./http-api";

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

export function sessionAttachmentContentUrl(
  sessionId: number,
  attachmentId: number,
): string {
  return `${sessionAttachmentsUrl(sessionId)}/${encodeURIComponent(String(attachmentId))}`;
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
