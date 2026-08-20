import { API_BASE_URL } from "../constants";
import {
  InspectionCatalogQuery,
  InspectionDeleteResponse,
  InspectionDrugAliasesResponse,
  InspectionLiverToxCatalogResponse,
  InspectionLiverToxExcerptResponse,
  InspectionRagDocumentsResponse,
  InspectionRagVectorStoreSummary,
  InspectionRxNavCatalogResponse,
  InspectionRxNavUpdateRequest,
} from "../models/types";
import { buildQueryString, requestJson } from "./http-api";

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
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
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

export async function fetchInspectionRagDocuments(
  query: InspectionCatalogQuery,
): Promise<InspectionRagDocumentsResponse> {
  const queryString = buildQueryString({
    search: query.search,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionRagDocumentsResponse>(
    `${API_BASE_URL}/inspection/rag/documents${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionRagVectorStore(): Promise<InspectionRagVectorStoreSummary> {
  return requestJson<InspectionRagVectorStoreSummary>(
    `${API_BASE_URL}/inspection/rag/vector-store`,
    { method: "GET" },
  );
}
