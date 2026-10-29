// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from "../constants";
import {
  TerminologyAliasDecision,
  TerminologyAliasDecisionResponse,
  TerminologyConflictsResponse,
  TerminologyImportRunItem,
  TerminologyResolution,
  TerminologySummaryResponse,
} from "../models/inspection-types";
import { buildQueryString, requestJson } from "./http-api";

export async function fetchTerminologySummary(): Promise<TerminologySummaryResponse> {
  return requestJson<TerminologySummaryResponse>(
    `${API_BASE_URL}/inspection/terminology/summary`,
    { method: "GET" },
  );
}

export async function fetchTerminologyRuns(limit = 20): Promise<TerminologyImportRunItem[]> {
  const response = await requestJson<{ items: TerminologyImportRunItem[] }>(
    `${API_BASE_URL}/inspection/terminology/runs${buildQueryString({ limit })}`,
    { method: "GET" },
  );
  return response.items;
}

export async function fetchTerminologyConflicts(
  offset = 0,
  limit = 20,
): Promise<TerminologyConflictsResponse> {
  return requestJson<TerminologyConflictsResponse>(
    `${API_BASE_URL}/inspection/terminology/conflicts${buildQueryString({ status: "open", offset, limit })}`,
    { method: "GET" },
  );
}

export async function resolveTerminologyName(
  name: string,
  allowExternal = true,
): Promise<TerminologyResolution> {
  return requestJson<TerminologyResolution>(`${API_BASE_URL}/inspection/terminology/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, allow_external: allowExternal }),
  });
}

export async function decideTerminologyAlias(
  decision: TerminologyAliasDecision,
  alias: string,
  drugId: number,
): Promise<TerminologyAliasDecisionResponse> {
  return requestJson<TerminologyAliasDecisionResponse>(
    `${API_BASE_URL}/inspection/terminology/aliases/${decision}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alias, drug_id: drugId }),
    },
  );
}
