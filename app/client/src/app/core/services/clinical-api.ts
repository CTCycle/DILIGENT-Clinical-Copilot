// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from "../constants";
import {
  ClinicalInputPreflightResponse,
  ClinicalSectionTemplateResponse,
  ClinicalRequestPayload,
} from "../models/types";
import { requestJson } from "./http-api";

export async function validateClinicalInput(
  payload: ClinicalRequestPayload,
): Promise<ClinicalInputPreflightResponse> {
  return requestJson<ClinicalInputPreflightResponse>(
    `${API_BASE_URL}/clinical/validate-input`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchClinicalSectionTemplate(): Promise<ClinicalSectionTemplateResponse> {
  return requestJson<ClinicalSectionTemplateResponse>(
    `${API_BASE_URL}/clinical/section-template`,
    { method: "GET" },
  );
}

export function resolvePollIntervalMs(pollIntervalSeconds: number): number {
  if (!Number.isFinite(pollIntervalSeconds) || pollIntervalSeconds <= 0) {
    return 1000;
  }
  return Math.max(250, Math.round(pollIntervalSeconds * 1000));
}
