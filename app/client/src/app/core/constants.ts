// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ClinicalFormState } from "./models/types";

declare const __DILIGENT_API_BASE_URL__: string | undefined;

export const API_BASE_URL =
  typeof __DILIGENT_API_BASE_URL__ === "string" &&
  __DILIGENT_API_BASE_URL__.trim()
    ? __DILIGENT_API_BASE_URL__.replace(/\/+$/, "")
    : "/api";

export const APP_LOCALE = "en-GB";

export const DEFAULT_FORM_STATE: ClinicalFormState = {
  patientName: "",
  visitDate: "",
  patientImageDataUrl: null,
  clinicalInput: "",
  useRag: false,
};

export const REPORT_EXPORT_FILENAME = "clinical_report.md";

export const HTTP_TIMEOUT_SECONDS = 3600;
