// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ClinicalFormState } from "./models/types";

declare const __DILIGENT_API_BASE_URL__: string | undefined;

const BUILD_API_BASE_URL =
  typeof __DILIGENT_API_BASE_URL__ === "string" &&
  __DILIGENT_API_BASE_URL__.trim()
    ? __DILIGENT_API_BASE_URL__.replace(/\/+$/, "")
    : "/api";
export let API_BASE_URL = BUILD_API_BASE_URL;

export function configureDesktopApiOrigin(origin: string | null): void {
  if (origin === null) {
    API_BASE_URL = BUILD_API_BASE_URL;
    return;
  }
  let parsed: URL;
  try {
    parsed = new URL(origin);
  } catch {
    throw new Error("Desktop API origin is invalid.");
  }
  if (
    parsed.protocol !== "http:" ||
    parsed.hostname !== "127.0.0.1" ||
    !parsed.port ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error("Desktop API origin must be a local loopback origin.");
  }
  API_BASE_URL = `${parsed.origin}/api`;
}

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
