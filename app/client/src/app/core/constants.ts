import { ClinicalFormState } from "./models/types";

export const API_BASE_URL = "/api";

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
