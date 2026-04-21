import {
  ClinicalFormState,
  ClinicalRequestPayload,
  RuntimeSettings,
} from "./models/types";

const DRUG_TIMING_CUE_RE =
  /\b(?:\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?|\d{4}[./-]\d{1,2}[./-]\d{1,2}|start(?:ed|ing)?|begin|began|since|from|on|until|stop(?:ped|ping)?|suspend(?:ed|ing)?|discontinu(?:e|ed)|interrott|sospes|inizi|avviat|ripres|terapia|treatment)\b/i;
const DRUG_SCHEDULE_CUE_RE = /\b\d+(?:[.,]\d+)?\s*-\s*\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?){1,2}\b/;

export function sanitizeField(value: string): string | null {
  const normalized = value.trim();
  return normalized.length ? normalized : null;
}

export function formatErrorMessage(message: string): string {
  const normalized = message.trim();
  if (!normalized) {
    return "[ERROR] Unexpected error";
  }
  return normalized.startsWith("[ERROR]") ? normalized : `[ERROR] ${normalized}`;
}

export function formatUnknownError(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    return formatErrorMessage(error.message);
  }
  return formatErrorMessage(fallback);
}

export function normalizeVisitDateInput(value: string): string {
  const normalized = value.trim();
  if (!normalized) {
    return "";
  }

  const parts = normalized.split("-");
  if (parts.length !== 3) {
    return "";
  }

  const [yearRaw, monthRaw, dayRaw] = parts;
  const year = Number.parseInt(yearRaw, 10);
  const month = Number.parseInt(monthRaw, 10);
  const day = Number.parseInt(dayRaw, 10);
  if (
    Number.isNaN(year) ||
    Number.isNaN(month) ||
    Number.isNaN(day) ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > 31
  ) {
    return "";
  }

  const candidate = new Date(Date.UTC(year, month - 1, day));
  if (Number.isNaN(candidate.getTime())) {
    return "";
  }

  const today = new Date();
  const todayUtc = new Date(
    Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()),
  );

  if (candidate > todayUtc) {
    return todayUtc.toISOString().slice(0, 10);
  }

  return `${year.toString().padStart(4, "0")}-${month
    .toString()
    .padStart(2, "0")}-${day.toString().padStart(2, "0")}`;
}

function buildVisitDatePayload(
  visitDate: string,
): { day: number; month: number; year: number } | null {
  const normalized = normalizeVisitDateInput(visitDate);
  if (!normalized) {
    return null;
  }

  const [yearRaw, monthRaw, dayRaw] = normalized.split("-");
  const year = Number.parseInt(yearRaw, 10);
  const month = Number.parseInt(monthRaw, 10);
  const day = Number.parseInt(dayRaw, 10);

  if (
    Number.isNaN(year) ||
    Number.isNaN(month) ||
    Number.isNaN(day) ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > 31
  ) {
    return null;
  }

  return { day, month, year };
}

function extractBase64Payload(dataUrl: string | null): string | null {
  if (!dataUrl) {
    return null;
  }
  const normalized = dataUrl.trim();
  if (!normalized) {
    return null;
  }
  if (normalized.startsWith("data:") && normalized.includes(",")) {
    return normalized.split(",", 2)[1]?.trim() || null;
  }
  return normalized;
}

export function buildClinicalPayload(
  form: ClinicalFormState,
  _settings: RuntimeSettings,
  allowMissingLabs: boolean | null = null,
): ClinicalRequestPayload {
  const payload: ClinicalRequestPayload = {
    name: sanitizeField(form.patientName),
    visit_date: buildVisitDatePayload(form.visitDate),
    anamnesis: sanitizeField(form.anamnesis),
    drugs: sanitizeField(form.drugs),
    laboratory_analysis: sanitizeField(form.laboratoryAnalysis),
    patient_image_base64: extractBase64Payload(form.patientImageDataUrl),
    use_rag: form.useRag,
    use_web_search: form.useWebSearch,
  };
  if (allowMissingLabs !== null) {
    payload.allow_missing_labs = allowMissingLabs;
  }
  return payload;
}

export function hasDrugTimingCue(value: string): boolean {
  const normalized = value.trim();
  if (!normalized) {
    return false;
  }
  if (DRUG_SCHEDULE_CUE_RE.test(normalized)) {
    return true;
  }
  return DRUG_TIMING_CUE_RE.test(normalized);
}

export function createDownloadUrl(content: string, filename: string): string {
  const blob = new Blob([content], { type: "text/markdown" });
  return URL.createObjectURL(blob);
}


