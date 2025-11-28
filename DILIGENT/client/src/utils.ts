import { CLOUD_MODEL_CHOICES, CLOUD_PROVIDERS } from "./constants";
import {
  ClinicalFormState,
  ClinicalRequestPayload,
  CloudSelection,
} from "./types";

export function sanitizeField(value: string): string | null {
  const normalized = value.trim();
  return normalized.length ? normalized : null;
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

export function resolveCloudSelection(
  provider: string,
  cloudModel: string | null,
): CloudSelection {
  const normalizedProvider = provider.trim().toLowerCase();
  const providerChoice = CLOUD_MODEL_CHOICES[normalizedProvider]
    ? normalizedProvider
    : CLOUD_PROVIDERS[0] ?? "";
  const models = CLOUD_MODEL_CHOICES[providerChoice] ?? [];
  const chosenModel =
    cloudModel && models.includes(cloudModel) ? cloudModel : models[0] ?? null;

  return {
    provider: providerChoice,
    models,
    model: chosenModel,
  };
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

export function buildClinicalPayload(
  form: ClinicalFormState,
): ClinicalRequestPayload {
  return {
    name: sanitizeField(form.patientName),
    visit_date: buildVisitDatePayload(form.visitDate),
    anamnesis: sanitizeField(form.anamnesis),
    drugs: sanitizeField(form.drugs),
    alt: sanitizeField(form.alt),
    alt_max: sanitizeField(form.altMax),
    alp: sanitizeField(form.alp),
    alp_max: sanitizeField(form.alpMax),
    use_rag: form.useRag,
  };
}

export function createDownloadUrl(content: string, filename: string): string {
  const blob = new Blob([content], { type: "text/markdown" });
  return URL.createObjectURL(blob);
}
