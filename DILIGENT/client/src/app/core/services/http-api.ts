import { HTTP_TIMEOUT_SECONDS } from "../constants";
import { ApiResult } from "../models/types";

export const HTTP_TIMEOUT = HTTP_TIMEOUT_SECONDS;
export const GENERIC_REQUEST_ERROR = "[ERROR] Request could not be completed. Please try again.";
export const TIMEOUT_REQUEST_ERROR = "[ERROR] Request timed out. Please retry.";
export const NETWORK_REQUEST_ERROR =
  "[ERROR] Network is unavailable. Check your connection and retry.";

const SENSITIVE_ERROR_FRAGMENTS = [
  "traceback",
  "stack",
  "exception",
  "token",
  "secret",
  "password",
  "authorization",
  "api key",
  "access key",
  "sql",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeUserFacingMessage(message: string): boolean {
  const candidate = message.trim();
  if (!candidate || candidate.length > 180) {
    return false;
  }
  const lowered = candidate.toLowerCase();
  return !SENSITIVE_ERROR_FRAGMENTS.some((fragment) => lowered.includes(fragment));
}

function extractServerDetail(data: unknown): string {
  if (!isRecord(data)) {
    return "";
  }
  const detail = data["detail"];
  if (typeof detail === "string") {
    return detail.trim();
  }
  if (Array.isArray(detail)) {
    const firstIssue = detail.find((item) => isRecord(item));
    if (firstIssue && typeof firstIssue["msg"] === "string") {
      return firstIssue["msg"].trim();
    }
  }
  return "";
}

function buildHttpErrorMessage(statusCode: number, detail: string): string {
  if (statusCode === 404) {
    return "[ERROR] Requested data was not found.";
  }
  if (statusCode === 408 || statusCode === 504) {
    return TIMEOUT_REQUEST_ERROR;
  }
  if (statusCode === 409) {
    return "[ERROR] Another operation is already running. Please wait and retry.";
  }
  if (statusCode === 422 || statusCode === 400) {
    if (isSafeUserFacingMessage(detail)) {
      return `[ERROR] ${detail}`;
    }
    return "[ERROR] Some inputs are invalid. Review the data and retry.";
  }
  if (statusCode === 429) {
    return "[ERROR] Service is busy. Wait a moment and retry.";
  }
  if (statusCode === 502 || statusCode === 503) {
    return "[ERROR] A required service is unavailable. Please retry shortly.";
  }
  if (statusCode >= 500) {
    return "[ERROR] Service is temporarily unavailable. Please retry.";
  }
  if (isSafeUserFacingMessage(detail)) {
    return `[ERROR] ${detail}`;
  }
  return GENERIC_REQUEST_ERROR;
}

export function normalizeThrownError(error: unknown, fallback: string): string {
  if (error instanceof DOMException && error.name === "AbortError") {
    return TIMEOUT_REQUEST_ERROR;
  }
  if (error instanceof TypeError) {
    return NETWORK_REQUEST_ERROR;
  }
  if (error instanceof Error && isSafeUserFacingMessage(error.message)) {
    return error.message.startsWith("[ERROR]")
      ? error.message
      : `[ERROR] ${error.message}`;
  }
  return fallback;
}

function extractTextFromResult(data: unknown): string {
  if (!data) {
    return "";
  }

  if (typeof data === "string") {
    return data;
  }

  if (Array.isArray(data)) {
    try {
      return `\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``;
    } catch {
      return `${data}`;
    }
  }

  if (isRecord(data)) {
    const candidates = ["output", "result", "text", "message", "response", "detail", "error"];
    for (const key of candidates) {
      const value = data[key];
      if (typeof value === "string" && value.trim()) {
        return value;
      }
    }
    try {
      return `\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``;
    } catch {
      return `${data}`;
    }
  }

  return `${data}`;
}

export async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutMs = Math.max(timeoutSeconds, 1) * 1000;
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    throw new Error(normalizeThrownError(error, GENERIC_REQUEST_ERROR));
  } finally {
    globalThis.clearTimeout(timer);
  }
}

export async function parseApiResponse(response: Response): Promise<ApiResult> {
  const contentType = response.headers.get("content-type") || "";
  const bodyText = await response.text();
  let parsed: unknown;
  let hasParsed = false;

  if (bodyText) {
    const shouldTryJson =
      contentType.includes("application/json") ||
      bodyText.trimStart().startsWith("{") ||
      bodyText.trimStart().startsWith("[");
    if (shouldTryJson) {
      try {
        parsed = JSON.parse(bodyText);
        hasParsed = true;
      } catch {
        hasParsed = false;
      }
    }
  }

  if (hasParsed) {
    const message = extractTextFromResult(parsed);
    const jsonPayload =
      typeof parsed === "object" || Array.isArray(parsed) ? parsed : null;

    if (response.ok) {
      return { message, json: jsonPayload };
    }

    const detail = extractServerDetail(parsed) || message;
    return {
      message: buildHttpErrorMessage(response.status, detail),
      json: jsonPayload,
    };
  }

  if (response.ok) {
    return { message: bodyText.trim(), json: null };
  }

  return {
    message: buildHttpErrorMessage(response.status, bodyText.trim()),
    json: null,
  };
}

export async function requestJson<T>(
  url: string,
  options: RequestInit,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<T> {
  const response = await fetchWithTimeout(url, options, timeoutSeconds);
  const result = await parseApiResponse(response);
  if (!response.ok) {
    throw new Error(result.message || GENERIC_REQUEST_ERROR);
  }
  if (!result.json) {
    throw new Error("[ERROR] Service returned an invalid response. Please retry.");
  }
  return JSON.parse(JSON.stringify(result.json));
}

export function buildQueryString(
  params: Record<string, string | number | null | undefined>,
): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) {
      continue;
    }
    const serialized = `${value}`.trim();
    if (!serialized) {
      continue;
    }
    searchParams.set(key, serialized);
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : "";
}
