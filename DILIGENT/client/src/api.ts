import { HTTP_TIMEOUT_SECONDS } from "./constants";
import { ApiResult, ClinicalRequestPayload } from "./types";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const HTTP_TIMEOUT =
  Number.parseFloat(import.meta.env.VITE_HTTP_TIMEOUT ?? "") ||
  HTTP_TIMEOUT_SECONDS;

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

  if (typeof data === "object") {
    const candidates = ["output", "result", "text", "message", "response"];
    for (const key of candidates) {
      const value = (data as Record<string, unknown>)[key];
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

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutMs = Math.max(HTTP_TIMEOUT, 1) * 1000;
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

async function parseApiResponse(
  response: Response,
  url: string,
): Promise<ApiResult> {
  const contentType = response.headers.get("content-type") || "";
  const bodyText = await response.text();
  let parsed: unknown | undefined;

  if (bodyText) {
    const shouldTryJson =
      contentType.includes("application/json") ||
      bodyText.trimStart().startsWith("{") ||
      bodyText.trimStart().startsWith("[");
    if (shouldTryJson) {
      try {
        parsed = JSON.parse(bodyText);
      } catch {
        parsed = undefined;
      }
    }
  }

  if (parsed !== undefined) {
    const message = extractTextFromResult(parsed);
    const jsonPayload =
      typeof parsed === "object" || Array.isArray(parsed) ? parsed : null;

    if (response.ok) {
      return { message, json: jsonPayload };
    }

    const errorPrefix = `[ERROR] Backend returned status ${response.status}.`;
    const combined = message ? `${errorPrefix}\n${message}` : errorPrefix;
    return { message: combined, json: jsonPayload };
  }

  if (response.ok) {
    return { message: bodyText.trim(), json: null };
  }

  const errorMessage = `[ERROR] Backend returned status ${response.status}.${
    bodyText ? `\n${bodyText}` : `\nURL: ${url}`
  }`;
  return { message: errorMessage, json: null };
}

export async function runClinicalSession(
  payload: ClinicalRequestPayload,
): Promise<ApiResult> {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/clinical`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    return await parseApiResponse(response, `${API_BASE_URL}/clinical`);
  } catch (error) {
    const description =
      error instanceof Error ? error.message : "Unexpected error";
    return { message: `[ERROR] ${description}`, json: null };
  }
}

export async function pullModels(models: string[]): Promise<ApiResult> {
  const selected = Array.from(new Set(models.filter((item) => !!item)));
  if (!selected.length) {
    return { message: "[ERROR] No models selected to pull.", json: null };
  }

  try {
    for (const model of selected) {
      const response = await fetchWithTimeout(
        `${API_BASE_URL}/models/pull?name=${encodeURIComponent(
          model,
        )}&stream=false`,
        { method: "GET" },
      );

      if (!response.ok) {
        return await parseApiResponse(
          response,
          `${API_BASE_URL}/models/pull?name=${model}`,
        );
      }
    }
  } catch (error) {
    const description =
      error instanceof Error ? error.message : "Unexpected error";
    return { message: `[ERROR] ${description}`, json: null };
  }

  return {
    message: `[INFO] Models available locally: ${selected.join(", ")}.`,
    json: null,
  };
}
