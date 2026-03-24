import { API_BASE_URL, HTTP_TIMEOUT_SECONDS } from "../constants";
import {
  AccessKeyProvider,
  AccessKeyRecord,
  ApiResult,
  ClinicalRequestPayload,
  InspectionCatalogQuery,
  InspectionDeleteResponse,
  InspectionDrugAliasesResponse,
  InspectionLiverToxCatalogResponse,
  InspectionLiverToxExcerptResponse,
  InspectionRxNavCatalogResponse,
  InspectionSessionCatalogResponse,
  InspectionSessionQuery,
  InspectionSessionReportResponse,
  JobCancelResponse,
  JobStartResponse,
  JobStatusResponse,
  ModelConfigStateResponse,
  ModelConfigUpdateRequest,
  OllamaPullJobResult,
} from "../types";

const HTTP_TIMEOUT =
  Number.parseFloat(import.meta.env.VITE_HTTP_TIMEOUT ?? "") ||
  HTTP_TIMEOUT_SECONDS;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
    const candidates = ["output", "result", "text", "message", "response"];
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

async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutMs = Math.max(timeoutSeconds, 1) * 1000;
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    globalThis.clearTimeout(timer);
  }
}

async function parseApiResponse(
  response: Response,
  url: string,
): Promise<ApiResult> {
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

    const errorPrefix = `[ERROR] Backend returned status ${response.status}.`;
    const combined = message ? `${errorPrefix}\n${message}` : errorPrefix;
    return { message: combined, json: jsonPayload };
  }

  if (response.ok) {
    return { message: bodyText.trim(), json: null };
  }

  const errorDetails = bodyText ? `\n${bodyText}` : `\nURL: ${url}`;
  const errorMessage = `[ERROR] Backend returned status ${response.status}.${errorDetails}`;
  return { message: errorMessage, json: null };
}

async function requestJson<T>(
  url: string,
  options: RequestInit,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<T> {
  const response = await fetchWithTimeout(url, options, timeoutSeconds);
  const result = await parseApiResponse(response, url);
  if (!response.ok) {
    throw new Error(result.message || `[ERROR] Request failed: ${response.status}`);
  }
  if (!result.json) {
    throw new Error("[ERROR] Expected JSON payload from backend.");
  }
  return result.json as T;
}

function buildQueryString(
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

export async function startClinicalJob(
  payload: ClinicalRequestPayload,
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/clinical/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchClinicalJobStatus(
  jobId: string,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(
    `${API_BASE_URL}/clinical/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
}

export async function cancelClinicalJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/clinical/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export function pollClinicalJobStatus(
  jobId: string,
  intervalMs: number,
  onUpdate: (status: JobStatusResponse) => void,
  onError: (message: string) => void,
): { stop: () => void } {
  const safeIntervalMs = Math.max(intervalMs, 250);
  const requestTimeoutSeconds = Math.min(
    30,
    Math.max(5, Math.ceil((safeIntervalMs / 1000) * 4)),
  );
  const maxConsecutivePollErrors = 3;
  let timeoutId: ReturnType<typeof globalThis.setTimeout> | null = null;
  let stopped = false;
  let consecutivePollErrors = 0;

  const poll = async () => {
    if (stopped) return;
    try {
      const status = await fetchClinicalJobStatus(jobId, requestTimeoutSeconds);
      if (stopped) return;
      consecutivePollErrors = 0;
      onUpdate(status);
      if (
        status.status === "completed" ||
        status.status === "failed" ||
        status.status === "cancelled"
      ) {
        return;
      }
    } catch (error) {
      if (stopped) return;
      consecutivePollErrors += 1;
      if (consecutivePollErrors < maxConsecutivePollErrors) {
        timeoutId = globalThis.setTimeout(poll, safeIntervalMs);
        return;
      }
      const message =
        error instanceof Error ? error.message : "Unexpected polling error";
      onError(
        `Polling failed after ${maxConsecutivePollErrors} attempts: ${message}`,
      );
      return;
    }
    if (stopped) return;
    timeoutId = globalThis.setTimeout(poll, safeIntervalMs);
  };

  poll();

  return {
    stop: () => {
      stopped = true;
      if (timeoutId !== null) {
        globalThis.clearTimeout(timeoutId);
      }
    },
  };
}

export async function fetchInspectionSessions(
  query: InspectionSessionQuery,
): Promise<InspectionSessionCatalogResponse> {
  const queryString = buildQueryString({
    search: query.search,
    status: query.status,
    date_mode: query.date_mode,
    date: query.date,
    offset: query.offset ?? 0,
    limit: query.limit ?? 10,
  });
  return requestJson<InspectionSessionCatalogResponse>(
    `${API_BASE_URL}/inspection/sessions${queryString}`,
    { method: "GET" },
  );
}

export async function fetchInspectionSessionReport(
  sessionId: number,
): Promise<InspectionSessionReportResponse> {
  return requestJson<InspectionSessionReportResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/report`,
    { method: "GET" },
  );
}

export async function deleteInspectionSession(
  sessionId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}`,
    { method: "DELETE" },
  );
}

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

export async function startInspectionRxNavUpdateJob(): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/rxnav/jobs`, {
    method: "POST",
  });
}

export async function fetchInspectionRxNavUpdateJobStatus(
  jobId: string,
): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelInspectionRxNavUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/rxnav/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
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

export async function startInspectionLiverToxUpdateJob(): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(`${API_BASE_URL}/inspection/livertox/jobs`, {
    method: "POST",
  });
}

export async function fetchInspectionLiverToxUpdateJobStatus(
  jobId: string,
): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelInspectionLiverToxUpdateJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/livertox/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchModelConfigState(): Promise<ModelConfigStateResponse> {
  return requestJson<ModelConfigStateResponse>(`${API_BASE_URL}/model-config`, {
    method: "GET",
  });
}

export async function updateModelConfigState(
  payload: ModelConfigUpdateRequest,
): Promise<ModelConfigStateResponse> {
  return requestJson<ModelConfigStateResponse>(`${API_BASE_URL}/model-config`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function startModelPullJob(
  modelName: string,
): Promise<JobStartResponse> {
  const candidate = modelName.trim();
  if (!candidate) {
    throw new Error("[ERROR] Enter a model name to pull from Ollama.");
  }
  const queryString = buildQueryString({ name: candidate, stream: "true" });
  return requestJson<JobStartResponse>(
    `${API_BASE_URL}/models/pull/jobs${queryString}`,
    { method: "POST" },
  );
}

export async function fetchModelPullJobStatus(
  jobId: string,
  timeoutSeconds: number = HTTP_TIMEOUT,
): Promise<JobStatusResponse<OllamaPullJobResult>> {
  return requestJson<JobStatusResponse<OllamaPullJobResult>>(
    `${API_BASE_URL}/models/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
    timeoutSeconds,
  );
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

export async function fetchAccessKeys(
  provider: AccessKeyProvider,
): Promise<AccessKeyRecord[]> {
  const encodedProvider = encodeURIComponent(provider);
  return requestJson<AccessKeyRecord[]>(
    `${API_BASE_URL}/access-keys?provider=${encodedProvider}`,
    {
      method: "GET",
    },
  );
}

export async function createAccessKey(
  provider: AccessKeyProvider,
  accessKey: string,
): Promise<AccessKeyRecord> {
  return requestJson<AccessKeyRecord>(`${API_BASE_URL}/access-keys`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ provider, access_key: accessKey }),
  });
}

export async function activateAccessKey(
  id: number,
  provider?: AccessKeyProvider,
): Promise<AccessKeyRecord> {
  const providerQuery = provider
    ? `?provider=${encodeURIComponent(provider)}`
    : "";
  return requestJson<AccessKeyRecord>(
    `${API_BASE_URL}/access-keys/${encodeURIComponent(String(id))}/activate${providerQuery}`,
    {
      method: "PUT",
    },
  );
}

export async function deleteAccessKey(
  id: number,
  provider?: AccessKeyProvider,
): Promise<void> {
  const providerQuery = provider
    ? `?provider=${encodeURIComponent(provider)}`
    : "";
  await requestJson<{ status: string; deleted: boolean }>(
    `${API_BASE_URL}/access-keys/${encodeURIComponent(String(id))}${providerQuery}`,
    {
      method: "DELETE",
    },
  );
}
