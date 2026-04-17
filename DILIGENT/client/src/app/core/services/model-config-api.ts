import { API_BASE_URL } from "../constants";
import {
  AccessKeyProvider,
  AccessKeyRecord,
  ApiResult,
  JobStartResponse,
  JobStatusResponse,
  ModelConfigStateResponse,
  ModelConfigUpdateRequest,
  OllamaPullJobResult,
} from "../models/types";
import {
  buildQueryString,
  fetchWithTimeout,
  HTTP_TIMEOUT,
  normalizeThrownError,
  parseApiResponse,
  requestJson,
  GENERIC_REQUEST_ERROR,
} from "./http-api";

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
        return await parseApiResponse(response);
      }
    }
  } catch (error) {
    return {
      message: normalizeThrownError(error, GENERIC_REQUEST_ERROR),
      json: null,
    };
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
