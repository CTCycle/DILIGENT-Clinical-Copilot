import { API_BASE_URL } from "../constants";
import {
  AccessKeyProvider,
  AccessKeyRecord,
  JobStartResponse,
  JobStatusResponse,
  ModelConfigStateResponse,
  ModelConfigUpdateRequest,
  OllamaPullJobResult,
} from "../models/types";
import {
  buildQueryString,
  HTTP_TIMEOUT,
  requestJson,
} from "./http-api";

const ACCESS_KEYS_TIMEOUT_SECONDS = 15;

export async function fetchModelConfigState(
  includeLocalAvailability?: boolean,
): Promise<ModelConfigStateResponse> {
  const queryString =
    typeof includeLocalAvailability === "boolean"
      ? buildQueryString({
          include_local_availability: includeLocalAvailability ? "true" : "false",
        })
      : "";
  return requestJson<ModelConfigStateResponse>(`${API_BASE_URL}/model-config${queryString}`, {
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

export async function fetchAccessKeys(
  provider: AccessKeyProvider,
): Promise<AccessKeyRecord[]> {
  const encodedProvider = encodeURIComponent(provider);
  return requestJson<AccessKeyRecord[]>(
    `${API_BASE_URL}/access-keys?provider=${encodedProvider}`,
    {
      method: "GET",
    },
    ACCESS_KEYS_TIMEOUT_SECONDS,
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
