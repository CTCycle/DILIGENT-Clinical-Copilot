// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from "../constants";
import {
  Agent,
  AgentCreateRequest,
  AgentDuplicateRequest,
  AgentLifecycleRequest,
  AgentListResponse,
  AgentRevisionListResponse,
  AgentUpdateRequest,
  AgentValidationResponse,
  Capability,
} from "../models/agent-types";
import {
  ApiHttpError,
  buildQueryString,
  fetchWithTimeout,
  parseApiResponse,
  requestJson,
} from "./http-api";

const JSON_HEADERS = { "Content-Type": "application/json" };

function agentUrl(agentId: string, suffix = ""): string {
  return `${API_BASE_URL}/agents/${encodeURIComponent(agentId)}${suffix}`;
}

export async function fetchAgents(
  options: { selectableFor?: Capability; includeArchived?: boolean } = {},
): Promise<AgentListResponse> {
  const query = buildQueryString({
    selectable_for: options.selectableFor,
    include_archived: options.includeArchived ? "true" : undefined,
  });
  return requestJson<AgentListResponse>(`${API_BASE_URL}/agents${query}`, { method: "GET" });
}

export async function fetchAgent(agentId: string): Promise<Agent> {
  return requestJson<Agent>(agentUrl(agentId), { method: "GET" });
}

export async function createAgent(payload: AgentCreateRequest): Promise<Agent> {
  return requestJson<Agent>(`${API_BASE_URL}/agents`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function updateAgent(agentId: string, payload: AgentUpdateRequest): Promise<Agent> {
  return requestJson<Agent>(agentUrl(agentId), {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function duplicateAgent(
  agentId: string,
  payload: AgentDuplicateRequest = {},
): Promise<Agent> {
  return requestJson<Agent>(agentUrl(agentId, "/duplicate"), {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function validateAgent(agentId: string): Promise<AgentValidationResponse> {
  return requestJson<AgentValidationResponse>(agentUrl(agentId, "/validate"), { method: "POST" });
}

export async function setDefaultAgent(agentId: string): Promise<Agent> {
  return requestJson<Agent>(agentUrl(agentId, "/default"), { method: "PUT" });
}

export async function setAgentLifecycle(
  agentId: string,
  payload: AgentLifecycleRequest,
): Promise<Agent> {
  return requestJson<Agent>(agentUrl(agentId, "/lifecycle"), {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  });
}

export async function deleteAgent(agentId: string): Promise<void> {
  const response = await fetchWithTimeout(agentUrl(agentId), { method: "DELETE" }, 30);
  if (!response.ok) {
    const result = await parseApiResponse(response);
    throw new ApiHttpError(result.message, response.status);
  }
}

export async function fetchAgentRevisions(agentId: string): Promise<AgentRevisionListResponse> {
  return requestJson<AgentRevisionListResponse>(agentUrl(agentId, "/revisions"), { method: "GET" });
}
