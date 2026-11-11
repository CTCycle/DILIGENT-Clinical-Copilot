// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { API_BASE_URL } from "../constants";
import {
  Agent,
  AgentCreateRequest,
  AgentDuplicateRequest,
  AgentKind,
  AgentListResponse,
  AgentRevisionListResponse,
  AgentUpdateRequest,
  AgentValidationResponse,
} from "../models/agent-types";
import { buildQueryString, requestJson } from "./http-api";

const JSON_HEADERS = { "Content-Type": "application/json" };

function agentUrl(agentId: string, suffix = ""): string {
  return `${API_BASE_URL}/agents/${encodeURIComponent(agentId)}${suffix}`;
}

export async function fetchAgents(
  options: { kind?: AgentKind; includeArchived?: boolean } = {},
): Promise<AgentListResponse> {
  const query = buildQueryString({
    kind: options.kind,
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

export async function archiveAgent(agentId: string): Promise<Agent> {
  return requestJson<Agent>(agentUrl(agentId, "/archive"), { method: "POST" });
}

export async function fetchAgentRevisions(agentId: string): Promise<AgentRevisionListResponse> {
  return requestJson<AgentRevisionListResponse>(agentUrl(agentId, "/revisions"), { method: "GET" });
}
