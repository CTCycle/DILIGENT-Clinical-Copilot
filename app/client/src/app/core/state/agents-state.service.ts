// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, computed, signal } from '@angular/core';

import {
  Agent,
  AgentConfig,
  AgentRevision,
  AgentValidationResponse,
} from '../models/agent-types';
import {
  archiveAgent,
  createAgent,
  duplicateAgent,
  fetchAgentRevisions,
  fetchAgents,
  setDefaultAgent,
  updateAgent,
  validateAgent,
} from '../services/agents-api';

export type AgentsLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

export type AgentsResource = {
  status: AgentsLoadStatus;
  items: Agent[];
  includeArchived: boolean;
  error: string | null;
};

const INITIAL_RESOURCE: AgentsResource = {
  status: 'idle',
  items: [],
  includeArchived: false,
  error: null,
};

@Injectable({ providedIn: 'root' })
export class AgentsStateService {
  private readonly resourceState = signal<AgentsResource>(INITIAL_RESOURCE);
  private loadRequest: Promise<Agent[]> | null = null;

  readonly resource = this.resourceState.asReadonly();
  readonly status = computed(() => this.resourceState().status);
  readonly error = computed(() => this.resourceState().error);
  readonly agents = computed(() => {
    const { items, includeArchived } = this.resourceState();
    return includeArchived ? items : items.filter((agent) => !agent.is_archived);
  });
  readonly diliAgents = computed(() => this.agents().filter((agent) => agent.kind === 'dili'));
  readonly revisionAgents = computed(() =>
    this.agents().filter((agent) => agent.kind === 'revision'),
  );
  readonly defaultDiliAgent = computed(() => this.defaultOf(this.diliAgents()));
  readonly defaultRevisionAgent = computed(() => this.defaultOf(this.revisionAgents()));

  byId(id: string): Agent | null {
    return this.agents().find((agent) => agent.id === id) ?? null;
  }

  async load(options: { includeArchived?: boolean; force?: boolean } = {}): Promise<Agent[]> {
    const includeArchived = options.includeArchived ?? false;
    const current = this.resourceState();
    if (
      !options.force &&
      current.status === 'ready' &&
      (current.includeArchived || !includeArchived)
    ) {
      return current.items;
    }
    if (this.loadRequest) {
      return this.loadRequest;
    }

    this.resourceState.update((state) => ({ ...state, status: 'loading', error: null }));
    this.loadRequest = fetchAgents({ includeArchived })
      .then((response) => {
        this.resourceState.set({
          status: 'ready',
          items: response.agents,
          includeArchived,
          error: null,
        });
        return response.agents;
      })
      .catch((error: unknown) => {
        this.resourceState.update((state) => ({
          ...state,
          status: 'error',
          error: this.errorMessage(error),
        }));
        throw error;
      })
      .finally(() => {
        this.loadRequest = null;
      });
    return this.loadRequest;
  }

  async create(name: string, description: string, configuration: AgentConfig): Promise<Agent> {
    const agent = await createAgent({ name, description, configuration });
    this.upsert(agent);
    return agent;
  }

  async save(
    id: string,
    expectedRevision: number,
    name: string,
    description: string,
    configuration: AgentConfig,
  ): Promise<Agent> {
    const agent = await updateAgent(id, {
      expected_revision: expectedRevision,
      name,
      description,
      configuration,
    });
    this.upsert(agent);
    return agent;
  }

  async duplicate(id: string, name?: string | null): Promise<Agent> {
    const agent = await duplicateAgent(id, name ? { name } : {});
    this.upsert(agent);
    return agent;
  }

  async setDefault(id: string): Promise<Agent> {
    const agent = await setDefaultAgent(id);
    this.upsert(agent);
    return agent;
  }

  async archive(id: string): Promise<Agent> {
    const agent = await archiveAgent(id);
    this.upsert(agent);
    return agent;
  }

  validate(id: string): Promise<AgentValidationResponse> {
    return validateAgent(id);
  }

  async listRevisions(id: string): Promise<AgentRevision[]> {
    return (await fetchAgentRevisions(id)).revisions;
  }

  // Replace the cached agent with the server response. A new default displaces
  // the previous default of the same kind, as the server keeps one per kind.
  private upsert(agent: Agent): void {
    this.resourceState.update((state) => {
      const normalized = state.items.map((item) =>
        agent.is_default && item.kind === agent.kind && item.id !== agent.id && item.is_default
          ? { ...item, is_default: false }
          : item,
      );
      const index = normalized.findIndex((item) => item.id === agent.id);
      const items =
        index >= 0
          ? normalized.map((item, i) => (i === index ? agent : item))
          : [...normalized, agent];
      return { ...state, items };
    });
  }

  private defaultOf(agents: Agent[]): Agent | null {
    return agents.find((agent) => agent.is_default) ?? null;
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
