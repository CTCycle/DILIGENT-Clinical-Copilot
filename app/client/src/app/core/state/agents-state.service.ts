// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, computed, signal } from '@angular/core';

import {
  Agent,
  AgentConfig,
  AgentLifecycleRequest,
  AgentRevision,
  Capability,
  isSelectableFor,
} from '../models/agent-types';
import {
  createAgent,
  deleteAgent,
  duplicateAgent,
  fetchAgentRevisions,
  fetchAgents,
  setAgentLifecycle,
  setDefaultAgent,
  updateAgent,
} from '../services/agents-api';

export type AgentsLoadStatus = 'idle' | 'loading' | 'ready' | 'error';

export type AgentsResource = {
  status: AgentsLoadStatus;
  /** Every persisted agent, archived ones included. */
  items: Agent[];
  error: string | null;
};

const INITIAL_RESOURCE: AgentsResource = {
  status: 'idle',
  items: [],
  error: null,
};

@Injectable({ providedIn: 'root' })
export class AgentsStateService {
  private readonly resourceState = signal<AgentsResource>(INITIAL_RESOURCE);
  private loadRequest: Promise<Agent[]> | null = null;

  /** Whether archived agents appear in `agents()`; they are always loaded. */
  readonly showArchived = signal(false);

  readonly resource = this.resourceState.asReadonly();
  readonly status = computed(() => this.resourceState().status);
  readonly error = computed(() => this.resourceState().error);
  readonly all = computed(() => this.resourceState().items);
  readonly agents = computed(() =>
    this.showArchived() ? this.all() : this.all().filter((agent) => !agent.is_archived),
  );
  readonly defaultAgent = computed(() => this.all().find((agent) => agent.is_default) ?? null);

  /** Agents that can run new operations of the capability (active, not archived, enabled). */
  selectableFor(capability: Capability): Agent[] {
    return this.all().filter((agent) => isSelectableFor(agent, capability));
  }

  /** The default agent when it supports the capability, else the first selectable agent. */
  preferredFor(capability: Capability): Agent | null {
    const selectable = this.selectableFor(capability);
    return selectable.find((agent) => agent.is_default) ?? selectable[0] ?? null;
  }

  byId(id: string): Agent | null {
    return this.all().find((agent) => agent.id === id) ?? null;
  }

  async load(options: { force?: boolean } = {}): Promise<Agent[]> {
    const current = this.resourceState();
    if (!options.force && current.status === 'ready') {
      return current.items;
    }
    if (this.loadRequest) {
      return this.loadRequest;
    }

    this.resourceState.update((state) => ({ ...state, status: 'loading', error: null }));
    this.loadRequest = fetchAgents({ includeArchived: true })
      .then((response) => {
        this.resourceState.set({ status: 'ready', items: response.agents, error: null });
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

  async setLifecycle(id: string, change: AgentLifecycleRequest): Promise<Agent> {
    const agent = await setAgentLifecycle(id, change);
    this.upsert(agent);
    return agent;
  }

  async remove(id: string): Promise<void> {
    await deleteAgent(id);
    this.resourceState.update((state) => ({
      ...state,
      items: state.items.filter((item) => item.id !== id),
    }));
  }

  async listRevisions(id: string): Promise<AgentRevision[]> {
    return (await fetchAgentRevisions(id)).revisions;
  }

  // Replace the cached agent with the server response. A new default displaces
  // the previous default, as the server keeps exactly one.
  private upsert(agent: Agent): void {
    this.resourceState.update((state) => {
      const normalized = state.items.map((item) =>
        agent.is_default && item.id !== agent.id && item.is_default
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

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
