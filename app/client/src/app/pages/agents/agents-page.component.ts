// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, OnInit, computed, inject, signal } from '@angular/core';

import { Agent, AgentKind } from '../../core/models/agent-types';
import { summarizeAgent } from '../../core/models/agent-types';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { ModelConfigPageComponent } from '../model-config/model-config-page.component';
import { AgentEditorComponent } from './agent-editor.component';

type StudioTab = 'agents' | 'editor' | 'providers';
const TABS: ReadonlyArray<{ id: StudioTab; label: string }> = [
  { id: 'agents', label: 'My Agents' },
  { id: 'editor', label: 'Agent Editor' },
  { id: 'providers', label: 'Providers & Resources' },
];
const KIND_LABEL: Record<AgentKind, string> = { dili: 'DILI', revision: 'Revision' };

@Component({
  selector: 'app-agents-page',
  standalone: true,
  imports: [AgentEditorComponent, ModelConfigPageComponent],
  templateUrl: './agents-page.component.html',
  styleUrl: './agents-page.component.scss',
})
export class AgentsPageComponent implements OnInit {
  readonly state = inject(AgentsStateService);

  readonly tabs = TABS;
  readonly tab = signal<StudioTab>('agents');
  readonly providersVisited = signal(false);
  readonly search = signal('');
  readonly selectedId = signal<string | null>(null);
  readonly editorDirty = signal(false);
  /** Agent the user wants to open while the editor has unsaved edits. */
  readonly pendingSelectId = signal<string | null>(null);
  readonly confirmArchiveId = signal<string | null>(null);
  readonly creating = signal<{ kind: AgentKind; name: string } | null>(null);
  readonly busy = signal(false);
  readonly message = signal('');
  readonly messageIsError = signal(false);

  readonly selected = computed(() => {
    const id = this.selectedId();
    return id ? this.state.byId(id) : null;
  });
  readonly visibleAgents = computed(() => {
    const query = this.search().trim().toLowerCase();
    return this.state
      .agents()
      .filter((agent) => !query || agent.name.toLowerCase().includes(query))
      .sort((a, b) => a.kind.localeCompare(b.kind) || a.name.localeCompare(b.name));
  });
  readonly showArchived = computed(() => this.state.resource().includeArchived);
  readonly seedFor = computed<Record<AgentKind, Agent | null>>(() => ({
    dili: this.state.defaultDiliAgent() ?? this.state.diliAgents()[0] ?? null,
    revision: this.state.defaultRevisionAgent() ?? this.state.revisionAgents()[0] ?? null,
  }));

  ngOnInit(): void {
    this.state.load().catch(() => undefined);
  }

  kindLabel(kind: AgentKind): string {
    return KIND_LABEL[kind];
  }

  modelLabel(agent: Agent): string {
    return summarizeAgent(agent).clinicalModel ?? '';
  }

  selectTab(id: StudioTab): void {
    this.tab.set(id);
    if (id === 'providers') this.providersVisited.set(true);
  }

  onTabKeydown(event: KeyboardEvent, index: number): void {
    const last = this.tabs.length - 1;
    const next =
      event.key === 'ArrowRight' ? (index + 1) % this.tabs.length
        : event.key === 'ArrowLeft' ? (index - 1 + this.tabs.length) % this.tabs.length
        : event.key === 'Home' ? 0
        : event.key === 'End' ? last
        : null;
    if (next === null) return;
    event.preventDefault();
    this.selectTab(this.tabs[next].id);
    queueMicrotask(() => document.getElementById(`agent-studio-tab-${this.tabs[next].id}`)?.focus());
  }

  async setShowArchived(show: boolean): Promise<void> {
    await this.run(() => this.state.load({ includeArchived: show, force: true }), '');
  }

  open(agent: Agent): void {
    if (agent.id === this.selectedId()) {
      this.selectTab('editor');
    } else if (this.editorDirty()) {
      this.pendingSelectId.set(agent.id);
    } else {
      this.selectedId.set(agent.id);
      this.selectTab('editor');
    }
  }

  discardAndOpen(): void {
    const id = this.pendingSelectId();
    this.pendingSelectId.set(null);
    if (id) {
      this.selectedId.set(id);
      this.selectTab('editor');
    }
  }

  startCreate(kind: AgentKind): void {
    this.creating.set({ kind, name: `New ${KIND_LABEL[kind]} agent` });
  }

  async create(): Promise<void> {
    const draft = this.creating();
    const seed = draft ? this.seedFor()[draft.kind] : null;
    if (!draft || !seed || !draft.name.trim()) return;
    const created = await this.run(
      () => this.state.create(draft.name.trim(), '', structuredClone(seed.revision.configuration)),
      `Created "${draft.name.trim()}" from ${seed.name}.`,
    );
    if (created) {
      this.creating.set(null);
      this.editorDirty.set(false);
      this.selectedId.set(created.id);
      this.selectTab('editor');
    }
  }

  async duplicate(agent: Agent): Promise<void> {
    const copy = await this.run(() => this.state.duplicate(agent.id), `Duplicated "${agent.name}".`);
    if (copy) this.open(copy);
  }

  async setDefault(agent: Agent): Promise<void> {
    await this.run(() => this.state.setDefault(agent.id), `"${agent.name}" is now the default ${KIND_LABEL[agent.kind]} agent.`);
  }

  async archive(agent: Agent): Promise<void> {
    this.confirmArchiveId.set(null);
    await this.run(() => this.state.archive(agent.id), `Archived "${agent.name}".`);
  }

  private async run<T>(action: () => Promise<T>, success: string): Promise<T | null> {
    if (this.busy()) return null;
    this.busy.set(true);
    this.message.set('');
    try {
      const result = await action();
      this.messageIsError.set(false);
      this.message.set(success);
      return result;
    } catch (error) {
      this.messageIsError.set(true);
      this.message.set((error instanceof Error ? error.message : String(error)).replace(/^\[ERROR\]\s*/, ''));
      return null;
    } finally {
      this.busy.set(false);
    }
  }
}
