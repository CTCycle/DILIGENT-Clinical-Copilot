// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';

import {
  Agent,
  AgentConfig,
  AgentRevision,
  AgentValidationIssue,
  DiliAgentConfig,
  RevisionAgentConfig,
} from '../../core/models/agent-types';
import { ApiHttpError } from '../../core/services/http-api';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { formatAppDateTime } from '../../core/utils/date-formatting';
import {
  DESCRIPTION_MAX_LENGTH,
  ModelCatalog,
  NAME_MAX_LENGTH,
  ProviderOption,
  validateAgentDraft,
} from './agent-form';
import { DiliAgentSectionsComponent } from './dili-agent-sections.component';
import { RevisionAgentSectionsComponent } from './revision-agent-sections.component';

type ValidationState = { issues: AgentValidationIssue[]; valid: boolean } | null;

const stripPrefix = (message: string): string => message.replace(/^\[ERROR\]\s*/, '');

/** Explicit-save editor for one agent: draft state, validation, conflict handling, revision history. */
@Component({
  selector: 'app-agent-editor',
  standalone: true,
  imports: [DiliAgentSectionsComponent, RevisionAgentSectionsComponent],
  styleUrls: ['./agent-sections.scss', './agent-editor.component.scss'],
  templateUrl: './agent-editor.component.html',
})
export class AgentEditorComponent {
  readonly agent = input.required<Agent>();
  readonly dirtyChange = output<boolean>();

  private readonly agents = inject(AgentsStateService);
  private readonly modelConfig = inject(ModelConfigStateService);

  readonly name = signal('');
  readonly description = signal('');
  readonly config = signal<AgentConfig | null>(null);
  private readonly baseline = signal('');
  readonly baseRevision = signal(0);

  readonly saving = signal(false);
  readonly saveError = signal('');
  readonly conflict = signal(false);
  readonly savedNotice = signal('');
  readonly validating = signal(false);
  readonly validation = signal<ValidationState>(null);
  readonly validationError = signal('');
  readonly revisions = signal<AgentRevision[]>([]);

  readonly nameLimit = NAME_MAX_LENGTH;
  readonly descriptionLimit = DESCRIPTION_MAX_LENGTH;
  readonly dili = computed(() => {
    const config = this.config();
    return config?.kind === 'dili' ? (config as DiliAgentConfig) : null;
  });
  readonly revision = computed(() => {
    const config = this.config();
    return config?.kind === 'revision' ? (config as RevisionAgentConfig) : null;
  });
  readonly dirty = computed(() => this.snapshot() !== this.baseline());
  readonly errors = computed(() => {
    const config = this.config();
    return config ? validateAgentDraft(this.name(), this.description(), config) : [];
  });

  readonly providerOptions = computed<ProviderOption[]>(() => [
    { value: 'ollama', label: 'Ollama (local)' },
    ...(this.modelConfig.data()?.cloud_providers ?? []).map((p) => ({
      value: p.id,
      label: p.display_name,
    })),
  ]);
  readonly catalog = computed<ModelCatalog>(() => {
    const data = this.modelConfig.data();
    const entries: Array<[string, string[]]> = [
      ['ollama', (data?.local_models ?? []).map((model) => model.name)],
      ...(data?.cloud_providers ?? []).map((p): [string, string[]] => [p.id, p.models.map((m) => m.id)]),
    ];
    return Object.fromEntries(entries);
  });

  private revisionsRequest = 0;

  constructor() {
    const key = computed(() => `${this.agent().id}:${this.agent().current_revision}`);
    effect(() => {
      key();
      untracked(() => this.reset(this.agent()));
    });
    effect(() => this.dirtyChange.emit(this.dirty()));
    this.modelConfig.load().catch(() => undefined);
  }

  formatDate(value: string | null): string {
    return formatAppDateTime(value, 'Unknown date');
  }

  setConfig(config: AgentConfig): void {
    this.config.set(config);
    this.savedNotice.set('');
  }

  setName(value: string): void {
    this.name.set(value);
    this.savedNotice.set('');
  }

  setDescription(value: string): void {
    this.description.set(value);
    this.savedNotice.set('');
  }

  async save(): Promise<void> {
    const config = this.config();
    if (!config || this.saving() || !this.dirty() || this.errors().length) return;
    this.saving.set(true);
    this.saveError.set('');
    this.conflict.set(false);
    try {
      const saved = await this.agents.save(
        this.agent().id,
        this.baseRevision(),
        this.name().trim(),
        this.description(),
        config,
      );
      this.reset(saved);
      this.savedNotice.set(`Saved as revision ${saved.current_revision}.`);
    } catch (error) {
      if (error instanceof ApiHttpError && error.statusCode === 409) {
        this.conflict.set(true);
      } else {
        this.saveError.set(stripPrefix(error instanceof Error ? error.message : String(error)));
      }
    } finally {
      this.saving.set(false);
    }
  }

  async reload(): Promise<void> {
    const id = this.agent().id;
    try {
      await this.agents.load({ includeArchived: this.agents.resource().includeArchived, force: true });
    } catch {
      this.saveError.set('Could not reload agents. Please retry.');
      return;
    }
    const fresh = this.agents.byId(id);
    if (fresh) this.reset(fresh);
  }

  async validate(): Promise<void> {
    if (this.validating()) return;
    this.validating.set(true);
    this.validationError.set('');
    try {
      this.validation.set(await this.agents.validate(this.agent().id));
    } catch (error) {
      this.validation.set(null);
      this.validationError.set(stripPrefix(error instanceof Error ? error.message : String(error)));
    } finally {
      this.validating.set(false);
    }
  }

  private snapshot(): string {
    return JSON.stringify([this.name(), this.description(), this.config()]);
  }

  private reset(agent: Agent): void {
    this.name.set(agent.name);
    this.description.set(agent.description);
    this.config.set(structuredClone(agent.revision.configuration));
    this.baseRevision.set(agent.current_revision);
    this.baseline.set(this.snapshot());
    this.conflict.set(false);
    this.saveError.set('');
    this.validation.set(null);
    this.validationError.set('');
    void this.loadRevisions(agent.id);
  }

  private async loadRevisions(id: string): Promise<void> {
    const request = ++this.revisionsRequest;
    try {
      const items = await this.agents.listRevisions(id);
      if (request === this.revisionsRequest) {
        this.revisions.set([...items].sort((a, b) => b.revision_number - a.revision_number));
      }
    } catch {
      if (request === this.revisionsRequest) this.revisions.set([]);
    }
  }
}
