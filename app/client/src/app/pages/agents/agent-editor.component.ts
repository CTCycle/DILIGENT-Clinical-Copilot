// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, OnInit, computed, effect, inject, signal, untracked } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import {
  LucideArrowLeft,
  LucideClipboardCheck,
  LucideFileCheck,
  LucideShieldCheck,
  LucideStethoscope,
} from '@lucide/angular';

import {
  Agent,
  AgentConfig,
  AgentLifecycleRequest,
  AgentRevision,
  AgentValidationIssue,
  CAPABILITY_LABELS,
  Capability,
} from '../../core/models/agent-types';
import { ApiHttpError } from '../../core/services/http-api';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { formatAppDateTime } from '../../core/utils/date-formatting';
import { CapabilitySectionComponent } from '../../components/capability-section/capability-section.component';
import { ConfirmDialogComponent } from '../../components/confirm-dialog/confirm-dialog.component';
import {
  DESCRIPTION_MAX_LENGTH,
  ModelCatalog,
  NAME_MAX_LENGTH,
  ProviderOption,
  defaultNewAgentConfig,
  validateAgentDraft,
} from './agent-form';
import { AssessmentSectionsComponent } from './assessment-sections.component';
import { RevisionSectionsComponent } from './revision-sections.component';
import { TimelineSectionsComponent } from './timeline-sections.component';

type ValidationState = { issues: AgentValidationIssue[]; valid: boolean } | null;

const stripPrefix = (message: string): string => message.replace(/^\[ERROR\]\s*/, '');

/**
 * Shared create/edit page for one agent: profile, validation placeholder and three
 * independently enabled capability sections. Explicit save; lifecycle changes apply
 * immediately and are never blocked by the lock they may be lifting.
 */
@Component({
  selector: 'app-agent-editor',
  standalone: true,
  imports: [
    RouterLink,
    CapabilitySectionComponent,
    ConfirmDialogComponent,
    AssessmentSectionsComponent,
    RevisionSectionsComponent,
    TimelineSectionsComponent,
    LucideArrowLeft,
    LucideClipboardCheck,
    LucideFileCheck,
    LucideShieldCheck,
    LucideStethoscope,
  ],
  styleUrls: ['./agent-sections.scss', './agent-editor.component.scss'],
  templateUrl: './agent-editor.component.html',
})
export class AgentEditorComponent implements OnInit {
  private readonly agents = inject(AgentsStateService);
  private readonly modelConfig = inject(ModelConfigStateService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);

  readonly labels = CAPABILITY_LABELS;
  readonly routeId = signal<string | null>(null);
  /** True for `/agents/new`: nothing is persisted until the first save. */
  readonly creating = computed(() => this.routeId() === 'new');
  readonly agent = computed<Agent | null>(() => {
    const id = this.routeId();
    return id && id !== 'new' ? this.agents.byId(id) : null;
  });
  readonly loadState = computed<'loading' | 'ready' | 'missing' | 'no-seed'>(() => {
    if (this.agents.status() === 'idle' || this.agents.status() === 'loading') {
      return this.agents.all().length ? 'ready' : 'loading';
    }
    if (this.creating()) return this.seed() ? 'ready' : 'no-seed';
    return this.agent() ? 'ready' : 'missing';
  });
  private readonly seed = computed(() => this.agents.defaultAgent() ?? this.agents.all()[0] ?? null);

  readonly name = signal('');
  readonly description = signal('');
  readonly config = signal<AgentConfig | null>(null);
  private readonly baseline = signal('');
  readonly baseRevision = signal(0);

  readonly saving = signal(false);
  readonly saveError = signal('');
  readonly conflict = signal(false);
  readonly savedNotice = signal('');
  readonly lifecycleBusy = signal(false);
  readonly lifecycleError = signal('');
  readonly validating = signal(false);
  readonly validation = signal<ValidationState>(null);
  readonly validationError = signal('');
  readonly revisions = signal<AgentRevision[]>([]);
  readonly leavePrompt = signal<((leave: boolean) => void) | null>(null);

  readonly nameLimit = NAME_MAX_LENGTH;
  readonly descriptionLimit = DESCRIPTION_MAX_LENGTH;
  readonly dirty = computed(() => this.snapshot() !== this.baseline());
  /** New agents are always editable; saved agents follow the server lifecycle rule. */
  readonly editable = computed(() => (this.creating() ? true : (this.agent()?.editable ?? false)));
  readonly readOnlyReason = computed(() => {
    const agent = this.agent();
    if (!agent || agent.editable) return '';
    if (agent.is_archived) return 'This agent is archived. Restore it to edit its configuration.';
    if (!agent.is_active) return 'This agent is inactive. Reactivate it to edit its configuration.';
    return 'This agent is locked. Unlock it to edit its configuration.';
  });
  readonly canSave = computed(
    () => !!this.config() && (this.creating() || this.dirty()) && this.errors().length === 0 && this.editable(),
  );
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
  private lastResetKey = '';

  constructor() {
    effect(() => {
      const state = this.loadState();
      const agent = this.agent();
      const creating = this.creating();
      const key = `${state}:${creating ? 'new' : (agent?.id ?? '')}:${agent?.current_revision ?? 0}`;
      if (state !== 'ready' || key === this.lastResetKey) return;
      this.lastResetKey = key;
      untracked(() => (creating ? this.resetForCreate() : agent && this.reset(agent)));
    });
    this.modelConfig.load().catch(() => undefined);
  }

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.lastResetKey = '';
      this.routeId.set(params.get('id'));
    });
    this.agents.load({ force: true }).catch(() => undefined);
  }

  formatDate(value: string | null | undefined): string {
    return value ? formatAppDateTime(value, 'Unknown date') : 'Not saved yet';
  }

  enabledCapabilities(): number {
    const config = this.config();
    return config ? [config.assessment, config.revision, config.timeline].filter((c) => c.enabled).length : 0;
  }

  summary(capability: Capability): string {
    const config = this.config();
    if (!config) return '';
    const ref =
      capability === 'assessment'
        ? config.assessment.models.clinical
        : capability === 'revision'
          ? config.revision.models.planning
          : config.timeline.model;
    return ref.model ? `${ref.provider} / ${ref.model}` : 'No model selected';
  }

  setCapabilityEnabled(capability: Capability, enabled: boolean): void {
    this.update((draft) => (draft[capability].enabled = enabled));
  }

  setAssessment(value: AgentConfig['assessment']): void {
    this.update((draft) => (draft.assessment = { ...value, enabled: draft.assessment.enabled }));
  }

  setRevision(value: AgentConfig['revision']): void {
    this.update((draft) => (draft.revision = { ...value, enabled: draft.revision.enabled }));
  }

  setTimeline(value: AgentConfig['timeline']): void {
    this.update((draft) => (draft.timeline = { ...value, enabled: draft.timeline.enabled }));
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
    if (!config || this.saving() || !this.canSave()) return;
    this.saving.set(true);
    this.saveError.set('');
    this.conflict.set(false);
    try {
      if (this.creating()) {
        const created = await this.agents.create(this.name().trim(), this.description(), config);
        this.reset(created);
        await this.router.navigate(['/agents', created.id], { replaceUrl: true });
        this.savedNotice.set('Agent created.');
        return;
      }
      const saved = await this.agents.save(
        this.agent()!.id,
        this.baseRevision(),
        this.name().trim(),
        this.description(),
        config,
      );
      this.reset(saved);
      this.savedNotice.set(`Saved as revision ${saved.current_revision}.`);
    } catch (error) {
      if (error instanceof ApiHttpError && error.statusCode === 409 && !this.creating()) {
        this.conflict.set(true);
      } else {
        this.saveError.set(stripPrefix(error instanceof Error ? error.message : String(error)));
      }
    } finally {
      this.saving.set(false);
    }
  }

  async reload(): Promise<void> {
    try {
      await this.agents.load({ force: true });
    } catch {
      this.saveError.set('Could not reload agents. Please retry.');
      return;
    }
    const fresh = this.agent();
    if (fresh) this.reset(fresh);
  }

  async setLifecycle(change: AgentLifecycleRequest): Promise<void> {
    const agent = this.agent();
    if (!agent || this.lifecycleBusy()) return;
    this.lifecycleBusy.set(true);
    this.lifecycleError.set('');
    try {
      await this.agents.setLifecycle(agent.id, change);
    } catch (error) {
      this.lifecycleError.set(stripPrefix(error instanceof Error ? error.message : String(error)));
    } finally {
      this.lifecycleBusy.set(false);
    }
  }

  async validate(): Promise<void> {
    const agent = this.agent();
    if (!agent || this.validating()) return;
    this.validating.set(true);
    this.validationError.set('');
    try {
      this.validation.set(await this.agents.validate(agent.id));
    } catch (error) {
      this.validation.set(null);
      this.validationError.set(stripPrefix(error instanceof Error ? error.message : String(error)));
    } finally {
      this.validating.set(false);
    }
  }

  /** Route guard hook: unsaved edits require an explicit confirmation to leave. */
  canLeave(): boolean | Promise<boolean> {
    if (!this.dirty() || this.saving()) return true;
    return new Promise<boolean>((resolve) => this.leavePrompt.set(resolve));
  }

  resolveLeave(leave: boolean): void {
    this.leavePrompt()?.(leave);
    this.leavePrompt.set(null);
  }

  private update(change: (draft: AgentConfig) => void): void {
    const current = this.config();
    if (!current || !this.editable()) return;
    const draft = structuredClone(current);
    change(draft);
    this.config.set(draft);
    this.savedNotice.set('');
  }

  private snapshot(): string {
    return JSON.stringify([this.name(), this.description(), this.config()]);
  }

  private resetForCreate(): void {
    this.name.set('New agent');
    this.description.set('');
    this.config.set(defaultNewAgentConfig(this.seed()));
    this.baseRevision.set(0);
    this.baseline.set(this.snapshot());
    this.clearTransient();
    this.revisions.set([]);
  }

  private reset(agent: Agent): void {
    this.name.set(agent.name);
    this.description.set(agent.description);
    this.config.set(structuredClone(agent.revision.configuration));
    this.baseRevision.set(agent.current_revision);
    this.baseline.set(this.snapshot());
    this.clearTransient();
    void this.loadRevisions(agent.id);
  }

  private clearTransient(): void {
    this.conflict.set(false);
    this.saveError.set('');
    this.lifecycleError.set('');
    this.validation.set(null);
    this.validationError.set('');
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
