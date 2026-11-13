// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, HostListener, OnInit, computed, effect, inject, signal, untracked } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';

import { LucideArrowLeft } from '@lucide/angular';

import {
  Agent,
  AgentConfig,
  AgentLifecycleRequest,
  AgentRevision,
  CAPABILITY_LABELS,
  Capability,
} from '../../core/models/agent-types';
import { ApiHttpError } from '../../core/services/http-api';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { formatAppDateTime } from '../../core/utils/date-formatting';
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

type EditorTab = 'overview' | Capability;
const TABS: ReadonlyArray<{ id: EditorTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'assessment', label: CAPABILITY_LABELS.assessment },
  { id: 'revision', label: CAPABILITY_LABELS.revision },
  { id: 'timeline', label: CAPABILITY_LABELS.timeline },
];

const stripPrefix = (message: string): string => message.replace(/^\[ERROR\]\s*/, '');

/**
 * Shared create/edit page for one agent: an Overview tab and one tab per independently
 * enabled capability. All tabs edit the same draft; explicit save; lifecycle changes apply
 * immediately and are never blocked by the lock they may be lifting.
 */
@Component({
  selector: 'app-agent-editor',
  standalone: true,
  imports: [
    RouterLink,
    ConfirmDialogComponent,
    AssessmentSectionsComponent,
    RevisionSectionsComponent,
    TimelineSectionsComponent,
    LucideArrowLeft,
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
  readonly tabs = TABS;
  readonly activeTab = signal<EditorTab>('overview');
  readonly routeId = signal<string | null>(null);
  /** True for `/agents/new`: nothing is persisted until the first save. */
  readonly creating = computed(() => this.routeId() === 'new');
  readonly agent = computed<Agent | null>(() => {
    const id = this.routeId();
    return id && id !== 'new' ? this.agents.byId(id) : null;
  });
  readonly displayName = computed(() => {
    const agent = this.agent();
    return agent && this.name() === agent.name ? this.agents.displayName(agent) : this.name();
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
  /** Short editing-context label shown in the header next to the active tab. */
  readonly stateLabel = computed(() => {
    const agent = this.agent();
    if (this.creating() || agent?.editable) return 'Editable';
    if (agent?.is_archived) return 'Read-only · archived';
    if (agent && !agent.is_active) return 'Read-only · inactive';
    return 'Read-only · locked';
  });
  readonly capability = computed<Capability | null>(() => {
    const tab = this.activeTab();
    return tab === 'overview' ? null : tab;
  });
  readonly activeTabLabel = computed(() => TABS.find((tab) => tab.id === this.activeTab())?.label ?? '');
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
      ['ollama', (data?.local_models ?? []).filter((model) => model.available_in_ollama).map((model) => model.name)],
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

  selectTab(id: EditorTab): void {
    this.activeTab.set(id);
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
    queueMicrotask(() => document.getElementById(`agent-tab-${this.tabs[next].id}`)?.focus());
  }

  isCapabilityEnabled(id: EditorTab): boolean {
    const config = this.config();
    return id === 'overview' || !config ? true : config[id].enabled;
  }

  /** Browser-level guard: closing or reloading the tab with unsaved edits asks first. */
  @HostListener('window:beforeunload', ['$event'])
  onBeforeUnload(event: BeforeUnloadEvent): void {
    if (this.dirty() && !this.saving()) event.preventDefault();
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
