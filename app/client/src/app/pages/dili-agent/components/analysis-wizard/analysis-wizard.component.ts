// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, effect, inject, input, output, signal, untracked } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

import { ModalShellComponent } from '../../../../components/modal-shell/modal-shell.component';
import { describeEvidenceSources } from '../../../../core/evidence-sources';
import { summarizeAgent } from '../../../../core/models/agent-types';
import {
  ClinicalInputPreflightIssue,
  ClinicalInputPreflightResponse,
  ClinicalRequestPayload,
} from '../../../../core/models/types';
import { validateClinicalInput } from '../../../../core/services/clinical-api';
import { ClinicalSessionDraftService } from '../../../../core/services/clinical-session-draft.service';
import { launchClinicalSession } from '../../../../core/services/clinical-session-lifecycle-api';
import { ApiHttpError } from '../../../../core/services/http-api';
import { AgentsStateService } from '../../../../core/state/agents-state.service';
import { AppStateService } from '../../../../core/state/app-state.service';
import { buildClinicalPayload } from '../../../../core/utils';
import { buildSectionPreviews, SourceSelection } from './section-preview';

export type WizardStepId = 'preflight' | 'agent' | 'review';

export const WIZARD_STEPS: ReadonlyArray<{ id: WizardStepId; label: string }> = [
  { id: 'preflight', label: 'Preflight' },
  { id: 'agent', label: 'Agent' },
  { id: 'review', label: 'Review' },
];

export const AGENT_CHANGED_MESSAGE = 'The agent changed after it was checked — re-run checks.';

function excerpt(text: string, limit = 160): string {
  const normalized = text.trim().replace(/\s+/g, ' ');
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized;
}

export const LIVERTOX_WARNING =
  'Excluding LiverTox removes drug-level hepatotoxicity evidence from the causality assessment and may reduce accuracy.';

type PreflightRecord = {
  result: ClinicalInputPreflightResponse;
  formKey: string;
  configKey: string;
  agentId: string | null;
  agentRevision: number | null;
};

function preflightUnavailable(): ClinicalInputPreflightResponse {
  const description =
    'The application could not complete the safety checks. Verify backend connectivity and runtime configuration, then retry.';
  return {
    ready: false,
    blocking_issues: [
      {
        severity: 'blocking',
        code: 'preflight_request_failed',
        message: description,
        field: null,
        title: 'Pre-flight checks are unavailable',
        description,
        affected_section: 'Analysis readiness',
        consequence: 'The analysis cannot start until the safety checks complete successfully.',
        continuation_allowed: false,
      },
    ],
    non_blocking_issues: [],
    runtime_settings: {},
    extraction_quality: {},
    deterministic_diagnostics: {},
    rag_readiness: null,
  };
}

@Component({
  selector: 'app-analysis-wizard',
  imports: [ModalShellComponent, RouterLink],
  templateUrl: './analysis-wizard.component.html',
  styleUrl: './analysis-wizard.component.scss',
})
export class AnalysisWizardComponent {
  readonly isOpen = input(false);
  readonly cancelled = output<void>();
  readonly started = output<void>();
  readonly fixInput = output<string | null>();
  readonly editSource = output<SourceSelection | null>();

  private readonly stateService = inject(AppStateService);
  private readonly agentsState = inject(AgentsStateService);
  private readonly drafts = inject(ClinicalSessionDraftService);
  private readonly router = inject(Router);

  readonly steps = WIZARD_STEPS;
  readonly livertoxWarning = LIVERTOX_WARNING;

  readonly step = signal<WizardStepId>('preflight');
  readonly selectedAgentId = signal<string | null>(null);
  readonly record = signal<PreflightRecord | null>(null);
  readonly checking = signal(false);
  readonly starting = signal(false);
  readonly startError = signal<string | null>(null);

  private checkAttempt = 0;

  readonly form = computed(() => this.stateService.state().diliAgent.form);
  readonly stepIndex = computed(() => this.steps.findIndex((item) => item.id === this.step()));
  readonly formKey = computed(() => JSON.stringify(this.form()));
  readonly agent = computed(() => {
    const selected = this.selectedAgentId();
    return (
      this.agentsState.selectableFor('assessment').find((item) => item.id === selected)
      ?? this.agentsState.preferredFor('assessment')
    );
  });
  readonly configKey = computed(() => JSON.stringify([this.agent()?.id ?? null, this.agent()?.current_revision ?? null]));

  private readonly formCurrent = computed(() => this.record()?.formKey === this.formKey());
  readonly result = computed(() => (this.formCurrent() ? this.record()!.result : null));
  readonly issues = computed<ClinicalInputPreflightIssue[]>(() => {
    const result = this.result();
    return result ? [...result.blocking_issues, ...result.non_blocking_issues] : [];
  });
  readonly sectionPreviews = computed(() =>
    buildSectionPreviews(this.result()?.deterministic_diagnostics, this.form().clinicalInput),
  );
  readonly blockingCount = computed(() => this.result()?.blocking_issues.length ?? 0);
  readonly warningCount = computed(() => this.result()?.non_blocking_issues.length ?? 0);
  readonly inputReady = computed(() => this.result() !== null && this.blockingCount() === 0);
  private readonly onlyConfigurationIssues = computed(() => {
    const blocking = this.result()?.blocking_issues ?? [];
    return blocking.length > 0 && blocking.every((issue) => issue.field && issue.field in CONFIGURATION_STEPS);
  });
  readonly checkCurrent = computed(
    () => this.formCurrent() && this.record()?.configKey === this.configKey(),
  );
  readonly canStart = computed(
    () => this.inputReady() && this.checkCurrent() && !this.checking() && !this.starting(),
  );
  readonly agentSummary = computed(() => {
    const agent = this.agent();
    return agent ? summarizeAgent(agent) : null;
  });
  readonly instructionExcerpts = computed(() => {
    const config = this.agent()?.revision.configuration.assessment;
    if (!config) return [];
    const { instructions } = config;
    return [
      { label: 'Custom instructions', text: excerpt(instructions.custom_instructions) },
      { label: 'Report template', text: config.report_template?.file_name ?? '' },
    ].filter((row) => row.text);
  });
  readonly livertoxExcluded = computed(() => {
    const summary = this.agentSummary();
    return summary !== null && !summary.evidenceSources.includes('livertox');
  });
  readonly sourcesSummary = computed(() => describeEvidenceSources(this.agentSummary()?.evidenceSources ?? []));
  readonly agents = computed(() => this.agentsState.selectableFor('assessment'));

  constructor() {
    effect(() => {
      if (this.isOpen()) {
        untracked(() => {
          void this.agentsState.load({ force: true }).catch(() => undefined);
          // The input changed since the last check: restart from preflight.
          if (!this.formCurrent()) {
            this.step.set('preflight');
          }
        });
      }
    });
    effect(() => {
      const needsCheck = this.isOpen()
        && !this.checkCurrent()
        && !this.checking()
        && (this.step() === 'preflight' || this.step() === 'review')
        && (this.agentsState.status() === 'ready' || this.agentsState.status() === 'error');
      if (needsCheck) {
        untracked(() => void this.runPreflight());
      }
    });
  }

  reset(): void {
    this.checkAttempt += 1;
    this.selectedAgentId.set(null);
    this.record.set(null);
    this.checking.set(false);
    this.starting.set(false);
    this.startError.set(null);
    this.step.set('preflight');
  }

  canVisit(index: number): boolean {
    return index <= this.stepIndex() || this.inputReady() || this.onlyConfigurationIssues();
  }

  goTo(index: number): void {
    if (this.canVisit(index)) {
      this.step.set(this.steps[index].id);
    }
  }

  next(): void {
    this.goTo(Math.min(this.stepIndex() + 1, this.steps.length - 1));
  }

  back(): void {
    this.goTo(Math.max(this.stepIndex() - 1, 0));
  }

  selectAgent(agentId: string): void {
    this.startError.set(null);
    this.selectedAgentId.set(agentId || null);
  }

  async runPreflight(): Promise<void> {
    const attempt = ++this.checkAttempt;
    const formKey = this.formKey();
    const configKey = this.configKey();
    const agentId = this.agent()?.id ?? null;
    const agentRevision = this.agent()?.current_revision ?? null;
    this.checking.set(true);
    let result: ClinicalInputPreflightResponse;
    try {
      result = await validateClinicalInput(this.buildPayload(agentId, agentRevision));
    } catch (error) {
      console.error('Clinical pre-flight request failed.', error);
      result = preflightUnavailable();
    }
    if (attempt !== this.checkAttempt) {
      return;
    }
    this.record.set({ result, formKey, configKey, agentId, agentRevision });
    this.checking.set(false);
  }

  fixInInput(): void {
    const fields = this.issues().map((issue) => issue.field ?? '');
    const configurationField = fields.find((field) => field in CONFIGURATION_STEPS);
    if (configurationField) {
      this.step.set(CONFIGURATION_STEPS[configurationField]);
      return;
    }
    this.fixInput.emit(fields.map((field) => PREFLIGHT_FIELD_TARGETS[field]).find(Boolean) ?? null);
  }

  editInInput(selection: SourceSelection | null): void {
    this.editSource.emit(selection);
  }

  async start(): Promise<void> {
    if (!this.canStart()) {
      return;
    }
    this.starting.set(true);
    this.startError.set(null);
    let sessionId: number;
    const checked = this.record()!;
    const checkedRevision = checked.result.runtime_settings['agent_revision'];
    try {
      const payload = this.buildPayload(
        checked.agentId,
        typeof checkedRevision === 'number' ? checkedRevision : checked.agentRevision,
      );
      const { draftSessionId, isDirty } = this.stateService.state().diliAgent;
      const savedId = draftSessionId !== null && !isDirty ? draftSessionId : await this.drafts.save();
      if (savedId === null) {
        throw new Error('The session could not be saved before starting. Your input is unchanged.');
      }
      sessionId = savedId;
      await launchClinicalSession(sessionId, payload);
    } catch (error) {
      this.starting.set(false);
      if (error instanceof ApiHttpError && error.statusCode === 409) {
        // A 409 is also returned for unrelated conflicts (e.g. another operation running):
        // only treat it as an agent change when the selected agent really moved on.
        await this.agentsState.load({ force: true }).catch(() => undefined);
        const current = this.agent();
        const launchedRevision = typeof checkedRevision === 'number' ? checkedRevision : checked.agentRevision;
        if (!current || current.id !== checked.agentId || current.current_revision !== launchedRevision) {
          this.record.set(null);
          this.startError.set(AGENT_CHANGED_MESSAGE);
          this.step.set('preflight');
          return;
        }
      }
      this.startError.set(error instanceof Error ? error.message : 'The analysis could not be started.');
      return;
    }
    // Navigate first so clearing the draft cannot race with the page's `?session=` sync.
    // Replacing the entry keeps Back from reopening the launched draft's editor URL.
    await this.router.navigate(['/report', sessionId], { replaceUrl: true });
    this.reset();
    this.drafts.reset();
    this.started.emit();
  }

  private buildPayload(agentId: string | null, agentRevision: number | null): ClinicalRequestPayload {
    return buildClinicalPayload(this.form(), agentId, agentRevision);
  }
}

export const PREFLIGHT_FIELD_TARGETS: Record<string, string> = {
  anamnesis: 'clinical-input',
  clinical_input: 'clinical-input',
  drugs: 'clinical-input',
  laboratory_analysis: 'clinical-input',
  visit_date: 'visit-date',
};

const CONFIGURATION_STEPS: Record<string, WizardStepId> = {
  agent: 'agent',
};
