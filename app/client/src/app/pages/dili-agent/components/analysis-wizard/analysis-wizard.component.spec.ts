// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Agent } from '../../../../core/models/agent-types';
import { makeAgent, makeAgentConfig } from '../../../../core/testing/agent-fixtures';
import {
  ClinicalInputPreflightIssue,
  ClinicalInputPreflightResponse,
  ClinicalRequestPayload,
} from '../../../../core/models/types';
import { jsonResponse, sessionState } from '../../../../core/testing/clinical-session-fixtures';
import { AppStateService } from '../../../../core/state/app-state.service';
import { AnalysisWizardComponent, AGENT_CHANGED_MESSAGE, LIVERTOX_WARNING } from './analysis-wizard.component';

function issue(severity: 'blocking' | 'non_blocking', code: string, field: string): ClinicalInputPreflightIssue {
  return {
    severity,
    code,
    field,
    message: code,
    title: `${code} title`,
    description: `${code} description`,
    affected_section: field,
    consequence: `${code} consequence`,
    continuation_allowed: severity === 'non_blocking',
  };
}

function preflight(blocking: ClinicalInputPreflightIssue[] = [], warnings: ClinicalInputPreflightIssue[] = []) {
  return {
    ready: blocking.length === 0,
    blocking_issues: blocking,
    non_blocking_issues: warnings,
    runtime_settings: {},
    extraction_quality: {},
    deterministic_diagnostics: {},
    rag_readiness: null,
  } satisfies ClinicalInputPreflightResponse;
}

function agent(id: string, overrides: Partial<Agent> = {}, sources: string[] = ['livertox']): Agent {
  const config = makeAgentConfig();
  config.assessment.instructions = {
    assessment_focus: 'Focus on RUCAM',
    report_style: 'Concise',
    extraction_guidance: '',
  };
  config.assessment.ollama_seed = null;
  config.assessment.retrieval.evidence_sources = sources as never;
  const result = makeAgent(id, { name: `Agent ${id}`, current_revision: 2, ...overrides }, config);
  result.revision.revision_number = 2;
  return result;
}

describe('AnalysisWizardComponent', () => {
  let fixture: ComponentFixture<AnalysisWizardComponent>;
  let component: AnalysisWizardComponent;
  let appState: AppStateService;
  let sessionRequests: string[];
  let launchPayloads: ClinicalRequestPayload[];
  let launchResponse: () => Response;
  let payloads: ClinicalRequestPayload[];
  let agents: Agent[];
  let responses: ClinicalInputPreflightResponse[];

  const root = () => fixture.nativeElement as HTMLElement;
  const button = (label: string) =>
    Array.from(root().querySelectorAll('button')).find((item) => item.textContent?.trim() === label) as
      | HTMLButtonElement
      | undefined;

  const startButton = () => root().querySelector('.wizard-footer .btn-primary') as HTMLButtonElement | null;

  async function settle(): Promise<void> {
    for (let i = 0; i < 5; i += 1) {
      fixture.detectChanges();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  beforeEach(async () => {
    localStorage.clear();
    payloads = [];
    responses = [];
    agents = [agent('a1', { is_default: true }), agent('a2', { name: 'No LiverTox' }, ['dilirank'])];
    sessionRequests = [];
    launchPayloads = [];
    launchResponse = () => jsonResponse(sessionState({ status: 'queued', queue_position: 1 }), 202);
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init) => {
      const target = String(url);
      if (target.includes('/clinical/sessions')) {
        sessionRequests.push(`${init?.method} ${target}`);
        if (target.endsWith('/launch')) {
          launchPayloads.push(JSON.parse(String(init?.body)) as ClinicalRequestPayload);
          return launchResponse();
        }
        return jsonResponse(sessionState(), init?.method === 'POST' ? 201 : 200);
      }
      if (target.includes('/agents')) {
        return jsonResponse({ agents });
      }
      payloads.push(JSON.parse(String(init?.body)) as ClinicalRequestPayload);
      return jsonResponse(responses.shift() ?? preflight());
    });
    await TestBed.configureTestingModule({
      imports: [AnalysisWizardComponent],
      providers: [provideRouter([])],
    }).compileComponents();
    appState = TestBed.inject(AppStateService);
    appState.updateDiliAgent({
      form: { patientName: 'P', visitDate: '2026-10-01', patientImageDataUrl: null, clinicalInput: 'text' },
    });
    fixture = TestBed.createComponent(AnalysisWizardComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('isOpen', true);
    fixture.detectChanges();
    await settle();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('runs preflight on open with the default agent and none of the removed fields', () => {
    expect(payloads).toHaveLength(1);
    expect(payloads[0].agent_id).toBe('a1');
    expect(payloads[0].agent_revision).toBe(2);
    for (const removed of ['evidence_sources', 'model_overrides', 'selected_model_providers']) {
      expect(removed in payloads[0]).toBe(false);
    }
  });

  it('returns to preflight when reopened after the clinical input changed', async () => {
    component.goTo(1);
    fixture.componentRef.setInput('isOpen', false);
    fixture.detectChanges();
    responses.push(preflight([issue('blocking', 'visit_date_missing', 'visit_date')]));
    appState.updateDiliAgent({
      form: { patientName: 'P', visitDate: '', patientImageDataUrl: null, clinicalInput: 'text' },
    });

    fixture.componentRef.setInput('isOpen', true);
    await settle();

    expect(component.step()).toBe('preflight');
    expect(payloads).toHaveLength(2);
    expect(component.blockingCount()).toBe(1);
  });

  it('shows the selected agent summary and warns when LiverTox is excluded', () => {
    component.goTo(1);
    fixture.detectChanges();
    const card = root().querySelector('[data-testid="agent-summary"]') as HTMLElement;
    expect(card.textContent).toContain('Agent a1');
    expect(card.textContent).toContain('Revision 2');
    expect(card.textContent).toContain('openai / gpt-x');
    expect(card.textContent).toContain('Focus on RUCAM');
    expect(card.textContent).not.toContain(LIVERTOX_WARNING);
    expect(root().querySelector('a[href="/agents"]')).not.toBeNull();

    component.selectAgent('a2');
    fixture.detectChanges();
    expect(root().textContent).toContain(LIVERTOX_WARNING);
  });

  it('blocks Next and later steps while preflight has blocking issues', async () => {
    responses.push(preflight([issue('blocking', 'visit_date_missing', 'visit_date')]));
    component.reset();
    await component.runPreflight();
    fixture.detectChanges();

    expect(button('Next')?.disabled).toBe(true);
    expect(component.canVisit(1)).toBe(false);
    expect(root().textContent).toContain('1 blocking');
  });

  it('invalidates the check when the agent changes and re-runs it on Review before Start enables', async () => {
    expect(component.checkCurrent()).toBe(true);

    component.selectAgent('a2');
    expect(component.checkCurrent()).toBe(false);

    component.goTo(2);
    fixture.detectChanges();
    expect(startButton()?.disabled).toBe(true);
    await settle();

    expect(payloads.at(-1)?.agent_id).toBe('a2');
    expect(component.checkCurrent()).toBe(true);
    expect(startButton()?.disabled).toBe(false);
  });

  it('routes agent preflight issues to the Agent step', async () => {
    responses.push(preflight([issue('blocking', 'agent_unavailable', 'agent')]));
    component.reset();
    await component.runPreflight();

    component.fixInInput();

    expect(component.step()).toBe('agent');
  });

  describe('start', () => {
    let navigate: ReturnType<typeof vi.spyOn>;
    let started: () => void;

    beforeEach(() => {
      navigate = vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
      started = vi.fn();
      component.started.subscribe(started);
      component.goTo(2);
      fixture.detectChanges();
    });

    it('saves a new draft, launches it once, opens its report, and resets the form', async () => {
      appState.updateDiliAgent({ isDirty: true });
      await Promise.all([component.start(), component.start()]);

      expect(sessionRequests).toEqual([
        'POST /api/clinical/sessions',
        'POST /api/clinical/sessions/5/launch',
      ]);
      expect(launchPayloads[0]).toMatchObject({ agent_id: 'a1', agent_revision: 2 });
      expect('evidence_sources' in launchPayloads[0]).toBe(false);
      expect(navigate).toHaveBeenCalledWith(['/report', 5], { replaceUrl: true });
      expect(started).toHaveBeenCalledTimes(1);
      expect(appState.state().diliAgent).toMatchObject({
        draftSessionId: null,
        isDirty: false,
        form: { patientName: '', clinicalInput: '' },
      });
      expect(component.selectedAgentId()).toBeNull();
    });

    it('launches with the agent revision returned by the successful preflight', async () => {
      responses.push({ ...preflight(), runtime_settings: { agent_id: 'a1', agent_revision: 3 } });
      await component.runPreflight();

      await component.start();

      expect(launchPayloads[0]).toMatchObject({ agent_id: 'a1', agent_revision: 3 });
    });

    it('returns to preflight and asks to re-run checks when the agent changed (409)', async () => {
      launchResponse = () => {
        agents = [agent('a1', { is_default: true, current_revision: 3 }), agents[1]];
        return jsonResponse({ detail: 'Agent revision changed.' }, 409);
      };

      await component.start();
      await settle();

      expect(component.step()).toBe('preflight');
      expect(navigate).not.toHaveBeenCalled();
      expect(root().textContent).toContain(AGENT_CHANGED_MESSAGE);
    });

    it('reports an unrelated 409 conflict instead of claiming the agent changed', async () => {
      launchResponse = () => jsonResponse({ detail: 'Another operation is running.' }, 409);

      await component.start();
      await settle();

      expect(component.step()).toBe('review');
      expect(navigate).not.toHaveBeenCalled();
      expect(root().textContent).not.toContain(AGENT_CHANGED_MESSAGE);
      expect(root().textContent).toContain('Another operation is already running');
    });

    it('launches a clean stored draft without rewriting it', async () => {
      appState.updateDiliAgent({ draftSessionId: 5, draftUpdatedAt: '2026-10-01T08:00:00Z', isDirty: false });

      await component.start();

      expect(sessionRequests).toEqual(['POST /api/clinical/sessions/5/launch']);
    });

    it('updates a modified draft before launching it', async () => {
      appState.updateDiliAgent({ draftSessionId: 5, draftUpdatedAt: '2026-10-01T08:00:00Z', isDirty: true });

      await component.start();

      expect(sessionRequests).toEqual([
        'PUT /api/clinical/sessions/5/input',
        'POST /api/clinical/sessions/5/launch',
      ]);
    });

    it('keeps the form and draft and shows the error when launch is rejected', async () => {
      launchResponse = () => jsonResponse({ detail: 'Backend validation failed.' }, 422);

      await component.start();
      fixture.detectChanges();

      expect(component.starting()).toBe(false);
      expect(navigate).not.toHaveBeenCalled();
      expect(started).not.toHaveBeenCalled();
      expect(appState.state().diliAgent.draftSessionId).toBe(5);
      expect(appState.state().diliAgent.form.clinicalInput).toBe('text');
      expect(root().textContent).toContain('Backend validation failed.');
    });
  });
});
