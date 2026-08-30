import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';

import { AppStateService } from './app-state.service';

const UI_STATE_KEY = 'dili-agent-ui-state-v1';

describe('AppStateService', () => {
  beforeEach(() => {
    localStorage.clear();
    TestBed.configureTestingModule({});
  });

  it('persists only the UI-local expansion preference', () => {
    const service = TestBed.inject(AppStateService);

    service.updateDiliAgent({
      isExpanded: true,
      form: {
        ...service.state().diliAgent.form,
        patientName: 'Transient patient',
      },
      message: 'Transient message',
      jobId: 'transient-job',
      isRunning: true,
    });
    TestBed.flushEffects();

    expect(JSON.parse(localStorage.getItem(UI_STATE_KEY) || '{}')).toEqual({
      isExpanded: true,
    });
  });

  it('does not restore legacy model, form, or job state', () => {
    localStorage.setItem('dili-agent-state-v2', JSON.stringify({
      isExpanded: true,
      jobId: 'legacy-job',
      isRunning: true,
      settings: { clinicalModel: 'legacy-model' },
    }));

    const service = TestBed.inject(AppStateService);

    expect(service.state().diliAgent.isExpanded).toBe(false);
    expect(service.state().diliAgent.jobId).toBeNull();
    expect(service.state().diliAgent.isRunning).toBe(false);
    expect(service.state().diliAgent.form.patientName).toBe('');
  });

  it('rehydrates the current UI preference without rehydrating runtime state', () => {
    localStorage.setItem(UI_STATE_KEY, JSON.stringify({
      isExpanded: true,
      jobId: 'stale-job',
      isRunning: true,
      settings: { clinicalModel: 'stale-model' },
    }));

    const service = TestBed.inject(AppStateService);

    expect(service.state().diliAgent.isExpanded).toBe(true);
    expect(service.state().diliAgent.jobId).toBeNull();
    expect(service.state().diliAgent.isRunning).toBe(false);
  });
});
