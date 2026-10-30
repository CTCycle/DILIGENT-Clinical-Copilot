// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { describe, expect, it } from 'vitest';

import { AnalysisProgressComponent } from './analysis-progress.component';
import { resolvePhaseIndex } from './analysis-progress';

describe('resolvePhaseIndex', () => {
  it.each([
    ['preflight.validated', 0],
    ['session_initialization', 0],
    ['drugs.extracting', 1],
    ['therapy_extraction', 1],
    ['drugs.resolving', 2],
    ['labs.extracting', 3],
    ['anamnesis_disease_extraction', 3],
    ['rucam.initial', 4],
    ['hepatotoxicity_pattern', 4],
    ['retrieval.evidence', 5],
    ['evidence_lookup.llm_only', 5],
    ['rucam.refined', 5],
    ['report.generating', 6],
    ['session.saving', 7],
    ['completed', 7],
  ])('maps %s to phase %i', (stage, index) => {
    expect(resolvePhaseIndex(stage)).toBe(index);
  });

  it('returns null for unknown or missing keys', () => {
    expect(resolvePhaseIndex('something.new')).toBeNull();
    expect(resolvePhaseIndex(null)).toBeNull();
  });
});

describe('AnalysisProgressComponent', () => {
  let fixture: ComponentFixture<AnalysisProgressComponent>;

  function set(inputs: Record<string, unknown>): void {
    for (const [name, value] of Object.entries(inputs)) {
      fixture.componentRef.setInput(name, value);
    }
    fixture.detectChanges();
  }

  const bar = () => fixture.nativeElement.querySelector('[role="progressbar"]') as HTMLElement;
  const text = () => (fixture.nativeElement as HTMLElement).textContent ?? '';

  beforeEach(() => {
    fixture = TestBed.createComponent(AnalysisProgressComponent);
  });

  it('is indeterminate before the first event', () => {
    set({ jobStatus: 'pending', jobProgress: 0 });

    expect(bar().getAttribute('aria-valuenow')).toBeNull();
    expect(text()).toContain('Starting analysis');
  });

  it('shows the phase, step count and a determinate bar without a numeric percentage', () => {
    set({
      jobStatus: 'running',
      jobProgress: 30,
      stage: 'drugs.resolving',
      stageMessage: 'Resolving drug names',
    });

    expect(bar().getAttribute('aria-valuenow')).toBe('30');
    expect(text()).toContain('Normalizing drug names');
    expect(text()).toContain('Step 3 of 8');
    expect(text()).toContain('Resolving drug names');
    expect(text()).not.toContain('30%');
  });

  it('keeps the current phase when an unknown key arrives', () => {
    set({ jobStatus: 'running', jobProgress: 54, stage: 'pattern.assessing' });
    set({ stage: 'something.new' });

    expect(text()).toContain('Step 5 of 8');
  });

  it.each([
    ['completed', 'Analysis complete'],
    ['failed', 'Analysis failed'],
    ['cancelled', 'Analysis cancelled'],
  ] as const)('renders the %s state distinctly', (status, label) => {
    set({ jobStatus: status, jobProgress: 100, errorMessage: '[ERROR] boom' });

    expect(text()).toContain(label);
    expect((fixture.nativeElement.querySelector('.analysis-progress') as HTMLElement).dataset['tone']).toBe(status);
    expect(text().includes('[ERROR] boom')).toBe(status === 'failed');
  });
});
