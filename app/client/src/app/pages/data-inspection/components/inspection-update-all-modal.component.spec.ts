import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  InspectionUpdateAllConfigErrorMap,
  InspectionUpdateAllConfigLoadingMap,
  InspectionUpdateAllConfigMap,
  InspectionUpdateAllState,
} from '../../../core/models/inspection-types';
import { InspectionUpdateAllModalComponent } from './inspection-update-all-modal.component';

describe('InspectionUpdateAllModalComponent', () => {
  let fixture: ComponentFixture<InspectionUpdateAllModalComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InspectionUpdateAllModalComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(InspectionUpdateAllModalComponent);
    fixture.componentRef.setInput('isOpen', true);
    fixture.componentRef.setInput('activeTarget', 'livertox');
    fixture.componentRef.setInput('configs', configs());
    fixture.componentRef.setInput('configLoading', configLoading());
    fixture.componentRef.setInput('configErrors', configErrors());
    fixture.componentRef.setInput('state', state());
    fixture.componentRef.setInput('canStart', true);
    fixture.detectChanges();
  });

  function configs(): InspectionUpdateAllConfigMap {
    return {
      livertox: {
        livertox_monograph_max_workers: 4,
        livertox_archive: 'livertox.tar.gz',
        redownload: false,
      },
      rxnav: { rxnav_request_timeout: 30, rxnav_max_concurrency: 4 },
      dilirank: { redownload: false },
    };
  }

  function configLoading(): InspectionUpdateAllConfigLoadingMap {
    return { livertox: false, rxnav: false, dilirank: false };
  }

  function configErrors(): InspectionUpdateAllConfigErrorMap {
    return { livertox: null, rxnav: null, dilirank: null };
  }

  function state(phase: InspectionUpdateAllState['phase'] = 'idle'): InspectionUpdateAllState {
    return {
      runId: phase === 'idle' ? null : 1,
      phase,
      progress: phase === 'running' ? 42 : 0,
      message: phase === 'running' ? 'Source updates are running.' : '',
      cancelRequested: false,
      targets: {
        livertox: { started: phase === 'running', jobId: phase === 'running' ? 'job-livertox' : null, status: phase === 'running' ? 'running' : null, progress: phase === 'running' ? 40 : 0, message: '', error: null },
        rxnav: { started: phase === 'running', jobId: phase === 'running' ? 'job-rxnav' : null, status: phase === 'running' ? 'running' : null, progress: phase === 'running' ? 44 : 0, message: '', error: null },
        dilirank: { started: phase === 'running', jobId: phase === 'running' ? 'job-dilirank' : null, status: phase === 'running' ? 'running' : null, progress: phase === 'running' ? 42 : 0, message: '', error: null },
      },
    };
  }

  it('renders exactly the three structured-source tabs and shared controls', () => {
    const tabs = fixture.nativeElement.querySelectorAll('[role="tab"]');
    expect(tabs).toHaveLength(3);
    expect(fixture.nativeElement.textContent).toContain('LiverTox');
    expect(fixture.nativeElement.textContent).toContain('RxNav');
    expect(fixture.nativeElement.textContent).toContain('DILIrank');
    expect(Array.from(tabs).map((tab) => (tab as HTMLElement).textContent)).not.toContain('RAG');
    expect(fixture.nativeElement.querySelector('input[type="number"]')).not.toBeNull();
    expect(fixture.nativeElement.textContent).toContain('Update All');
  });

  it('switches tabs and disables configuration/start controls while running', () => {
    const selectedTargets: string[] = [];
    fixture.componentInstance.targetChange.subscribe((target) => selectedTargets.push(target));
    const rxnavTab = fixture.nativeElement.querySelector('#inspection-update-all-tab-rxnav') as HTMLButtonElement;
    rxnavTab.click();
    expect(selectedTargets).toEqual(['rxnav']);

    fixture.componentRef.setInput('state', state('running'));
    fixture.componentRef.setInput('activeTarget', 'rxnav');
    fixture.detectChanges();

    const buttons = Array.from(
      fixture.nativeElement.querySelectorAll('button'),
    ) as HTMLButtonElement[];
    const updateButton = buttons
      .find((button) => button.textContent?.trim() === 'Update All') as HTMLButtonElement;
    const cancelButton = buttons
      .find((button) => button.textContent?.trim() === 'Cancel updates') as HTMLButtonElement;
    expect(updateButton.disabled).toBe(true);
    expect(cancelButton.disabled).toBe(false);
    expect(fixture.nativeElement.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('42');
  });
});
