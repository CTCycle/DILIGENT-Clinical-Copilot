// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { beforeEach, describe, expect, it } from 'vitest';
import { InspectionUpdateControlsComponent } from './inspection-update-controls.component';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { InspectionUpdateAllConfigErrorMap, InspectionUpdateAllConfigLoadingMap, InspectionUpdateAllConfigMap, InspectionUpdateAllState } from '../../../core/models/inspection-types';
import { InspectionUpdateAllModalComponent } from './inspection-update-all-modal.component';

describe('InspectionUpdateControlsComponent', () => {
  it('emits a bounded integer concurrency value', () => {
    const component = new InspectionUpdateControlsComponent();
    let emitted: { key: string; value: unknown } | null = null;
    component.configChange.subscribe((value) => {
      emitted = value;
    });

    component.updateNumber('livertox_monograph_max_workers', 80, 1, 32, true);

    expect(emitted).toEqual({ key: 'livertox_monograph_max_workers', value: 32 });
  });

  it('emits the selected archive handling mode', () => {
    const component = new InspectionUpdateControlsComponent();
    let emitted: { key: string; value: unknown } | null = null;
    component.configChange.subscribe((value) => {
      emitted = value;
    });

    component.updateBoolean('redownload', true);

    expect(emitted).toEqual({ key: 'redownload', value: true });
  });

  it('reads displayed values from the loaded configuration', () => {
    const component = new InspectionUpdateControlsComponent();
    component.config = {
      livertox_monograph_max_workers: 6,
      livertox_archive: 'livertox_NBK547852.tar.gz',
      redownload: false,
    };

    expect(component.numberValue('livertox_monograph_max_workers', 4)).toBe(6);
    expect(component.stringValue('livertox_archive')).toBe('livertox_NBK547852.tar.gz');
    expect(component.booleanValue('redownload', true)).toBe(false);
  });
});

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
      rxnav: { force_full: false },
      terminology: { force_full: false, drugcentral_enabled: true, chembl_enabled: true, swissmedic_enabled: true },
      dilirank: { redownload: false },
    };
  }

  function configLoading(): InspectionUpdateAllConfigLoadingMap {
    return { livertox: false, rxnav: false, terminology: false, dilirank: false };
  }

  function configErrors(): InspectionUpdateAllConfigErrorMap {
    return { livertox: null, rxnav: null, terminology: null, dilirank: null };
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
        terminology: { started: phase === 'running', jobId: phase === 'running' ? 'job-terminology' : null, status: phase === 'running' ? 'running' : null, progress: phase === 'running' ? 41 : 0, message: '', error: null },
        dilirank: { started: phase === 'running', jobId: phase === 'running' ? 'job-dilirank' : null, status: phase === 'running' ? 'running' : null, progress: phase === 'running' ? 42 : 0, message: '', error: null },
      },
    };
  }

  it('renders exactly the four structured-source tabs and shared controls', () => {
    const tabs = fixture.nativeElement.querySelectorAll('[role="tab"]');
    expect(tabs).toHaveLength(4);
    expect(fixture.nativeElement.textContent).toContain('LiverTox');
    expect(fixture.nativeElement.textContent).toContain('RxNav');
    expect(fixture.nativeElement.textContent).toContain('DILIrank');
    expect(fixture.nativeElement.textContent).toContain('Terminology');
    expect(Array.from(tabs).map((tab) => (tab as HTMLElement).textContent)).not.toContain('RAG');
    expect(fixture.nativeElement.querySelector('input[type="number"]')).not.toBeNull();
    expect(fixture.nativeElement.textContent).toContain('Update all');
    const sourceProgressbars = fixture.nativeElement.querySelectorAll(
      '.inspection-update-all-source [role="progressbar"]',
    );
    expect(sourceProgressbars).toHaveLength(4);
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
      .find((button) => button.textContent?.trim() === 'Update all') as HTMLButtonElement;
    const cancelButton = buttons
      .find((button) => button.textContent?.trim() === 'Cancel') as HTMLButtonElement;
    expect(updateButton.disabled).toBe(true);
    expect(cancelButton.disabled).toBe(false);
    expect(fixture.nativeElement.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('42');
    expect(fixture.nativeElement.querySelector('[aria-label="RxNav update progress"]')?.getAttribute('aria-valuenow')).toBe('44');
  });
});
