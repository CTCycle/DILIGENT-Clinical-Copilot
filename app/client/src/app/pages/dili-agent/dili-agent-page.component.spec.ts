import { ComponentFixture, TestBed } from '@angular/core/testing';
import { vi } from 'vitest';

import { DiliAgentPageComponent } from './dili-agent-page.component';

describe('DiliAgentPageComponent', () => {
  let fixture: ComponentFixture<DiliAgentPageComponent>;
  let component: DiliAgentPageComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [DiliAgentPageComponent] }).compileComponents();
    fixture = TestBed.createComponent(DiliAgentPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  it('canStartSession false when visit date missing', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '', clinicalInput: 'word '.repeat(60) },
    });
    expect(component.canStartSession()).toBeFalsy();
  });

  it('canStartSession false when input too short', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '2025-01-01', clinicalInput: 'word '.repeat(59) },
    });
    expect(component.canStartSession()).toBeFalsy();
  });

  it('canStartSession true when all preconditions are met', () => {
    component.stateService.updateDiliAgent({
      form: { ...component.vm.form, visitDate: '2025-01-01', clinicalInput: 'word '.repeat(60) },
      settings: { ...component.vm.settings, provider: 'openai' },
    });
    expect(component.canStartSession()).toBeTruthy();
  });

  it('allows an active run to be stopped even while the start click debounce is still active', () => {
    const stopSessionSpy = vi.spyOn(component, 'stopSession').mockResolvedValue();
    component.stateService.updateDiliAgent({
      isRunning: true,
      jobId: 'job-123',
    });
    (component as unknown as { runControlDebounced: boolean }).runControlDebounced = true;

    component.runOrStop();

    expect(stopSessionSpy).toHaveBeenCalled();
  });
});
