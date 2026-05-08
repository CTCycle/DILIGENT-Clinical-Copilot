import { ComponentFixture, TestBed } from '@angular/core/testing';

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
});
