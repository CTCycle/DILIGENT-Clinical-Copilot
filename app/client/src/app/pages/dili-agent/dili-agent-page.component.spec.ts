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
});
