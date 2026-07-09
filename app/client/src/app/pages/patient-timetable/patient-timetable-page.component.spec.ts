import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { InspectionSessionTimeline } from '../../core/models/types';
import { PatientTimetablePageComponent } from './patient-timetable-page.component';

describe('PatientTimetablePageComponent', () => {
  let fixture: ComponentFixture<PatientTimetablePageComponent>;
  let component: PatientTimetablePageComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PatientTimetablePageComponent],
      providers: [
        provideRouter([]),
        {
          provide: ActivatedRoute,
          useValue: {
            paramMap: of(convertToParamMap({ sessionId: '12' })),
          },
        },
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(PatientTimetablePageComponent);
    component = fixture.componentInstance;
  });

  it('describes cloud-generated timelines correctly', () => {
    const timeline: InspectionSessionTimeline = {
      timeline_id: 7,
      session_id: 12,
      generated_at: '2026-07-09T08:00:00Z',
      generation_status: 'llm_generated',
      generation_note: null,
      source_model: 'gpt-4.1-mini',
      source_kind: 'cloud',
      model_provider: 'openai',
      events: [],
    };

    component.timeline.set(timeline);

    expect(component.generationNote()).toBe(
      'Generated with the configured cloud timeline extraction model.',
    );
  });

  it('renders a warning when source evidence is missing', () => {
    const timeline: InspectionSessionTimeline = {
      timeline_id: 8,
      session_id: 12,
      generated_at: '2026-07-09T08:00:00Z',
      generation_status: 'llm_generated',
      generation_note: null,
      source_model: 'llama3.1',
      source_kind: 'local',
      model_provider: 'ollama',
      events: [
        {
          event_id: 'event-1',
          title: 'Therapy started',
          description: 'No source snippet preserved.',
          event_type: 'therapy',
          timing_type: 'uncertain',
          event_date: null,
          relative_time: null,
          extracted_timing_text: null,
          source_evidence: null,
          linked_patient_event_ids: [],
          source: 'fallback_parser',
          confidence: null,
          confidence_rationale: null,
          sort_order: 0,
        },
      ],
    };

    component.timeline.set(timeline);
    component.selectedEventId.set('event-1');
    fixture.detectChanges();

    const root = fixture.nativeElement as HTMLElement;
    const sourceEvidence = root.querySelector<HTMLElement>('.source-evidence');

    expect(sourceEvidence?.classList.contains('is-warning')).toBe(true);
    expect(sourceEvidence?.textContent).toContain(
      'Missing source evidence. Do not treat this event as clinically grounded chronology.',
    );
  });
});
