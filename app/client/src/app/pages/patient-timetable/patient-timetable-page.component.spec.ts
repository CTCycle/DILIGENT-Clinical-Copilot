import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { InspectionSessionTimeline } from '../../core/models/inspection-types';
import { PatientTimetablePageComponent } from './patient-timetable-page.component';
import { createTimelineScale, normalizeTimelineDate } from './timeline-date';

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

  it('surfaces the classified reason for a fallback timeline', () => {
    component.timeline.set({
      timeline_id: 11,
      session_id: 12,
      generated_at: '2026-07-09T08:00:00Z',
      generation_status: 'fallback',
      generation_note: 'The provider could not be reached. Retry after restoring network access.',
      generation_error_code: 'network_unavailable',
      source_model: 'deepseek-v4-flash',
      source_kind: 'cloud',
      model_provider: 'opencode_go',
      events: [],
    });

    expect(component.generationErrorLabel()).toBe('Provider network unavailable');
    expect(component.generationNote()).toContain('provider could not be reached');
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

  it('uses explicit partial-date precision without browser partial-date parsing', () => {
    const resolver = (component as unknown as { resolveEventDate: (value: string | null) => unknown }).resolveEventDate;
    const label = (component as unknown as { precisionLabel: (value: string | null) => string }).precisionLabel.bind(component);

    expect(resolver.call(component, '2025-02-03')).toEqual({ date: new Date(Date.UTC(2025, 1, 3)), precision: 'day' });
    expect(resolver.call(component, '2025-02')).toEqual({ date: new Date(Date.UTC(2025, 1, 1)), precision: 'month' });
    expect(resolver.call(component, '2025')).toEqual({ date: new Date(Date.UTC(2025, 0, 1)), precision: 'year' });
    expect(resolver.call(component, '2025-99')).toBeNull();
    expect(resolver.call(component, null)).toBeNull();
    expect(label('2025-02')).toBe('Month precision');
  });

  it('filters evidence, keeps uncertain events exclusive, and supports category collapse', () => {
    component.timeline.set({
      timeline_id: 9, session_id: 12, generated_at: '2026-07-09T08:00:00Z', events: [
        { event_id: 'therapy', title: 'Therapy', description: null, event_type: 'therapy', timing_type: 'explicit_date', event_date: '2025-01-01', relative_time: null, extracted_timing_text: null, source_evidence: 'source', linked_patient_event_ids: [], source: null, confidence: null, confidence_rationale: null, sort_order: 0 },
        { event_id: 'uncertain', title: 'Uncertain', description: null, event_type: 'therapy', timing_type: 'uncertain', event_date: null, relative_time: null, extracted_timing_text: null, source_evidence: null, linked_patient_event_ids: [], source: null, confidence: null, confidence_rationale: null, sort_order: 1 },
      ],
    });

    expect(component.timelineGroups().find((group) => group.isUnanchored)?.events.map((event) => event.event_id)).toEqual(['uncertain']);
    component.setEvidenceFilter('with_evidence');
    expect(component.filteredEvents().map((event) => event.event_id)).toEqual(['therapy']);
    component.toggleLaneCollapsed('therapy');
    expect(component.laneSummaries().find((lane) => lane.id === 'therapy')?.collapsed).toBe(true);
    expect(component.timelineGroups()).toEqual([]);
  });

  it('guards typed filter and density values', () => {
    component.setEvidenceFilter('with_evidence');
    component.setDensity('dense');
    component.setEvidenceFilter('not-a-filter');
    component.setDensity('not-a-density');

    expect(component.evidenceFilter()).toBe('with_evidence');
    expect(component.density()).toBe('dense');
  });

  it('groups same-day events and keeps undated events in a review group', () => {
    component.timeline.set({
      timeline_id: 10,
      session_id: 12,
      generated_at: '2026-07-09T08:00:00Z',
      events: [
        { event_id: 'first', title: 'First', description: 'One', event_type: 'disease', timing_type: 'explicit_date', event_date: '2025-02-03', relative_time: null, extracted_timing_text: null, source_evidence: 'source', linked_patient_event_ids: [], source: 'anamnesis', confidence: 0.9, confidence_rationale: null, sort_order: 0 },
        { event_id: 'second', title: 'Second', description: 'Two', event_type: 'lab', timing_type: 'explicit_date', event_date: '2025-02-03', relative_time: null, extracted_timing_text: null, source_evidence: 'source', linked_patient_event_ids: [], source: 'laboratory history', confidence: 0.8, confidence_rationale: null, sort_order: 1 },
        { event_id: 'undated', title: 'Undated', description: 'Three', event_type: 'other', timing_type: 'uncertain', event_date: null, relative_time: 'After treatment', extracted_timing_text: null, source_evidence: 'source', linked_patient_event_ids: [], source: 'anamnesis', confidence: null, confidence_rationale: null, sort_order: 2 },
      ],
    });

    expect(component.timelineGroups().map((group) => [group.label, group.events.length])).toEqual([
      ['Feb 3, 2025', 2],
      ['Date not reported', 1],
    ]);
    expect(component.eventDateSummary(component.timeline()!.events[2])).toBe('No canonical date · After treatment');
  });

  it('keeps January 5 on the same UTC scale as the month boundaries', () => {
    const start = normalizeTimelineDate('2024-12-01');
    const event = normalizeTimelineDate('2025-01-05');
    const end = normalizeTimelineDate('2025-02-01');

    expect(start && event && end).toBeTruthy();
    const scale = createTimelineScale(start!.startDay, end!.endDay);
    expect(scale.toPercent(event!.startDay)).toBeCloseTo(56.4516, 3);
    expect(scale.toPercent(normalizeTimelineDate('2025-01-01')!.startDay)).toBeLessThan(scale.toPercent(event!.startDay));
    expect(scale.toPercent(normalizeTimelineDate('2025-02-01')!.startDay)).toBeGreaterThan(scale.toPercent(event!.startDay));
  });
});
