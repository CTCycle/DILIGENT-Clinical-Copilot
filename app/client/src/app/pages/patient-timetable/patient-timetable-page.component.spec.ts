// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { InspectionSessionTimeline, InspectionTimelineEvent } from '../../core/models/inspection-types';
import { PatientTimetablePageComponent } from './patient-timetable-page.component';
import { formatDayGap } from './timeline-date';
import { eventDateSummary, precisionLabel } from './timeline-presentation';

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => { resolve = next; });
  return { promise, resolve };
}

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

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
    // jsdom does not implement scrolling; selection scrolls the chosen event into view.
    Element.prototype.scrollIntoView = vi.fn();
    fixture = TestBed.createComponent(PatientTimetablePageComponent);
    component = fixture.componentInstance;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fixture.destroy();
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

  it.each([
    ['timeout', 'Provider timeout'],
    ['authentication', 'Provider authentication rejected'],
    ['rate_limited', 'Provider rate limit reached'],
  ] as const)(
    'labels the %s fallback reason',
    (generation_error_code, expectedLabel) => {
      component.timeline.set({
        timeline_id: 12,
        session_id: 12,
        generated_at: '2026-07-09T08:00:00Z',
        generation_status: 'fallback',
        generation_note: 'Controlled provider failure for timeline recovery QA.',
        generation_error_code,
        source_model: 'deepseek-v4-flash',
        source_kind: 'cloud',
        model_provider: 'opencode_go',
        events: [],
      });

      expect(component.generationErrorLabel()).toBe(expectedLabel);
    },
  );

  function timelineEvent(overrides: Partial<InspectionTimelineEvent> & Pick<InspectionTimelineEvent, 'event_id'>): InspectionTimelineEvent {
    return {
      title: overrides.event_id,
      description: null,
      event_type: 'other',
      timing_type: 'explicit_date',
      event_date: null,
      relative_time: null,
      extracted_timing_text: null,
      source_evidence: 'source',
      linked_patient_event_ids: [],
      source: null,
      confidence: null,
      confidence_rationale: null,
      sort_order: 0,
      ...overrides,
    };
  }

  function setEvents(events: InspectionTimelineEvent[]): void {
    (component as unknown as { applyTimeline: (value: InspectionSessionTimeline) => void }).applyTimeline({
      timeline_id: 9,
      session_id: 12,
      generated_at: '2026-07-09T08:00:00Z',
      events,
    });
  }

  it('renders a warning when source evidence is missing', () => {
    setEvents([
      timelineEvent({ event_id: 'event-1', title: 'Therapy started', event_type: 'therapy', timing_type: 'uncertain', source_evidence: null }),
    ]);
    component.expandAll();
    fixture.detectChanges();

    const sourceEvidence = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.source-evidence');

    expect(sourceEvidence?.classList.contains('is-warning')).toBe(true);
    expect(sourceEvidence?.textContent).toContain(
      'Missing source evidence. Do not treat this event as clinically grounded chronology.',
    );
  });

  it('keeps events collapsed and the undated group closed until expanded on demand', () => {
    setEvents([
      timelineEvent({ event_id: 'dated', event_date: '2025-02-03', description: 'Dated detail' }),
      timelineEvent({ event_id: 'undated', timing_type: 'uncertain' }),
    ]);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    expect(root.querySelectorAll('.event-row').length).toBe(1);
    expect(root.querySelector('app-timeline-event-details')).toBeNull();

    root.querySelector<HTMLButtonElement>('.event-row-toggle')!.click();
    fixture.detectChanges();
    expect(root.querySelector('.event-row-toggle')?.getAttribute('aria-expanded')).toBe('true');
    expect(root.querySelector('app-timeline-event-details')?.textContent).toContain('Dated detail');

    component.expandAll();
    fixture.detectChanges();
    expect(root.querySelectorAll('.event-row').length).toBe(2);
    expect(component.allExpanded()).toBe(true);

    component.collapseAll();
    fixture.detectChanges();
    expect(root.querySelector('app-timeline-event-details')).toBeNull();
  });

  it('formats partial-date precision without browser partial-date parsing', () => {
    expect(precisionLabel('2025-02')).toBe('Month precision');
    expect(precisionLabel('2025-99')).toBe('Not reported');
    expect(eventDateSummary(timelineEvent({ event_id: 'range', event_date: '2024-12-30', event_date_end: '2025-01-03' })))
      .toBe('30 Dec 2024 – 3 Jan 2025');
    expect(eventDateSummary(timelineEvent({ event_id: 'year', event_date: '2025' }))).toBe('2025 · year-level date');
  });

  it('filters uncertain timing and hidden categories', () => {
    setEvents([
      timelineEvent({ event_id: 'therapy', event_type: 'therapy', event_date: '2025-01-01' }),
      timelineEvent({ event_id: 'uncertain', event_type: 'therapy', timing_type: 'uncertain' }),
      timelineEvent({ event_id: 'lab', event_type: 'lab', event_date: '2025-01-02' }),
    ]);

    component.toggleUncertainEvents();
    expect(component.filteredEvents().map((event) => event.event_id)).toEqual(['therapy', 'lab']);
    component.toggleCategory('therapy');
    expect(component.categorySummaries().find((category) => category.id === 'therapy')?.hidden).toBe(true);
    expect(component.filteredEvents().map((event) => event.event_id)).toEqual(['lab']);
    expect(component.visibleCategories()).toEqual(['labs']);
  });

  it('groups by date and precision, labels day gaps, and keeps undated events last', () => {
    setEvents([
      timelineEvent({ event_id: 'first', event_type: 'disease', event_date: '2025-02-03', sort_order: 0 }),
      timelineEvent({ event_id: 'second', event_type: 'lab', event_date: '2025-02-03', sort_order: 1 }),
      timelineEvent({ event_id: 'undated', timing_type: 'uncertain', relative_time: 'After treatment', sort_order: 2 }),
      timelineEvent({ event_id: 'year', event_date: '2025', sort_order: 3 }),
      timelineEvent({ event_id: 'later', event_date: '2025-03-20', sort_order: 4 }),
    ]);

    expect(component.timelineGroups().map((group) => [group.label, group.events.length, group.gapLabel])).toEqual([
      ['2025', 1, null],
      ['3 Feb 2025', 2, null],
      ['20 Mar 2025', 1, '+45 d'],
      ['Date not reported', 1, null],
    ]);
    expect(eventDateSummary(component.timeline()!.events[2])).toBe('No canonical date · After treatment');
  });

  it('formats gaps between timeline columns', () => {
    expect(formatDayGap(0)).toBeNull();
    expect(formatDayGap(5)).toBe('+5 d');
    expect(formatDayGap(91)).toBe('+3 mo');
    expect(formatDayGap(800)).toBe('+2 y');
  });

  it('resolves linked events and selects them across groups', () => {
    setEvents([
      timelineEvent({ event_id: 'start', title: 'Therapy start', event_date: '2025-01-01', linked_patient_event_ids: ['alt', 'missing'] }),
      timelineEvent({ event_id: 'alt', title: 'ALT rise', timing_type: 'uncertain' }),
    ]);

    const [start] = component.orderedEvents();
    expect(component.linkedEvents(start).map((event) => event.event_id)).toEqual(['alt']);
    component.selectEvent(component.linkedEvents(start)[0]);
    expect(component.selectedEventId()).toBe('alt');
    expect(component.isGroupCollapsed(component.undatedGroup()!)).toBe(false);
    expect(component.isExpanded(component.selectedEvent()!)).toBe(true);
  });

  it('renders the horizontal view with dated columns and category lanes', () => {
    setEvents([
      timelineEvent({ event_id: 'therapy', title: 'Amoxicillin', event_type: 'therapy', event_date: '2024-12-30', event_date_end: '2025-01-06' }),
      timelineEvent({ event_id: 'lab', title: 'ALT rise', event_type: 'lab', event_date: '2025-01-06' }),
    ]);
    component.setViewMode('horizontal');
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;

    expect([...root.querySelectorAll('.column-header strong')].map((node) => node.textContent?.trim())).toEqual(['30 Dec 2024', '6 Jan 2025']);
    expect([...root.querySelectorAll('.lane-label')].map((node) => node.textContent?.trim())).toEqual(['Medications', 'Laboratory']);
    expect(root.querySelector('.gap-chip')?.textContent?.trim()).toBe('+7 d');
    expect(root.querySelectorAll('.range-bar').length).toBe(1);

    root.querySelector<HTMLButtonElement>('[data-chip-id="lab"]')!.click();
    fixture.detectChanges();
    expect(component.selectedEventId()).toBe('lab');
    expect(root.querySelector('.selection-panel h3')?.textContent).toContain('ALT rise');
  });

  it('ignores a late timetable response after the route changes', async () => {
    const firstResponse = deferred<Response>();
    const secondResponse = deferred<Response>();
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/1/timelines/11')) return firstResponse.promise;
      if (url.endsWith('/sessions/2/timelines/22')) return secondResponse.promise;
      throw new Error(`Unexpected request: ${url}`);
    });
    const handleRouteChange = (component as unknown as {
      handleRouteChange: (sessionId: number, timelineId: number | null) => Promise<void>;
    }).handleRouteChange.bind(component);

    const firstLoad = handleRouteChange(1, 11);
    await Promise.resolve();
    const secondLoad = handleRouteChange(2, 22);
    await Promise.resolve();

    secondResponse.resolve(jsonResponse({
      timeline_id: 22,
      session_id: 2,
      generated_at: '2026-09-17T10:00:00Z',
      events: [],
    }));
    await secondLoad;
    firstResponse.resolve(jsonResponse({
      timeline_id: 11,
      session_id: 1,
      generated_at: '2026-09-17T09:00:00Z',
      events: [],
    }));
    await firstLoad;

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(component.sessionId()).toBe(2);
    expect(component.timelineId()).toBe(22);
    expect(component.timeline()?.session_id).toBe(2);
    expect(component.loading()).toBe(false);
    expect(component.error()).toBeNull();
  });
});
