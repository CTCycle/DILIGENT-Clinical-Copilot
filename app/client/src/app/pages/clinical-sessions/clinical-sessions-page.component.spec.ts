import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ClinicalSessionDetail,
  InspectionSessionItem,
} from '../../core/models/inspection-types';
import { SessionVersionSummary } from '../../core/models/revision-types';
import { ClinicalSessionsPageComponent } from './clinical-sessions-page.component';

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

function sessionDetail(sessionId: number): ClinicalSessionDetail {
  return {
    session_id: sessionId,
    patient_name: `Patient ${sessionId}`,
    visit_date: '2026-09-17',
    session_timestamp: '2026-09-17T08:00:00Z',
    version: 1,
    original_session_id: null,
    status: 'successful',
    text_extraction_model: 'parser',
    clinical_model: 'clinical',
    metadata: {},
    sections: {},
    session_text: '',
    source_clinical_text: '',
    result_payload: {},
    report: 'Report',
    official_report_text: 'Report',
    manual_edit_history: [],
  };
}

function sessionItem(sessionId: number): InspectionSessionItem {
  return {
    session_id: sessionId,
    patient_name: `Patient ${sessionId}`,
    session_timestamp: '2026-09-17T08:00:00Z',
    version: 1,
    original_session_id: null,
    status: 'successful',
    total_duration: null,
    has_report: true,
    has_timeline: false,
    can_generate_timeline: true,
  };
}

function revisionVersion(
  sessionId: number,
  versionStatus: SessionVersionSummary['version_status'],
  overrides: Partial<SessionVersionSummary> = {},
): SessionVersionSummary {
  return {
    version_id: 100 + sessionId,
    session_id: versionStatus === 'cancelled' ? null : sessionId,
    root_session_id: sessionId,
    source_version_id: 1,
    revision_version_id: 100 + sessionId,
    version_number: 2,
    version_status: versionStatus,
    revision_kind: 'llm_assisted_revision',
    llm_qa_status: versionStatus === 'llm_qa_passed' ? 'passed' : 'not_run',
    clinical_review_status: 'not_reviewed',
    pipeline_run_id: null,
    model_configuration: {},
    created_at: '2026-09-17T08:00:00Z',
    updated_at: '2026-09-17T08:01:00Z',
    completed_at: '2026-09-17T08:01:00Z',
    ...overrides,
  };
}

describe('ClinicalSessionsPageComponent request generations', () => {
  let fixture: ComponentFixture<ClinicalSessionsPageComponent>;
  let component: ClinicalSessionsPageComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ClinicalSessionsPageComponent],
      providers: [provideRouter([])],
    }).compileComponents();
    fixture = TestBed.createComponent(ClinicalSessionsPageComponent);
    component = fixture.componentInstance;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fixture.destroy();
  });

  it('keeps the newest session list when responses complete out of order', async () => {
    const firstResponse = deferred<Response>();
    const secondResponse = deferred<Response>();
    let listCall = 0;
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes('/inspection/sessions?')) {
        listCall += 1;
        return listCall === 1 ? firstResponse.promise : secondResponse.promise;
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    component.selected.set(sessionDetail(2));
    const firstLoad = component.loadSessions();
    await Promise.resolve();
    const secondLoad = component.loadSessions();
    await Promise.resolve();

    secondResponse.resolve(jsonResponse({
      items: [sessionItem(2)], total: 1, offset: 0, limit: 100,
    }));
    await secondLoad;
    firstResponse.resolve(jsonResponse({
      items: [sessionItem(1)], total: 1, offset: 0, limit: 100,
    }));
    await firstLoad;

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(component.sessions().map((session) => session.session_id)).toEqual([2]);
    expect(component.loading()).toBe(false);
  });

  it('ignores a stale session detail response after navigation', async () => {
    const firstResponse = deferred<Response>();
    const secondResponse = deferred<Response>();
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/inspection/sessions/1')) return firstResponse.promise;
      if (url.endsWith('/inspection/sessions/2')) return secondResponse.promise;
      if (url.includes('/versions/')) return jsonResponse({ items: [] });
      if (url.endsWith('/versions')) return jsonResponse({ items: [] });
      throw new Error(`Unexpected request: ${url}`);
    });

    const firstOpen = component.openSession(1);
    await Promise.resolve();
    const secondOpen = component.openSession(2);
    await Promise.resolve();

    secondResponse.resolve(jsonResponse(sessionDetail(2)));
    await secondOpen;
    firstResponse.resolve(jsonResponse(sessionDetail(1)));
    await firstOpen;
    await Promise.resolve();

    expect(fetchSpy).toHaveBeenCalled();
    expect(component.selected()?.session_id).toBe(2);
    expect(component.editorText()).toContain('Report');
    expect(component.detailLoading()).toBe(false);
    expect(component.detailError()).toBeNull();
  });

  it('restores completed revision review state and keeps cancelled revisions terminal', async () => {
    component.selected.set(sessionDetail(12));
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/12/versions')) {
        return jsonResponse({ items: [revisionVersion(12, 'llm_qa_passed')] });
      }
      if (url.endsWith('/sessions/12/versions/112/artifacts')) {
        return jsonResponse({ items: [] });
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    await (component as unknown as {
      loadPersistedRevision: (sessionId: number, generation: number) => Promise<void>;
    }).loadPersistedRevision(12, 0);

    expect(component.revisionVersionStatus()).toBe('llm_qa_passed');
    expect(component.revisionReviewAvailable()).toBe(true);

    fetchSpy.mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/12/versions')) {
        return jsonResponse({ items: [
          revisionVersion(12, 'cancelled', {
            version_id: 1,
            revision_version_id: 1,
            session_id: 12,
            source_version_id: null,
            version_number: 1,
            version_status: 'current',
            revision_kind: 'manual_edit',
            llm_qa_status: 'not_run',
          }),
          revisionVersion(12, 'cancelled'),
        ] });
      }
      if (url.endsWith('/sessions/12/versions/112/artifacts')) {
        return jsonResponse({ items: [] });
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    await (component as unknown as {
      loadPersistedRevision: (sessionId: number, generation: number) => Promise<void>;
    }).loadPersistedRevision(12, 0);

    expect(component.revisionVersionStatus()).toBe('cancelled');
    expect(component.revisionReviewAvailable()).toBe(false);
    expect(component.revisionRunning()).toBe(false);
    expect(fetchSpy.mock.calls.some(([input]) => String(input).includes('/revision/jobs/'))).toBe(false);
  });

  it('restores the selected session revision instead of an unrelated newer root draft', async () => {
    component.selected.set(sessionDetail(22));
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/22/versions')) {
        return jsonResponse({ items: [
          revisionVersion(22, 'llm_qa_passed', {
            version_id: 139,
            revision_version_id: 139,
            version_number: 6,
            updated_at: '2026-09-16T16:49:58Z',
          }),
          revisionVersion(19, 'requires_human_review', {
            version_id: 146,
            revision_version_id: 146,
            session_id: null,
            source_version_id: 26,
            root_session_id: 19,
            version_number: 10,
            updated_at: '2026-09-19T15:45:18Z',
          }),
        ] });
      }
      if (url.endsWith('/sessions/22/versions/139/artifacts')) {
        return jsonResponse({ items: [] });
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    await (component as unknown as {
      loadPersistedRevision: (sessionId: number, generation: number) => Promise<void>;
    }).loadPersistedRevision(22, 0);

    expect(component.revisionVersionId()).toBe(139);
    expect(component.revisionVersionStatus()).toBe('llm_qa_passed');
    expect(fetchSpy.mock.calls.some(([input]) => String(input).endsWith('/versions/146/artifacts'))).toBe(false);
  });

  it('clears a stale revision draft after saving a manual report edit', async () => {
    component.selected.set(sessionDetail(19));
    component.editorText.set('Edited report');
    component.revisionVersionId.set(146);
    component.revisionVersionStatus.set('requires_human_review');
    component.revisionDraftReport.set('Stale draft');
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith('/sessions/19/report')) {
        return jsonResponse({ session: sessionDetail(19), audit: {} });
      }
      if (url.endsWith('/sessions/19/versions')) {
        return jsonResponse({ items: [] });
      }
      throw new Error(`Unexpected request: ${url}`);
    });

    await component.saveManualReportEdit();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(component.revisionVersionId()).toBeNull();
    expect(component.revisionVersionStatus()).toBeNull();
    expect(component.revisionDraftReport()).toBe('');
    expect(fetchSpy.mock.calls.some(([input]) => String(input).endsWith('/sessions/19/versions'))).toBe(true);
  });
});
