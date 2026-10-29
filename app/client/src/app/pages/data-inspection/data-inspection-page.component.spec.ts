// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DesktopDialogService } from '../../core/services/desktop-dialog.service';
import { InspectionUpdateJobTrackerService } from '../../core/state/inspection-update-job-tracker.service';
import { DataInspectionPageComponent } from './data-inspection-page.component';

describe('DataInspectionPageComponent folder selection', () => {
  let fixture: ComponentFixture<DataInspectionPageComponent>;
  let component: DataInspectionPageComponent;
  let desktopDialog: {
    isTauriSurface: ReturnType<typeof vi.fn>;
    openDirectory: ReturnType<typeof vi.fn>;
  };
  let tracker: {
    targetState: ReturnType<typeof signal>;
    updateAllState: ReturnType<typeof signal>;
    configureRefreshers: ReturnType<typeof vi.fn>;
    discover: ReturnType<typeof vi.fn>;
    start: ReturnType<typeof vi.fn>;
    startAll: ReturnType<typeof vi.fn>;
    cancelAll: ReturnType<typeof vi.fn>;
  };
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    desktopDialog = {
      isTauriSurface: vi.fn(),
      openDirectory: vi.fn(),
    };
    tracker = {
      targetState: signal({
        rxnav: { jobId: null, running: false, progress: 0, message: '', error: null },
        livertox: { jobId: null, running: false, progress: 0, message: '', error: null },
        terminology: { jobId: null, running: false, progress: 0, message: '', error: null },
        dilirank: { jobId: null, running: false, progress: 0, message: '', error: null },
        rag: { jobId: null, running: false, progress: 0, message: '', error: null },
      }),
      updateAllState: signal({
        runId: null,
        phase: 'idle',
        progress: 0,
        message: '',
        cancelRequested: false,
        targets: {
          livertox: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
          rxnav: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
          terminology: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
          dilirank: { started: false, jobId: null, status: null, progress: 0, message: '', error: null },
        },
      }),
      configureRefreshers: vi.fn(),
      discover: vi.fn().mockResolvedValue(undefined),
      start: vi.fn().mockResolvedValue(undefined),
      startAll: vi.fn().mockResolvedValue(undefined),
      cancelAll: vi.fn().mockResolvedValue(undefined),
    };
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await TestBed.configureTestingModule({
      imports: [DataInspectionPageComponent],
      providers: [
        { provide: DesktopDialogService, useValue: desktopDialog },
        { provide: InspectionUpdateJobTrackerService, useValue: tracker },
      ],
    }).compileComponents();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function createComponent(isTauriSurface: boolean): void {
    desktopDialog.isTauriSurface.mockReturnValue(isTauriSurface);
    fixture = TestBed.createComponent(DataInspectionPageComponent);
    component = fixture.componentInstance;
  }

  function jsonResponse(payload: unknown, status = 200): Response {
    return new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  it('uses the native directory picker and stores paths with spaces and non-ASCII characters', async () => {
    createComponent(true);
    desktopDialog.openDirectory.mockResolvedValue('C:\\Clinical Documents\\RAG Démo');

    await component.openRagFolderPicker();

    expect(desktopDialog.openDirectory).toHaveBeenCalledWith('Select RAG documents folder');
    expect(component.ragSelectedFolderPath()).toBe('C:\\Clinical Documents\\RAG Démo');
    expect(component.ragError()).toBeNull();
  });

  it('leaves the current selection unchanged when the native picker is cancelled', async () => {
    createComponent(true);
    component.ragSelectedFolderPath.set('C:\\Existing RAG');
    component.ragError.set(null);
    desktopDialog.openDirectory.mockResolvedValue(null);

    await component.openRagFolderPicker();

    expect(component.ragSelectedFolderPath()).toBe('C:\\Existing RAG');
    expect(component.ragError()).toBeNull();
  });

  it('reports native picker failures without replacing the current path', async () => {
    createComponent(true);
    component.ragSelectedFolderPath.set('C:\\Existing RAG');
    desktopDialog.openDirectory.mockRejectedValue(new Error('dialog unavailable'));

    await component.openRagFolderPicker();

    expect(component.ragSelectedFolderPath()).toBe('C:\\Existing RAG');
    expect(component.ragError()).toBe('Unable to open the native RAG folder picker.');
  });

  it('opens the browser folder modal and loads server-canonical roots', async () => {
    createComponent(false);
    fetchMock.mockResolvedValue(
      jsonResponse({
        current_path: '',
        parent_path: null,
        items: [{ name: 'G:\\', path: 'G:\\', is_dir: true }],
        drives: ['G:\\'],
      }),
    );

    await component.openRagFolderPicker();

    expect(component.ragFolderBrowserOpen()).toBe(true);
    expect(component.ragFolderBrowsePath()).toBe('');
    expect(component.ragFolderBrowseItems()[0].path).toBe('G:\\');
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/inspection/rag/browse'),
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('navigates with canonical paths and propagates the selected folder', async () => {
    createComponent(false);
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          current_path: '',
          parent_path: null,
          items: [{ name: 'G:\\', path: 'G:\\', is_dir: true }],
          drives: ['G:\\'],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          current_path: 'G:\\RAG Démo',
          parent_path: 'G:\\',
          items: [],
          drives: ['G:\\'],
        }),
      );

    await component.openRagFolderPicker();
    component.navigateRagFolder('G:\\RAG Démo');
    await vi.waitFor(() => {
      expect(component.ragFolderBrowsePath()).toBe('G:\\RAG Démo');
    });

    component.selectRagFolder();

    expect(component.ragSelectedFolderPath()).toBe('G:\\RAG Démo');
    expect(component.ragFolderBrowserOpen()).toBe(false);
    const browseUrls = fetchMock.mock.calls
      .map((call) => call[0] as string)
      .filter((url) => url.includes('/inspection/rag/browse'));
    const lastUrl = browseUrls.at(-1) as string;
    expect(browseUrls).toHaveLength(2);
    expect(new URL(lastUrl, window.location.origin).searchParams.get('path')).toBe(
      'G:\\RAG Démo',
    );
  });

  it('passes the selected canonical folder to the RAG update payload', async () => {
    createComponent(false);
    component.ragSelectedFolderPath.set('G:\\RAG Démo');
    fetchMock.mockResolvedValue(
      jsonResponse({
        target: 'rag',
        defaults: {},
        allowed_fields: [],
        summary: {},
        read_only: true,
      }),
    );

    await component.openUpdateModal('rag');
    await component.startUpdateJob();

    expect(tracker.start).toHaveBeenCalledWith({
      target: 'rag',
      payload: { documents_path: 'G:\\RAG Démo' },
    });
  });

  it('loads all structured source configurations and starts only those sources', async () => {
    createComponent(false);
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/inspection/jobs')) {
        return jsonResponse({ jobs: [] });
      }
      const target = url.split('/').at(-2);
      const defaults = target === 'livertox'
        ? { livertox_monograph_max_workers: 4, livertox_archive: 'livertox.tar.gz', redownload: false }
        : target === 'rxnav' || target === 'terminology'
          ? { force_full: false }
          : { redownload: false };
      return jsonResponse({
        target,
        defaults,
        allowed_fields: Object.keys(defaults),
        summary: {},
        read_only: false,
      });
    });

    await component.openUpdateAllModal();
    component.updateUpdateAllConfig({
      target: 'dilirank',
      change: { key: 'redownload', value: true },
    });
    await component.startUpdateAll();

    expect(component.updateAllModalOpen()).toBe(true);
    expect(tracker.startAll).toHaveBeenCalledWith({
      livertox: {
        target: 'livertox',
        payload: {
          livertox_monograph_max_workers: 4,
          livertox_archive: 'livertox.tar.gz',
          redownload: false,
        },
      },
      rxnav: {
        target: 'rxnav',
        payload: {
          force_full: false,
        },
      },
      terminology: {
        target: 'terminology',
        payload: { force_full: false },
      },
      dilirank: {
        target: 'dilirank',
        payload: { redownload: true },
      },
    });
    expect(tracker.startAll.mock.calls[0][0]).not.toHaveProperty('rag');
  });

  it('resolves a name and records a reviewer confirmation for a candidate', async () => {
    createComponent(false);
    const ambiguous = {
      original: 'hydroxazine',
      normalized: 'hydroxazine',
      strengths: [],
      qualifiers: [],
      status: 'ambiguous',
      accepted: false,
      method: 'fuzzy',
      drug_id: null,
      preferred_name: null,
      concept_type: null,
      ingredients: [],
      precise_forms: [],
      matched_alias: null,
      alias_kind: null,
      source: null,
      source_version: null,
      country: null,
      strength_supported: null,
      candidates: [{ drug_id: 3, name: 'hydroxyzine', concept_type: 'ingredient', matched_alias: 'hydroxyzine', alias_kind: 'canonical', source: 'rxnorm', country: null, edit_distance: 1, similarity: 0.9 }],
      review_reasons: ['lookalike_within_edit_bound'],
      external_calls: 0,
      terminology_revision: 4,
    };
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/terminology/aliases/confirm')) {
        return jsonResponse({ alias: 'hydroxazine', drug_id: 3, validation_status: 'confirmed' });
      }
      if (url.endsWith('/terminology/resolve')) {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ name: 'hydroxazine', allow_external: false });
        return jsonResponse(ambiguous);
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.spyOn(globalThis, 'confirm').mockReturnValue(true);

    component.resolveInput.set(' hydroxazine ');
    component.resolveAllowExternal.set(false);
    await component.resolveName();
    expect(component.resolveResult()?.status).toBe('ambiguous');

    await component.decideAlias('confirm', 3, 'hydroxyzine');

    const confirmCall = fetchMock.mock.calls.find((call) => String(call[0]).endsWith('/aliases/confirm'));
    expect(JSON.parse(String((confirmCall?.[1] as RequestInit).body))).toEqual({ alias: 'hydroxazine', drug_id: 3 });
    expect(component.aliasDecisionMessage()).toBe('"hydroxazine" confirmed for hydroxyzine.');
  });

  it('keeps the modal open and exposes a safe browse error', async () => {
    createComponent(false);
    fetchMock.mockResolvedValue(
      jsonResponse(
        { detail: 'The selected folder path is invalid or cannot be read.' },
        422,
      ),
    );

    await component.openRagFolderPicker();

    expect(component.ragFolderBrowserOpen()).toBe(true);
    expect(component.ragFolderBrowseError()).toBe(
      '[ERROR] The selected folder path is invalid or cannot be read.',
    );
    component.closeRagFolderBrowser();
    expect(component.ragSelectedFolderPath()).toBe('');
  });
});
