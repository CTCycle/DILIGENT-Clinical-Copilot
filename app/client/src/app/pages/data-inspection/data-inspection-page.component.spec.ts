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
    configureRefreshers: ReturnType<typeof vi.fn>;
    discover: ReturnType<typeof vi.fn>;
    start: ReturnType<typeof vi.fn>;
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
        rag: { jobId: null, running: false, progress: 0, message: '', error: null },
      }),
      configureRefreshers: vi.fn(),
      discover: vi.fn().mockResolvedValue(undefined),
      start: vi.fn().mockResolvedValue(undefined),
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
