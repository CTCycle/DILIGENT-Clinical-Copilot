import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { beforeEach, describe, expect, it, vi } from 'vitest';

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

  beforeEach(async () => {
    desktopDialog = {
      isTauriSurface: vi.fn().mockReturnValue(true),
      openDirectory: vi.fn(),
    };
    const tracker = {
      targetState: signal({
        rxnav: { jobId: null, running: false, progress: 0, message: '', error: null },
        livertox: { jobId: null, running: false, progress: 0, message: '', error: null },
        rag: { jobId: null, running: false, progress: 0, message: '', error: null },
      }),
      configureRefreshers: vi.fn(),
      discover: vi.fn().mockResolvedValue(undefined),
    };

    await TestBed.configureTestingModule({
      imports: [DataInspectionPageComponent],
      providers: [
        { provide: DesktopDialogService, useValue: desktopDialog },
        { provide: InspectionUpdateJobTrackerService, useValue: tracker },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DataInspectionPageComponent);
    component = fixture.componentInstance;
  });

  it('uses the native directory picker and stores paths with spaces and non-ASCII characters', async () => {
    desktopDialog.openDirectory.mockResolvedValue('C:\\Clinical Documents\\RAG Démo');

    await component.openRagFolderPicker();

    expect(desktopDialog.openDirectory).toHaveBeenCalledWith('Select RAG documents folder');
    expect(component.ragSelectedFolderPath()).toBe('C:\\Clinical Documents\\RAG Démo');
    expect(component.ragError()).toBeNull();
  });

  it('leaves the current selection unchanged when the native picker is cancelled', async () => {
    component.ragSelectedFolderPath.set('C:\\Existing RAG');
    component.ragError.set(null);
    desktopDialog.openDirectory.mockResolvedValue(null);

    await component.openRagFolderPicker();

    expect(component.ragSelectedFolderPath()).toBe('C:\\Existing RAG');
    expect(component.ragError()).toBeNull();
  });

  it('reports native picker failures without replacing the current path', async () => {
    component.ragSelectedFolderPath.set('C:\\Existing RAG');
    desktopDialog.openDirectory.mockRejectedValue(new Error('dialog unavailable'));

    await component.openRagFolderPicker();

    expect(component.ragSelectedFolderPath()).toBe('C:\\Existing RAG');
    expect(component.ragError()).toBe('Unable to open the native RAG folder picker.');
  });
});
