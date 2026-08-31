import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { isTauri } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';

import { DesktopDialogService } from './desktop-dialog.service';

vi.mock('@tauri-apps/api/core', () => ({
  isTauri: vi.fn(),
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn(),
}));

describe('DesktopDialogService', () => {
  let service: DesktopDialogService;

  beforeEach(() => {
    vi.resetAllMocks();
    service = TestBed.inject(DesktopDialogService);
  });

  it('does not invoke the Tauri dialog from a browser surface', async () => {
    vi.mocked(isTauri).mockReturnValue(false);

    await expect(service.openDirectory()).resolves.toBeNull();
    expect(open).not.toHaveBeenCalled();
  });

  it('returns a selected directory and forwards the picker title', async () => {
    vi.mocked(isTauri).mockReturnValue(true);
    vi.mocked(open).mockResolvedValue('C:\\Clinical Documents\\RAG');

    await expect(service.openDirectory('Select RAG documents folder')).resolves.toBe(
      'C:\\Clinical Documents\\RAG',
    );
    expect(open).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
      title: 'Select RAG documents folder',
    });
  });

  it('normalizes a single-item selection array', async () => {
    vi.mocked(isTauri).mockReturnValue(true);
    vi.mocked(open).mockResolvedValue(['C:\\RAG Docs'] as never);

    await expect(service.openDirectory()).resolves.toBe('C:\\RAG Docs');
  });

  it('preserves cancellation as null', async () => {
    vi.mocked(isTauri).mockReturnValue(true);
    vi.mocked(open).mockResolvedValue(null);

    await expect(service.openDirectory()).resolves.toBeNull();
  });
});
