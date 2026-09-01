import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DesktopDialogService } from './desktop-dialog.service';

describe('DesktopDialogService', () => {
  let service: DesktopDialogService;
  let tauriInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.resetAllMocks();
    tauriInvoke = vi.fn();
    Object.defineProperty(window, '__TAURI_INTERNALS__', {
      configurable: true,
      value: { invoke: tauriInvoke },
    });
    service = TestBed.inject(DesktopDialogService);
  });

  afterEach(() => {
    delete (window as typeof window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  });

  it('does not invoke the Tauri dialog from a browser surface', async () => {
    vi.spyOn(service, 'isTauriSurface').mockReturnValue(false);

    await expect(service.openDirectory()).resolves.toBeNull();
    expect(tauriInvoke).not.toHaveBeenCalled();
  });

  it('returns a selected directory and forwards the picker title', async () => {
    vi.spyOn(service, 'isTauriSurface').mockReturnValue(true);
    tauriInvoke.mockResolvedValue('C:\\Clinical Documents\\RAG');

    await expect(service.openDirectory('Select RAG documents folder')).resolves.toBe(
      'C:\\Clinical Documents\\RAG',
    );
    expect(tauriInvoke).toHaveBeenCalledWith('plugin:dialog|open', {
      options: {
        directory: true,
        multiple: false,
        title: 'Select RAG documents folder',
      },
    }, undefined);
  });

  it('normalizes a single-item selection array', async () => {
    vi.spyOn(service, 'isTauriSurface').mockReturnValue(true);
    tauriInvoke.mockResolvedValue(['C:\\RAG Docs']);

    await expect(service.openDirectory()).resolves.toBe('C:\\RAG Docs');
  });

  it('preserves cancellation as null', async () => {
    vi.spyOn(service, 'isTauriSurface').mockReturnValue(true);
    tauriInvoke.mockResolvedValue(null);

    await expect(service.openDirectory()).resolves.toBeNull();
  });
});
