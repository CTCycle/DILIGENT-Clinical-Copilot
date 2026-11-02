// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiHttpError, fetchWithTimeout } from './http-api';
import { fetchSessionAttachmentBlob } from './clinical-sessions-api';
import { loadStoredDesktopSession, setDesktopSessionToken, storeDesktopSession } from './desktop-session';
import { TestBed } from '@angular/core/testing';
import { DesktopDialogService } from './desktop-dialog.service';
import { MarkdownRendererService } from './markdown-renderer.service';
import { NotificationService } from './notification.service';
import { API_BASE_URL } from '../constants';
import { cancelInspectionSessionTimelineJob } from './session-timeline-api';

describe('desktop session reload persistence', () => {
  afterEach(() => {
    window.sessionStorage.clear();
  });

  it('restores the loopback API origin and token after the bootstrap URL is gone', () => {
    storeDesktopSession({ apiOrigin: 'http://127.0.0.1:51884', token: 'private-session-token' });

    expect(loadStoredDesktopSession()).toEqual({
      apiOrigin: 'http://127.0.0.1:51884',
      token: 'private-session-token',
    });
  });

  it('ignores missing or malformed stored sessions', () => {
    expect(loadStoredDesktopSession()).toBeNull();

    window.sessionStorage.setItem('diligent.desktop-session', '{"apiOrigin":"http://127.0.0.1:1"}');
    expect(loadStoredDesktopSession()).toBeNull();

    window.sessionStorage.setItem('diligent.desktop-session', 'not json');
    expect(loadStoredDesktopSession()).toBeNull();
  });
});

describe('HTTP API desktop session header', () => {
  afterEach(() => {
    setDesktopSessionToken('');
    vi.unstubAllGlobals();
  });

  it('adds the in-memory session token and preserves request headers', async () => {
    setDesktopSessionToken('private-session-token');
    const fetchMock = vi.fn().mockResolvedValue({ ok: true } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await fetchWithTimeout('/api/settings', {
      headers: { 'Content-Type': 'application/json' },
    });

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = new Headers(request.headers);
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(headers.get('X-DILIGENT-Desktop-Session')).toBe('private-session-token');
  });

  it('does not add a session header in source mode', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true } as Response);
    vi.stubGlobal('fetch', fetchMock);

    await fetchWithTimeout('/api/settings', { headers: { Accept: 'application/json' } });

    const request = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(request.headers).has('X-DILIGENT-Desktop-Session')).toBe(false);
  });
});

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

describe('MarkdownRendererService', () => {
  it('renders headings and inline formatting', () => {
    const service = new MarkdownRendererService();
    const rendered = service.render('# Title\n\n**bold** and *italic*');
    expect(rendered.html).toContain('<h1>Title</h1>');
    expect(rendered.html).toContain('<strong>bold</strong>');
    expect(rendered.html).toContain('<em>italic</em>');
  });

  it('blocks javascript links and escapes html', () => {
    const service = new MarkdownRendererService();
    const rendered = service.render('[x](javascript:alert(1)) <script>x</script>');
    expect(rendered.html).not.toContain('javascript:');
    expect(rendered.html).toContain('&lt;script&gt;');
  });
});

describe('NotificationService', () => {
  it('publishes a transient notification and stores a clean history entry', () => {
    const service = new NotificationService();

    service.notify('[ERROR] Provider connection failed.', 'error', 0);

    expect(service.active()?.message).toBe('Provider connection failed.');
    expect(service.active()?.tone).toBe('error');
    expect(service.history()).toHaveLength(1);
    expect(service.unreadCount()).toBe(1);
  });

  it('marks history as read without removing entries', () => {
    const service = new NotificationService();
    service.notify('Configuration saved.', 'success', 0);
    service.notify('Catalog refreshed.', 'info', 0);

    service.markAllRead();

    expect(service.history()).toHaveLength(2);
    expect(service.unreadCount()).toBe(0);
  });

  it('clears history independently from the active popup', () => {
    const service = new NotificationService();
    service.notify('Configuration saved.', 'success', 0);

    service.clearHistory();

    expect(service.history()).toEqual([]);
    expect(service.active()?.message).toBe('Configuration saved.');
  });
});

describe('session timeline API', () => {
  afterEach(() => vi.restoreAllMocks());

  it('requests cancellation through the existing timeline job endpoint', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      JSON.stringify({ job_id: 'job/1', success: true, message: 'Cancellation requested' }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ));

    const response = await cancelInspectionSessionTimelineJob(42, 'job/1');

    expect(response.success).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toBe(
      `${API_BASE_URL}/inspection/sessions/42/timeline-jobs/job%2F1`,
    );
    expect(init?.method).toBe('DELETE');
  });
});

describe('session attachment download API', () => {
  afterEach(() => {
    setDesktopSessionToken('');
    vi.restoreAllMocks();
  });

  it('fetches the attachment blob with the desktop session header', async () => {
    setDesktopSessionToken('private-session-token');
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('pdf-bytes'));

    const blob = await fetchSessionAttachmentBlob(42, 7);

    expect(await blob.text()).toBe('pdf-bytes');
    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toBe(`${API_BASE_URL}/inspection/sessions/42/attachments/7`);
    expect(new Headers(init?.headers).get('X-DILIGENT-Desktop-Session')).toBe('private-session-token');
  });

  it('raises ApiHttpError with the server detail on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      JSON.stringify({ detail: 'Attachment not found.' }),
      { status: 404, headers: { 'Content-Type': 'application/json' } },
    ));

    const failure = fetchSessionAttachmentBlob(42, 7);

    await expect(failure).rejects.toBeInstanceOf(ApiHttpError);
    await expect(failure).rejects.toMatchObject({ statusCode: 404 });
  });
});
