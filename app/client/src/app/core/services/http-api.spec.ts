// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { afterEach, describe, expect, it, vi } from 'vitest';

import { fetchWithTimeout } from './http-api';
import { setDesktopSessionToken } from './desktop-session';

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
