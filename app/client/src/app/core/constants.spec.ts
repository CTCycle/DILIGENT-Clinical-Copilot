// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { afterEach, describe, expect, it } from 'vitest';

import { API_BASE_URL, configureDesktopApiOrigin } from './constants';

describe('desktop API runtime configuration', () => {
  afterEach(() => {
    configureDesktopApiOrigin(null);
  });

  it('uses the shell supplied loopback origin for the packaged API', () => {
    configureDesktopApiOrigin('http://127.0.0.1:49152');
    expect(API_BASE_URL).toBe('http://127.0.0.1:49152/api');
  });

  it('rejects API origins outside the loopback HTTP boundary', () => {
    expect(() => configureDesktopApiOrigin('https://127.0.0.1:49152')).toThrow(
      /local loopback origin/,
    );
    expect(() => configureDesktopApiOrigin('http://example.org:49152')).toThrow(
      /local loopback origin/,
    );
    expect(() => configureDesktopApiOrigin('http://127.0.0.1:49152/api')).toThrow(
      /local loopback origin/,
    );
  });
});
