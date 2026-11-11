// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it } from 'vitest';
import { API_BASE_URL, configureDesktopApiOrigin } from './constants';
import { ClinicalFormState } from './models/types';
import { buildClinicalPayload, isRecord } from './utils';
import {
  apiBaseUrlDefine,
  buildApiProxyConfig,
  parseEnvText,
  resolveApiBaseUrl,
  resolveRuntimeNetworkSettings,
  rewriteApiRequestPath,
} from '../../../scripts/runtime-config.mjs';

describe('runtime network configuration', () => {
  it('imports every node:fs function the development server launcher calls', () => {
    const source = readFileSync('scripts/ng-serve.mjs', 'utf8');
    const imported = /import \{([^}]*)\} from 'node:fs'/.exec(source)?.[1] ?? '';
    const called = [...source.matchAll(/(\w+Sync)\(/g)].map((match) => match[1]);

    expect(called.length).toBeGreaterThan(0);
    for (const name of new Set(called)) {
      expect(imported).toContain(name);
    }
  });

  it('keeps default development API routing same-origin', () => {
    const settings = resolveRuntimeNetworkSettings({}, {});

    expect(settings).toEqual({
      apiHost: '127.0.0.1',
      apiPort: 7690,
      uiHost: '127.0.0.1',
      uiPort: 9847,
      apiBaseUrl: '/api',
    });
    expect(buildApiProxyConfig(settings.apiBaseUrl, settings.apiHost, settings.apiPort)).toEqual({
      '/api': {
        target: 'http://127.0.0.1:7690',
        secure: false,
        changeOrigin: true,
      },
    });
  });

  it('applies environment overrides and rewrites only the configured API path', () => {
    const fileEnv = parseEnvText(
      "FASTAPI_HOST=127.0.0.2\nFASTAPI_PORT=7701\nUI_PORT=9850\nVITE_API_BASE_URL='/backend-api'",
    );
    const settings = resolveRuntimeNetworkSettings({ UI_PORT: '9851' }, fileEnv);

    expect(settings).toMatchObject({
      apiHost: '127.0.0.2',
      apiPort: 7701,
      uiPort: 9851,
      apiBaseUrl: '/backend-api',
    });
    expect(buildApiProxyConfig(settings.apiBaseUrl, settings.apiHost, settings.apiPort)[
      '/backend-api'
    ].pathRewrite?.['^/backend-api']).toBe('/api');
    expect(rewriteApiRequestPath('/backend-api/settings?mode=full', settings.apiBaseUrl)).toBe(
      '/api/settings?mode=full',
    );
    expect(rewriteApiRequestPath('/backend-apix/settings', settings.apiBaseUrl)).toBeNull();
  });

  it('bypasses the local proxy for absolute API URLs', () => {
    const apiBaseUrl = resolveApiBaseUrl('https://api.example.org/v1/');

    expect(apiBaseUrl).toBe('https://api.example.org/v1');
    expect(buildApiProxyConfig(apiBaseUrl, '127.0.0.1', 7690)).toEqual({});
    expect(rewriteApiRequestPath('/api/settings', apiBaseUrl)).toBeNull();
    expect(apiBaseUrlDefine(apiBaseUrl)).toBe(
      '__DILIGENT_API_BASE_URL__="https://api.example.org/v1"',
    );
  });

  it('pins desktop builds to the shell-managed /api route', () => {
    expect(resolveRuntimeNetworkSettings(
      { VITE_API_BASE_URL: 'https://api.example.org' },
      {},
      { desktopBuild: true },
    ).apiBaseUrl).toBe('/api');
  });

  it('rejects invalid ports and unsafe API URLs', () => {
    expect(() => resolveRuntimeNetworkSettings({ FASTAPI_PORT: '70000' }, {})).toThrow(
      /FASTAPI_PORT must be an integer/,
    );
    expect(() => resolveApiBaseUrl('//api.example.org/v1')).toThrow(/protocol-relative/);
    expect(() => resolveApiBaseUrl('ftp://api.example.org')).toThrow(/HTTP\(S\) URL/);
    expect(() => resolveApiBaseUrl('https://user:pass@api.example.org')).toThrow(/HTTP\(S\) URL/);
  });
});

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

describe('buildClinicalPayload', () => {
  const form: ClinicalFormState = {
    patientName: 'John',
    visitDate: '2025-01-02',
    patientImageDataUrl: null,
    clinicalInput: 'test input',
  };

  it('carries the agent id and revision and none of the removed fields', () => {
    const payload = buildClinicalPayload(form, 'agent-1', 3);
    expect(payload.agent_id).toBe('agent-1');
    expect(payload.agent_revision).toBe(3);
    for (const removed of ['selected_model_providers', 'model_overrides', 'evidence_sources', 'use_rag']) {
      expect(removed in payload).toBe(false);
    }
  });

  it('sends null agent fields to select the default agent', () => {
    const payload = buildClinicalPayload(form, null, null);
    expect(payload.agent_id).toBeNull();
    expect(payload.agent_revision).toBeNull();
  });
});

describe('isRecord', () => {
  it('narrows plain objects while excluding null and arrays', () => {
    expect(isRecord({ value: 1 })).toBe(true);
    expect(isRecord(null)).toBe(false);
    expect(isRecord(['value'])).toBe(false);
    expect(isRecord('value')).toBe(false);
  });
});
