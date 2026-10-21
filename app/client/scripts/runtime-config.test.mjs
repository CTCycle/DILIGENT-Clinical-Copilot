// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  apiBaseUrlDefine,
  buildApiProxyConfig,
  parseEnvText,
  resolveApiBaseUrl,
  resolveRuntimeNetworkSettings,
  rewriteApiRequestPath,
} from './runtime-config.mjs';

test('runtime network defaults retain same-origin API routing', () => {
  const settings = resolveRuntimeNetworkSettings({}, {});

  assert.deepEqual(settings, {
    apiHost: '127.0.0.1',
    apiPort: 7690,
    uiHost: '127.0.0.1',
    uiPort: 9847,
    apiBaseUrl: '/api',
  });
  assert.deepEqual(buildApiProxyConfig(settings.apiBaseUrl, settings.apiHost, settings.apiPort), {
    '/api': {
      target: 'http://127.0.0.1:7690',
      secure: false,
      changeOrigin: true,
    },
  });
});

test('runtime environment overrides ports and API base URL', () => {
  const fileEnv = parseEnvText(
    "FASTAPI_HOST=127.0.0.2\nFASTAPI_PORT=7701\nUI_PORT=9850\nVITE_API_BASE_URL='/backend-api'",
  );
  const settings = resolveRuntimeNetworkSettings({ UI_PORT: '9851' }, fileEnv);

  assert.equal(settings.apiHost, '127.0.0.2');
  assert.equal(settings.apiPort, 7701);
  assert.equal(settings.uiPort, 9851);
  assert.equal(settings.apiBaseUrl, '/backend-api');
  assert.equal(
    buildApiProxyConfig(settings.apiBaseUrl, settings.apiHost, settings.apiPort)[
      '/backend-api'
    ].pathRewrite['^/backend-api'],
    '/api',
  );
  assert.equal(
    rewriteApiRequestPath('/backend-api/settings?mode=full', settings.apiBaseUrl),
    '/api/settings?mode=full',
  );
  assert.equal(rewriteApiRequestPath('/backend-apix/settings', settings.apiBaseUrl), null);
});

test('absolute API URLs bypass the local development proxy', () => {
  const apiBaseUrl = resolveApiBaseUrl('https://api.example.org/v1/');

  assert.equal(apiBaseUrl, 'https://api.example.org/v1');
  assert.deepEqual(buildApiProxyConfig(apiBaseUrl, '127.0.0.1', 7690), {});
  assert.equal(rewriteApiRequestPath('/api/settings', apiBaseUrl), null);
  assert.equal(
    apiBaseUrlDefine(apiBaseUrl),
    '__DILIGENT_API_BASE_URL__="https://api.example.org/v1"',
  );
});

test('packaged desktop builds pin the API base URL to the shell-managed route', () => {
  const settings = resolveRuntimeNetworkSettings(
    { VITE_API_BASE_URL: 'https://api.example.org' },
    {},
    { desktopBuild: true },
  );

  assert.equal(settings.apiBaseUrl, '/api');
});

test('network settings reject invalid ports and unsafe API URLs', () => {
  assert.throws(
    () => resolveRuntimeNetworkSettings({ FASTAPI_PORT: '70000' }, {}),
    /FASTAPI_PORT must be an integer/,
  );
  assert.throws(() => resolveApiBaseUrl('//api.example.org/v1'), /protocol-relative/);
  assert.throws(() => resolveApiBaseUrl('ftp://api.example.org'), /HTTP\(S\) URL/);
  assert.throws(() => resolveApiBaseUrl('https://user:pass@api.example.org'), /HTTP\(S\) URL/);
});
