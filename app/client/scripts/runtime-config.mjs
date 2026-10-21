// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { readFileSync } from 'node:fs';

export const DEFAULT_RUNTIME_ENV = Object.freeze({
  FASTAPI_HOST: '127.0.0.1',
  FASTAPI_PORT: '7690',
  UI_HOST: '127.0.0.1',
  UI_PORT: '9847',
  VITE_API_BASE_URL: '/api',
});

export function parseEnvText(content, defaults = DEFAULT_RUNTIME_ENV) {
  const env = { ...defaults };
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#') || line.startsWith(';')) continue;
    const separator = line.indexOf('=');
    if (separator < 1) continue;
    const key = line.slice(0, separator).trim();
    let value = line.slice(separator + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (key) env[key] = value;
  }
  return env;
}

export function loadEnvFile(filePath, defaults = DEFAULT_RUNTIME_ENV) {
  try {
    return parseEnvText(readFileSync(filePath, 'utf8'), defaults);
  } catch {
    return { ...defaults };
  }
}

export function resolveEnvValue(name, processEnv, fileEnv, fallback) {
  const processValue = processEnv[name];
  if (typeof processValue === 'string' && processValue.trim()) {
    return processValue.trim();
  }
  const fileValue = fileEnv[name];
  if (typeof fileValue === 'string' && fileValue.trim()) {
    return fileValue.trim();
  }
  return fallback;
}

export function parsePort(value, name) {
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${name} must be an integer between 1 and 65535.`);
  }
  return port;
}

export function resolveApiBaseUrl(value) {
  const candidate = String(value ?? '').trim() || '/api';
  if (candidate.includes('\\')) {
    throw new Error('VITE_API_BASE_URL must use forward slashes.');
  }

  if (candidate.startsWith('/')) {
    if (candidate.startsWith('//')) {
      throw new Error('VITE_API_BASE_URL cannot be a protocol-relative URL.');
    }
    const parsed = new URL(candidate, 'http://diligent.local');
    if (
      parsed.origin !== 'http://diligent.local' ||
      parsed.search ||
      parsed.hash ||
      parsed.pathname === '/'
    ) {
      throw new Error('VITE_API_BASE_URL must be an API path or an HTTP(S) URL.');
    }
    return parsed.pathname.replace(/\/+$/, '');
  }

  let parsed;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error('VITE_API_BASE_URL must be an API path or an HTTP(S) URL.');
  }
  if (
    !['http:', 'https:'].includes(parsed.protocol) ||
    !parsed.hostname ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error('VITE_API_BASE_URL must be an API path or an HTTP(S) URL.');
  }
  return `${parsed.origin}${parsed.pathname.replace(/\/+$/, '')}`;
}

export function resolveRuntimeNetworkSettings(processEnv, fileEnv, { desktopBuild = false } = {}) {
  return {
    apiHost: resolveEnvValue('FASTAPI_HOST', processEnv, fileEnv, DEFAULT_RUNTIME_ENV.FASTAPI_HOST),
    apiPort: parsePort(
      resolveEnvValue('FASTAPI_PORT', processEnv, fileEnv, DEFAULT_RUNTIME_ENV.FASTAPI_PORT),
      'FASTAPI_PORT',
    ),
    uiHost: resolveEnvValue('UI_HOST', processEnv, fileEnv, DEFAULT_RUNTIME_ENV.UI_HOST),
    uiPort: parsePort(
      resolveEnvValue('UI_PORT', processEnv, fileEnv, DEFAULT_RUNTIME_ENV.UI_PORT),
      'UI_PORT',
    ),
    apiBaseUrl: desktopBuild
      ? '/api'
      : resolveApiBaseUrl(
          resolveEnvValue(
            'VITE_API_BASE_URL',
            processEnv,
            fileEnv,
            DEFAULT_RUNTIME_ENV.VITE_API_BASE_URL,
          ),
        ),
  };
}

export function buildApiProxyConfig(apiBaseUrl, apiHost, apiPort) {
  if (!apiBaseUrl.startsWith('/')) return {};
  const proxy = {
    target: `http://${apiHost}:${apiPort}`,
    secure: false,
    changeOrigin: true,
  };
  if (apiBaseUrl !== '/api') {
    proxy.pathRewrite = { [`^${apiBaseUrl}`]: '/api' };
  }
  return { [apiBaseUrl]: proxy };
}

export function rewriteApiRequestPath(requestUrl, apiBaseUrl) {
  if (!apiBaseUrl.startsWith('/')) return null;
  const parsed = new URL(requestUrl, 'http://diligent.local');
  if (
    parsed.pathname !== apiBaseUrl &&
    !parsed.pathname.startsWith(`${apiBaseUrl}/`)
  ) {
    return null;
  }
  const backendPath = `/api${parsed.pathname.slice(apiBaseUrl.length)}`;
  return `${backendPath}${parsed.search}`;
}

export function apiBaseUrlDefine(apiBaseUrl) {
  return `__DILIGENT_API_BASE_URL__=${JSON.stringify(apiBaseUrl)}`;
}
