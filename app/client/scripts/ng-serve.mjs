// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import {
  apiBaseUrlDefine,
  buildApiProxyConfig,
  loadEnvFile,
  resolveRuntimeNetworkSettings,
} from './runtime-config.mjs';

const projectRoot = process.cwd();
const envPath = path.resolve(projectRoot, '../../settings/.env');
const env = loadEnvFile(envPath);
const { apiHost, apiPort, apiBaseUrl, uiHost, uiPort } =
  resolveRuntimeNetworkSettings(process.env, env);

const proxyConfigPath = path.resolve(projectRoot, 'proxy.conf.json');
const proxyConfigContent = `${JSON.stringify(buildApiProxyConfig(apiBaseUrl, apiHost, apiPort), null, 2)}\n`;

let existingProxyConfig = '';
try {
  existingProxyConfig = readFileSync(proxyConfigPath, 'utf8');
} catch {
  existingProxyConfig = '';
}

if (existingProxyConfig !== proxyConfigContent) {
  try {
    writeFileSync(proxyConfigPath, proxyConfigContent, 'utf8');
  } catch (error) {
    // Under some Windows launch paths proxy.conf.json can be transiently locked.
    // Continue when the existing file already points to the desired backend target.
    const latestProxyConfig = readFileSync(proxyConfigPath, 'utf8');
    const expectedTargetSnippet = `"target": "http://${apiHost}:${apiPort}"`;
    if (!latestProxyConfig.includes(expectedTargetSnippet)) {
      throw error;
    }
  }
}

const angularCli = path.resolve(projectRoot, 'node_modules/@angular/cli/bin/ng.js');
const child = spawn(
  process.execPath,
  [
    angularCli,
    'serve',
    '--host',
    uiHost,
    '--port',
    String(uiPort),
    '--define',
    apiBaseUrlDefine(apiBaseUrl),
    ...(apiBaseUrl.startsWith('/') ? ['--proxy-config', 'proxy.conf.json'] : []),
  ],
  { stdio: 'inherit' },
);

child.on('exit', (code) => {
  process.exit(code ?? 1);
});

