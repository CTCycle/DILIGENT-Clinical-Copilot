// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { spawn } from 'node:child_process';
import path from 'node:path';
import {
  apiBaseUrlDefine,
  loadEnvFile,
  resolveRuntimeNetworkSettings,
} from './runtime-config.mjs';

const projectRoot = process.cwd();
const envPath = path.resolve(projectRoot, '../../settings/.env');
const env = loadEnvFile(envPath);
const desktopBuild = ['1', 'true', 'yes', 'on'].includes(
  String(process.env.DILIGENT_DESKTOP_BUILD || '').trim().toLowerCase(),
);
const { apiBaseUrl } = resolveRuntimeNetworkSettings(process.env, env, { desktopBuild });
const angularCli = path.resolve(projectRoot, 'node_modules/@angular/cli/bin/ng.js');
const child = spawn(
  process.execPath,
  [
    angularCli,
    'build',
    ...process.argv.slice(2),
    '--define',
    apiBaseUrlDefine(apiBaseUrl),
  ],
  { stdio: 'inherit' },
);

child.on('error', (error) => {
  // eslint-disable-next-line no-console
  console.error(error);
  process.exitCode = 1;
});

child.on('exit', (code) => {
  process.exitCode = code ?? 1;
});
