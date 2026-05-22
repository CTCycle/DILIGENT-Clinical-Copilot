import { readFileSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';

const projectRoot = process.cwd();
const envPath = path.resolve(projectRoot, '../../settings/.env');

const defaults = {
  FASTAPI_HOST: '127.0.0.1',
  FASTAPI_PORT: '8000',
  UI_HOST: '127.0.0.1',
  UI_PORT: '7861',
};

function loadEnvFile(filePath) {
  const env = { ...defaults };
  try {
    const content = readFileSync(filePath, 'utf8');
    for (const rawLine of content.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#') || line.startsWith(';')) continue;
      const idx = line.indexOf('=');
      if (idx < 0) continue;
      const key = line.slice(0, idx).trim();
      let value = line.slice(idx + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      if (key) env[key] = value;
    }
  } catch {
    // keep defaults
  }
  return env;
}

const env = loadEnvFile(envPath);
const uiHost = env.UI_HOST || defaults.UI_HOST;
const uiPort = env.UI_PORT || defaults.UI_PORT;
const apiHost = env.FASTAPI_HOST || defaults.FASTAPI_HOST;
const apiPort = env.FASTAPI_PORT || defaults.FASTAPI_PORT;

const proxyConfigPath = path.resolve(projectRoot, 'proxy.conf.json');
writeFileSync(
  proxyConfigPath,
  JSON.stringify(
    {
      '/api': {
        target: `http://${apiHost}:${apiPort}`,
        secure: false,
        changeOrigin: true,
      },
    },
    null,
    2,
  ),
  'utf8',
);

const isWindows = process.platform === 'win32';
const child = spawn(
  'npx',
  ['ng', 'serve', '--host', uiHost, '--port', uiPort, '--proxy-config', 'proxy.conf.json'],
  {
    stdio: 'inherit',
    shell: isWindows,
  },
);

child.on('exit', (code) => {
  process.exit(code ?? 1);
});

