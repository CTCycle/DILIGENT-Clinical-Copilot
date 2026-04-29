import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { createServer, request } from 'node:http';
import path from 'node:path';

const rootDir = process.cwd();
const distDir = path.resolve(rootDir, 'dist/browser');
const fallbackDistDir = path.resolve(rootDir, 'dist');
const staticRoot = existsSync(distDir) ? distDir : fallbackDistDir;
const envPath = path.resolve(rootDir, '../settings/.env');

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
const args = process.argv.slice(2);

function resolveArg(flag, fallback) {
  const i = args.indexOf(flag);
  if (i >= 0 && args[i + 1]) return args[i + 1];
  return fallback;
}

const uiHost = resolveArg('--host', env.UI_HOST || defaults.UI_HOST);
const uiPort = Number.parseInt(resolveArg('--port', env.UI_PORT || defaults.UI_PORT), 10);
const apiHost = env.FASTAPI_HOST || defaults.FASTAPI_HOST;
const apiPort = Number.parseInt(env.FASTAPI_PORT || defaults.FASTAPI_PORT, 10);
const fallbackApiPort = defaults.FASTAPI_PORT;

function forwardApiRequest({ req, res, port, allowFallback }) {
  const upstream = request(
    {
      host: apiHost,
      port,
      path: req.url,
      method: req.method,
      headers: req.headers,
    },
    (upstreamRes) => {
      res.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
      upstreamRes.pipe(res);
    },
  );
  upstream.on('error', () => {
    if (allowFallback && String(port) !== String(fallbackApiPort)) {
      forwardApiRequest({
        req,
        res,
        port: fallbackApiPort,
        allowFallback: false,
      });
      return;
    }
    res.statusCode = 502;
    res.end('Bad Gateway');
  });
  req.pipe(upstream);
}

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

const server = createServer((req, res) => {
  if (!req.url) {
    res.statusCode = 400;
    res.end('Bad Request');
    return;
  }

  if (req.url.startsWith('/api')) {
    forwardApiRequest({
      req,
      res,
      port: apiPort,
      allowFallback: true,
    });
    return;
  }

  let reqPath = decodeURIComponent(req.url.split('?')[0]);
  if (reqPath === '/' || reqPath === '') {
    reqPath = '/index.html';
  }

  const filePath = path.resolve(staticRoot, `.${reqPath}`);
  const safeRoot = path.resolve(staticRoot);
  if (!filePath.startsWith(safeRoot)) {
    res.statusCode = 403;
    res.end('Forbidden');
    return;
  }

  const tryPaths = [filePath, path.resolve(staticRoot, 'index.html')];
  const chosen = tryPaths.find((p) => existsSync(p));
  if (!chosen) {
    res.statusCode = 404;
    res.end('Not Found');
    return;
  }

  const ext = path.extname(chosen).toLowerCase();
  res.setHeader('Content-Type', mime[ext] || 'application/octet-stream');
  createReadStream(chosen).pipe(res);
});

server.listen(uiPort, uiHost, () => {
  // eslint-disable-next-line no-console
  console.log(`Preview server running at http://${uiHost}:${uiPort}`);
});
