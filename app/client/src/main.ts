import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { App } from './app/app';

const DESKTOP_BOOTSTRAP_PREFIX = '#desktop-bootstrap=';

function showStartupError(): void {
  document.body.textContent =
    'DILIGENT could not start its local interface. Please restart the application.';
}

async function bootstrapDesktopSession(): Promise<void> {
  if (!window.location.hash.startsWith(DESKTOP_BOOTSTRAP_PREFIX)) {
    return;
  }

  const bootstrapHash = window.location.hash;
  window.history.replaceState(
    null,
    document.title,
    window.location.pathname + window.location.search,
  );
  const token = decodeURIComponent(bootstrapHash.slice(DESKTOP_BOOTSTRAP_PREFIX.length));
  if (!token) {
    throw new Error('Missing desktop bootstrap token');
  }

  const response = await fetch('/api/desktop/bootstrap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({ token }),
  });
  if (!response.ok) {
    throw new Error('Desktop session bootstrap failed');
  }
}

async function startApplication(): Promise<void> {
  try {
    await bootstrapDesktopSession();
    await bootstrapApplication(App, appConfig);
  } catch {
    showStartupError();
  }
}

void startApplication();
