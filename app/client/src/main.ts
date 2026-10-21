// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { App } from './app/app';
import { API_BASE_URL, configureDesktopApiOrigin } from './app/core/constants';
import { setDesktopSessionToken } from './app/core/services/desktop-session';

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
  const token = decodeURIComponent(bootstrapHash.slice(DESKTOP_BOOTSTRAP_PREFIX.length));
  if (!token) {
    throw new Error('Missing desktop bootstrap token');
  }
  const apiOrigin = new URLSearchParams(window.location.search).get('apiOrigin');
  if (!apiOrigin) {
    throw new Error('Missing desktop API origin');
  }
  configureDesktopApiOrigin(apiOrigin);
  window.history.replaceState(null, document.title, window.location.pathname);

  const response = await fetch(`${API_BASE_URL}/desktop/bootstrap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    mode: 'cors',
    credentials: 'omit',
    body: JSON.stringify({ token }),
  });
  if (!response.ok) {
    throw new Error('Desktop session bootstrap failed');
  }
  setDesktopSessionToken(token);
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
