let desktopSessionToken = "";

export function setDesktopSessionToken(token: string): void {
  desktopSessionToken = token;
}

export function getDesktopSessionToken(): string {
  return desktopSessionToken;
}

const DESKTOP_SESSION_STORAGE_KEY = "diligent.desktop-session";

export interface StoredDesktopSession {
  apiOrigin: string;
  token: string;
}

// A full page reload drops the one-time bootstrap URL, so the window keeps the
// loopback API origin and session token for the lifetime of the webview.
export function storeDesktopSession(session: StoredDesktopSession): void {
  window.sessionStorage.setItem(DESKTOP_SESSION_STORAGE_KEY, JSON.stringify(session));
}

export function loadStoredDesktopSession(): StoredDesktopSession | null {
  try {
    const parsed: unknown = JSON.parse(
      window.sessionStorage.getItem(DESKTOP_SESSION_STORAGE_KEY) ?? "null",
    );
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      typeof (parsed as StoredDesktopSession).apiOrigin === "string" &&
      typeof (parsed as StoredDesktopSession).token === "string" &&
      (parsed as StoredDesktopSession).token !== ""
    ) {
      const { apiOrigin, token } = parsed as StoredDesktopSession;
      return { apiOrigin, token };
    }
  } catch {
    return null;
  }
  return null;
}

