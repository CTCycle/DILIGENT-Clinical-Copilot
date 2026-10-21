let desktopSessionToken = "";

export function setDesktopSessionToken(token: string): void {
  desktopSessionToken = token;
}

export function getDesktopSessionToken(): string {
  return desktopSessionToken;
}
