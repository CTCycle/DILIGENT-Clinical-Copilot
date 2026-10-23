declare module '*.mjs' {
  interface RuntimeNetworkSettings {
    apiHost: string;
    apiPort: number;
    uiHost: string;
    uiPort: number;
    apiBaseUrl: string;
  }

  interface ApiProxyConfig {
    target: string;
    secure: boolean;
    changeOrigin: boolean;
    pathRewrite?: Record<string, string>;
  }

  export const DEFAULT_RUNTIME_ENV: Readonly<Record<string, string>>;
  export function apiBaseUrlDefine(apiBaseUrl: string): string;
  export function buildApiProxyConfig(
    apiBaseUrl: string,
    apiHost: string,
    apiPort: number,
  ): Record<string, ApiProxyConfig>;
  export function parseEnvText(
    content: string,
    defaults?: Readonly<Record<string, string>>,
  ): Record<string, string>;
  export function resolveApiBaseUrl(value: string | null | undefined): string;
  export function resolveRuntimeNetworkSettings(
    processEnv: Record<string, string>,
    fileEnv: Record<string, string>,
    options?: { desktopBuild?: boolean },
  ): RuntimeNetworkSettings;
  export function rewriteApiRequestPath(requestUrl: string, apiBaseUrl: string): string | null;
}
