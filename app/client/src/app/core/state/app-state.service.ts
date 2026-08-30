import { Injectable, effect, signal } from '@angular/core';

import { DEFAULT_FORM_STATE } from '../constants';
import { ClinicalFormState, JobStatus } from '../models/types';

export type PageId = 'dili-agent' | 'clinical-sessions' | 'data-inspection' | 'model-config';
export type ThemeMode = 'light' | 'dark';

const DEFAULT_PAGE: PageId = 'dili-agent';
const DILI_AGENT_UI_STATE_KEY = 'dili-agent-ui-state-v1';
const PAGE_PATHS: Record<PageId, string> = {
  'dili-agent': '/',
  'clinical-sessions': '/clinical-sessions',
  'data-inspection': '/data',
  'model-config': '/model-config',
};

export function normalizePathname(pathname: string): string {
  const trimmed = pathname.trim();
  if (!trimmed) return '/';
  if (trimmed.length > 1 && trimmed.endsWith('/')) {
    return trimmed.slice(0, -1);
  }
  return trimmed;
}

export function resolvePageIdFromPath(pathname: string): PageId {
  const normalized = normalizePathname(pathname);
  if (normalized === PAGE_PATHS['clinical-sessions'] || normalized.startsWith('/sessions/')) return 'clinical-sessions';
  if (normalized === PAGE_PATHS['data-inspection']) return 'data-inspection';
  if (normalized === PAGE_PATHS['model-config']) return 'model-config';
  return DEFAULT_PAGE;
}

export function resolvePathFromPage(page: PageId): string {
  return PAGE_PATHS[page] || PAGE_PATHS[DEFAULT_PAGE];
}

export interface DiliAgentState {
  form: ClinicalFormState;
  message: string;
  exportUrl: string | null;
  jobId: string | null;
  jobProgress: number;
  jobStatus: JobStatus | null;
  jobStage: string | null;
  jobStageMessage: string | null;
  isStarting: boolean;
  isRunning: boolean;
  isPulling: boolean;
  isExpanded: boolean;
  jobStartedAtMs: number | null;
  jobLastProgressAtMs: number | null;
  pollIntervalMs: number | null;
}

export interface AppState {
  activePage: PageId;
  theme: ThemeMode;
  diliAgent: DiliAgentState;
}

type PersistedDiliAgentUiState = {
  isExpanded: boolean;
};

const DEFAULT_DILI_AGENT_STATE: DiliAgentState = {
  form: DEFAULT_FORM_STATE,
  message: '',
  exportUrl: null,
  jobId: null,
  jobProgress: 0,
  jobStatus: null,
  jobStage: null,
  jobStageMessage: null,
  isStarting: false,
  isRunning: false,
  isPulling: false,
  isExpanded: false,
  jobStartedAtMs: null,
  jobLastProgressAtMs: null,
  pollIntervalMs: null,
};

function readPersistedDiliAgentUiState(): Pick<DiliAgentState, 'isExpanded'> | null {
  if (!('localStorage' in globalThis)) {
    return null;
  }

  try {
    const serializedState = globalThis.localStorage.getItem(DILI_AGENT_UI_STATE_KEY);
    if (!serializedState) {
      return null;
    }

    const parsedState: unknown = JSON.parse(serializedState);
    if (typeof parsedState !== 'object' || parsedState === null) {
      return null;
    }
    const isExpanded = (parsedState as Partial<PersistedDiliAgentUiState>).isExpanded;
    if (typeof isExpanded !== 'boolean') {
      return null;
    }
    return { isExpanded };
  } catch {
    return null;
  }
}

function writePersistedDiliAgentState(isExpanded: boolean): void {
  if (!('localStorage' in globalThis)) {
    return;
  }

  const persistedState: PersistedDiliAgentUiState = {
    isExpanded,
  };

  try {
    globalThis.localStorage.setItem(
      DILI_AGENT_UI_STATE_KEY,
      JSON.stringify(persistedState),
    );
  } catch {
    // keep runtime state only when browser storage is unavailable
  }
}

function createDefaultAppState(): AppState {
  return {
    activePage: resolvePageIdFromPath(globalThis.location?.pathname ?? '/'),
    theme: globalThis.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light',
    diliAgent: {
      ...DEFAULT_DILI_AGENT_STATE,
      ...readPersistedDiliAgentUiState(),
    },
  };
}

@Injectable({ providedIn: 'root' })
export class AppStateService {
  readonly state = signal<AppState>(createDefaultAppState());

  constructor() {
    effect(() => {
      const theme = this.state().theme;
      document.documentElement.dataset['theme'] = theme;
      document.documentElement.style.colorScheme = theme;
    });

    effect(() => {
      writePersistedDiliAgentState(this.state().diliAgent.isExpanded);
    });
  }

  setActivePage(page: PageId): void {
    this.state.update((prev) => ({ ...prev, activePage: page }));
  }

  setTheme(theme: ThemeMode): void {
    this.state.update((prev) => ({ ...prev, theme }));
  }

  toggleTheme(): void {
    this.state.update((prev) => ({ ...prev, theme: prev.theme === 'dark' ? 'light' : 'dark' }));
  }

  updateDiliAgent(updates: Partial<DiliAgentState>): void {
    this.state.update((prev) => ({ ...prev, diliAgent: { ...prev.diliAgent, ...updates } }));
  }
}

