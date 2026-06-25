import { Injectable, effect, signal } from '@angular/core';

import { DEFAULT_FORM_STATE, DEFAULT_SETTINGS } from '../constants';
import { buildRuntimeSettingsFromConfig } from '../model-config';
import { fetchModelConfigState } from '../services/model-config-api';
import { ClinicalFormState, JobStatus, RuntimeSettings } from '../models/types';

export type PageId = 'dili-agent' | 'clinical-sessions' | 'data-inspection' | 'model-config';
export type ThemeMode = 'light' | 'dark';

const DEFAULT_PAGE: PageId = 'dili-agent';
const DILI_AGENT_PERSISTED_STATE_KEY = 'dili-agent-state-v2';
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
  if (normalized === PAGE_PATHS['clinical-sessions']) return 'clinical-sessions';
  if (normalized === PAGE_PATHS['data-inspection']) return 'data-inspection';
  if (normalized === PAGE_PATHS['model-config']) return 'model-config';
  return DEFAULT_PAGE;
}

export function resolvePathFromPage(page: PageId): string {
  return PAGE_PATHS[page] || PAGE_PATHS[DEFAULT_PAGE];
}

export interface DiliAgentState {
  settings: RuntimeSettings;
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

type PersistedDiliAgentState = {
  settings: RuntimeSettings;
  form: ClinicalFormState;
  message: string;
  jobId: string | null;
  jobProgress: number;
  jobStatus: JobStatus | null;
  jobStage: string | null;
  jobStageMessage: string | null;
  isStarting: boolean;
  isRunning: boolean;
  isExpanded: boolean;
  jobStartedAtMs: number | null;
  jobLastProgressAtMs: number | null;
  pollIntervalMs: number | null;
};

const DEFAULT_DILI_AGENT_STATE: DiliAgentState = {
  settings: DEFAULT_SETTINGS,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isJobStatus(value: unknown): value is JobStatus | null {
  return value === null
    || value === 'pending'
    || value === 'running'
    || value === 'completed'
    || value === 'failed'
    || value === 'cancelled';
}

function readFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function readOptionalString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function readPersistedDiliAgentState(): Partial<DiliAgentState> | null {
  if (!('sessionStorage' in globalThis)) {
    return null;
  }

  try {
    const serializedState = globalThis.sessionStorage.getItem(DILI_AGENT_PERSISTED_STATE_KEY);
    if (!serializedState) {
      return null;
    }

    const parsedState: unknown = JSON.parse(serializedState);
    if (!isRecord(parsedState)) {
      return null;
    }

    const persistedState = parsedState as Partial<PersistedDiliAgentState>;
    const settings = persistedState.settings;
    const form = persistedState.form;
    const message = persistedState.message;
    const jobId = persistedState.jobId;
    const jobProgress = persistedState.jobProgress;
    const jobStatus = persistedState.jobStatus;
    const jobStage = persistedState.jobStage;
    const jobStageMessage = persistedState.jobStageMessage;
    const isStarting = persistedState.isStarting;
    const isRunning = persistedState.isRunning;
    const isExpanded = persistedState.isExpanded;
    const jobStartedAtMs = persistedState.jobStartedAtMs;
    const jobLastProgressAtMs = persistedState.jobLastProgressAtMs;
    const pollIntervalMs = persistedState.pollIntervalMs;

    if (
      !settings ||
      !form ||
      typeof message !== 'string' ||
      !isJobStatus(jobStatus) ||
      (jobId !== null && typeof jobId !== 'string') ||
      typeof isExpanded !== 'boolean' ||
      typeof isStarting !== 'boolean' ||
      typeof isRunning !== 'boolean'
    ) {
      return null;
    }

    const normalizedJobProgress = readFiniteNumber(jobProgress) ?? (jobStatus === 'completed' ? 100 : 0);

    return {
      settings: {
        ...DEFAULT_SETTINGS,
        ...settings,
      },
      form: {
        ...DEFAULT_FORM_STATE,
        ...form,
      },
      message,
      jobId,
      jobProgress: normalizedJobProgress,
      jobStatus,
      jobStage: readOptionalString(jobStage),
      jobStageMessage: readOptionalString(jobStageMessage),
      isStarting,
      isRunning,
      isExpanded,
      jobStartedAtMs: readFiniteNumber(jobStartedAtMs),
      jobLastProgressAtMs: readFiniteNumber(jobLastProgressAtMs),
      pollIntervalMs: readFiniteNumber(pollIntervalMs),
      exportUrl: null,
    };
  } catch {
    return null;
  }
}

function writePersistedDiliAgentState(state: DiliAgentState): void {
  if (!('sessionStorage' in globalThis)) {
    return;
  }

  const persistedState: PersistedDiliAgentState = {
    settings: state.settings,
    form: state.form,
    message: state.message,
    jobId: state.jobId,
    jobProgress: state.jobProgress,
    jobStatus: state.jobStatus,
    jobStage: state.jobStage,
    jobStageMessage: state.jobStageMessage,
    isStarting: state.isStarting,
    isRunning: state.isRunning,
    isExpanded: state.isExpanded,
    jobStartedAtMs: state.jobStartedAtMs,
    jobLastProgressAtMs: state.jobLastProgressAtMs,
    pollIntervalMs: state.pollIntervalMs,
  };

  try {
    globalThis.sessionStorage.setItem(
      DILI_AGENT_PERSISTED_STATE_KEY,
      JSON.stringify(persistedState),
    );
  } catch {
    // keep runtime state only when browser storage is unavailable
  }
}

const DEFAULT_APP_STATE: AppState = {
  activePage: resolvePageIdFromPath(globalThis.location?.pathname ?? '/'),
  theme: globalThis.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light',
  diliAgent: {
    ...DEFAULT_DILI_AGENT_STATE,
    ...readPersistedDiliAgentState(),
  },
};

@Injectable({ providedIn: 'root' })
export class AppStateService {
  readonly state = signal<AppState>(DEFAULT_APP_STATE);

  constructor() {
    effect(() => {
      const theme = this.state().theme;
      document.documentElement.dataset['theme'] = theme;
      document.documentElement.style.colorScheme = theme;
    });

    effect(() => {
      writePersistedDiliAgentState(this.state().diliAgent);
    });

    if (this.state().activePage !== 'model-config') {
      void this.hydrateSettings();
    }
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

  private async hydrateSettings(): Promise<void> {
    try {
      const payload = await fetchModelConfigState();
      this.state.update((prev) => ({
        ...prev,
        diliAgent: {
          ...prev.diliAgent,
          settings: buildRuntimeSettingsFromConfig(payload, prev.diliAgent.settings),
        },
      }));
    } catch {
      // keep defaults
    }
  }
}

