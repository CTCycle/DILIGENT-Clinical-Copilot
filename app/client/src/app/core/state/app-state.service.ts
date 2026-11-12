// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Injectable, effect, signal } from '@angular/core';

import { DEFAULT_FORM_STATE } from '../constants';
import { ClinicalFormState } from '../models/types';

export type PageId = 'dili-agent' | 'clinical-sessions' | 'agent-studio' | 'data-inspection' | 'settings';
export type ThemeMode = 'light' | 'dark';

const DEFAULT_PAGE: PageId = 'dili-agent';
const PAGE_PATHS: Record<PageId, string> = {
  'dili-agent': '/',
  'clinical-sessions': '/clinical-sessions',
  'agent-studio': '/agents',
  'data-inspection': '/data',
  settings: '/settings/general',
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
  const normalized = normalizePathname(pathname.split(/[?#]/, 1)[0]);
  if (normalized === PAGE_PATHS['clinical-sessions'] || normalized.startsWith('/sessions/')) return 'clinical-sessions';
  if (normalized === PAGE_PATHS['agent-studio'] || normalized.startsWith('/agents/')) return 'agent-studio';
  if (normalized === PAGE_PATHS['data-inspection']) return 'data-inspection';
  if (normalized === '/settings' || normalized.startsWith('/settings/')) return 'settings';
  return DEFAULT_PAGE;
}

export function resolvePathFromPage(page: PageId): string {
  return PAGE_PATHS[page] || PAGE_PATHS[DEFAULT_PAGE];
}

export type DraftSaveStatus = 'idle' | 'saving' | 'saved' | 'error' | 'conflict';

export interface DiliAgentState {
  form: ClinicalFormState;
  draftSessionId: number | null;
  draftUpdatedAt: string | null;
  saveStatus: DraftSaveStatus;
  lastSavedAt: string | null;
  isDirty: boolean;
  isPulling: boolean;
}

export interface AppState {
  activePage: PageId;
  theme: ThemeMode;
  diliAgent: DiliAgentState;
}

export const DEFAULT_DILI_AGENT_STATE: DiliAgentState = {
  form: DEFAULT_FORM_STATE,
  draftSessionId: null,
  draftUpdatedAt: null,
  saveStatus: 'idle',
  lastSavedAt: null,
  isDirty: false,
  isPulling: false,
};

const THEME_STORAGE_KEY = 'diligent.theme';

function readStoredTheme(): ThemeMode | null {
  try {
    const value = globalThis.localStorage?.getItem(THEME_STORAGE_KEY);
    return value === 'light' || value === 'dark' ? value : null;
  } catch {
    return null;
  }
}

function storeTheme(theme: ThemeMode): void {
  try {
    globalThis.localStorage?.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Storage may be unavailable; the choice then lasts for this session only.
  }
}

function createDefaultAppState(): AppState {
  return {
    activePage: resolvePageIdFromPath(globalThis.location?.pathname ?? '/'),
    theme:
      readStoredTheme() ??
      (globalThis.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light'),
    diliAgent: { ...DEFAULT_DILI_AGENT_STATE },
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
  }

  setActivePage(page: PageId): void {
    this.state.update((prev) => ({ ...prev, activePage: page }));
  }

  setTheme(theme: ThemeMode): void {
    this.state.update((prev) => ({ ...prev, theme }));
    storeTheme(theme);
  }

  toggleTheme(): void {
    this.setTheme(this.state().theme === 'dark' ? 'light' : 'dark');
  }

  updateDiliAgent(updates: Partial<DiliAgentState>): void {
    this.state.update((prev) => ({ ...prev, diliAgent: { ...prev.diliAgent, ...updates } }));
  }
}
