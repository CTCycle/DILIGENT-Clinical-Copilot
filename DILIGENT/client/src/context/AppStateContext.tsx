import React, { createContext, useContext, useState, useCallback, useMemo } from "react";
import {
    DEFAULT_FORM_STATE,
    DEFAULT_SETTINGS,
} from "../constants";
import { ClinicalFormState, RuntimeSettings } from "../types";
import { resolveCloudSelection } from "../utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type PageId = "dili-agent" | "model-config";

const DEFAULT_PAGE: PageId = "dili-agent";
const PAGE_PATHS: Record<PageId, string> = {
    "dili-agent": "/",
    "model-config": "/model-config",
};

function normalizePathname(pathname: string): string {
    const trimmed = pathname.trim();
    if (!trimmed) return "/";
    if (trimmed.length > 1 && trimmed.endsWith("/")) {
        return trimmed.slice(0, -1);
    }
    return trimmed;
}

export function resolvePageIdFromPath(pathname: string): PageId {
    const normalized = normalizePathname(pathname);
    if (normalized === PAGE_PATHS["model-config"]) {
        return "model-config";
    }
    return DEFAULT_PAGE;
}

export function resolvePathFromPage(page: PageId): string {
    return PAGE_PATHS[page] || PAGE_PATHS[DEFAULT_PAGE];
}

export interface DiluAgentState {
    settings: RuntimeSettings;
    form: ClinicalFormState;
    cloudSelection: { provider: string; model: string | null };
    message: string;
    jsonPayload: unknown | null;
    exportUrl: string | null;
    jobId: string | null;
    jobProgress: number;
    jobStatus: string | null;
    isRunning: boolean;
    isPulling: boolean;
    isExpanded: boolean;
}

export interface AppState {
    activePage: PageId;
    diluAgent: DiluAgentState;
}

interface AppStateContextValue {
    state: AppState;
    setActivePage: (page: PageId) => void;
    updateDiluAgent: (updates: Partial<DiluAgentState>) => void;
}

// ---------------------------------------------------------------------------
// Default State
// ---------------------------------------------------------------------------
const defaultCloudSelection = resolveCloudSelection(
    DEFAULT_SETTINGS.provider,
    DEFAULT_SETTINGS.cloudModel,
);

const DEFAULT_DILU_AGENT_STATE: DiluAgentState = {
    settings: DEFAULT_SETTINGS,
    form: DEFAULT_FORM_STATE,
    cloudSelection: {
        provider: defaultCloudSelection.provider,
        model: defaultCloudSelection.model,
    },
    message: "",
    jsonPayload: null,
    exportUrl: null,
    jobId: null,
    jobProgress: 0,
    jobStatus: null,
    isRunning: false,
    isPulling: false,
    isExpanded: false,
};

const DEFAULT_APP_STATE: AppState = {
    activePage: resolvePageIdFromPath(
        typeof window !== "undefined" ? window.location.pathname : "/",
    ),
    diluAgent: DEFAULT_DILU_AGENT_STATE,
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
const AppStateContext = createContext<AppStateContextValue | null>(null);

export function useAppState(): AppStateContextValue {
    const context = useContext(AppStateContext);
    if (!context) {
        throw new Error("useAppState must be used within AppStateProvider");
    }
    return context;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------
interface AppStateProviderProps {
    children: React.ReactNode;
}

export function AppStateProvider({ children }: AppStateProviderProps): React.JSX.Element {
    const [state, setState] = useState<AppState>(DEFAULT_APP_STATE);

    const setActivePage = useCallback((page: PageId) => {
        setState((prev) => ({ ...prev, activePage: page }));
    }, []);

    const updateDiluAgent = useCallback((updates: Partial<DiluAgentState>) => {
        setState((prev) => ({
            ...prev,
            diluAgent: { ...prev.diluAgent, ...updates },
        }));
    }, []);

    const value = useMemo<AppStateContextValue>(
        () => ({
            state,
            setActivePage,
            updateDiluAgent,
        }),
        [state, setActivePage, updateDiluAgent],
    );

    return (
        <AppStateContext.Provider value={value}>
            {children}
        </AppStateContext.Provider>
    );
}
