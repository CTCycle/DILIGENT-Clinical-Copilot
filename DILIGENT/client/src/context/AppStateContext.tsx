import React, { createContext, useContext, useState, useCallback, useMemo, useEffect } from "react";
import {
    DEFAULT_FORM_STATE,
    DEFAULT_SETTINGS,
} from "../constants";
import { buildRuntimeSettingsFromConfig } from "../modelConfig";
import { ClinicalFormState, JobStatus, RuntimeSettings } from "../types";
import { fetchModelConfigState } from "../services/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type PageId = "dili-agent" | "data-inspection" | "model-config";

const DEFAULT_PAGE: PageId = "dili-agent";
const PAGE_PATHS: Record<PageId, string> = {
    "dili-agent": "/",
    "data-inspection": "/data",
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
    if (normalized === PAGE_PATHS["data-inspection"]) {
        return "data-inspection";
    }
    if (normalized === PAGE_PATHS["model-config"]) {
        return "model-config";
    }
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
    isRunning: boolean;
    isPulling: boolean;
    isExpanded: boolean;
}

export interface AppState {
    activePage: PageId;
    diliAgent: DiliAgentState;
}

interface AppStateContextValue {
    state: AppState;
    setActivePage: (page: PageId) => void;
    updateDiliAgent: (updates: Partial<DiliAgentState>) => void;
}

// ---------------------------------------------------------------------------
// Default State
// ---------------------------------------------------------------------------
const DEFAULT_DILI_AGENT_STATE: DiliAgentState = {
    settings: DEFAULT_SETTINGS,
    form: DEFAULT_FORM_STATE,
    message: "",
    exportUrl: null,
    jobId: null,
    jobProgress: 0,
    jobStatus: null,
    jobStage: null,
    jobStageMessage: null,
    isRunning: false,
    isPulling: false,
    isExpanded: false,
};

const DEFAULT_APP_STATE: AppState = {
    activePage: resolvePageIdFromPath(globalThis.location?.pathname ?? "/"),
    diliAgent: DEFAULT_DILI_AGENT_STATE,
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
    readonly children: React.ReactNode;
}

export function AppStateProvider({ children }: AppStateProviderProps): React.JSX.Element {
    const [state, setState] = useState<AppState>(DEFAULT_APP_STATE);

    useEffect(() => {
        let cancelled = false;
        const hydrateSettings = async () => {
            try {
                const payload = await fetchModelConfigState();
                if (cancelled) {
                    return;
                }
                setState((prev) => {
                    const nextSettings: RuntimeSettings = buildRuntimeSettingsFromConfig(
                        payload,
                        prev.diliAgent.settings,
                    );
                    return {
                        ...prev,
                        diliAgent: {
                            ...prev.diliAgent,
                            settings: nextSettings,
                        },
                    };
                });
            } catch {
                // Keep defaults when backend config is not reachable.
            }
        };
        void hydrateSettings();
        return () => {
            cancelled = true;
        };
    }, []);

    const setActivePage = useCallback((page: PageId) => {
        setState((prev) => ({ ...prev, activePage: page }));
    }, []);

    const updateDiliAgent = useCallback((updates: Partial<DiliAgentState>) => {
        setState((prev) => ({
            ...prev,
            diliAgent: { ...prev.diliAgent, ...updates },
        }));
    }, []);

    const value = useMemo<AppStateContextValue>(
        () => ({
            state,
            setActivePage,
            updateDiliAgent,
        }),
        [state, setActivePage, updateDiliAgent],
    );

    return (
        <AppStateContext.Provider value={value}>
            {children}
        </AppStateContext.Provider>
    );
}
