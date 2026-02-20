import React, { createContext, useContext, useState, useCallback, useMemo, useEffect } from "react";
import {
    CLOUD_MODEL_CHOICES,
    DEFAULT_FORM_STATE,
    DEFAULT_SETTINGS,
} from "../constants";
import { ClinicalFormState, RuntimeSettings } from "../types";
import { fetchModelConfigState } from "../services/api";

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
    message: string;
    jsonPayload: unknown;
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
const DEFAULT_DILU_AGENT_STATE: DiluAgentState = {
    settings: DEFAULT_SETTINGS,
    form: DEFAULT_FORM_STATE,
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
    activePage: resolvePageIdFromPath(globalThis.location?.pathname ?? "/"),
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
                const providerRaw = (payload.llm_provider || "").trim().toLowerCase();
                const cloudChoices = payload.cloud_model_choices || CLOUD_MODEL_CHOICES;
                const provider = cloudChoices[providerRaw] ? providerRaw : DEFAULT_SETTINGS.provider;
                const providerModels = cloudChoices[provider] || [];
                const cloudModel = (
                    payload.cloud_model && providerModels.includes(payload.cloud_model)
                        ? payload.cloud_model
                        : providerModels[0] ?? null
                );
                setState((prev) => {
                    const nextSettings: RuntimeSettings = {
                        ...prev.diluAgent.settings,
                        useCloudServices: payload.use_cloud_services,
                        provider,
                        cloudModel,
                        parsingModel: payload.text_extraction_model || prev.diluAgent.settings.parsingModel,
                        clinicalModel: payload.clinical_model || prev.diluAgent.settings.clinicalModel,
                        reasoning: payload.ollama_reasoning,
                    };
                    return {
                        ...prev,
                        diluAgent: {
                            ...prev.diluAgent,
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
