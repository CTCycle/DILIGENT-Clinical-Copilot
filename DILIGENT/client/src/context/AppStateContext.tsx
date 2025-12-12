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
export type PageId = "dili-agent" | "database-browser";

export type TableId = "sessions" | "livertox" | "drugs";

export interface TableData {
    columns: string[];
    rows: Record<string, unknown>[];
    totalRows: number;
}

export interface TableCache {
    sessions: TableData | null;
    livertox: TableData | null;
    drugs: TableData | null;
}

export interface DiluAgentState {
    settings: RuntimeSettings;
    form: ClinicalFormState;
    cloudSelection: { provider: string; model: string | null };
    message: string;
    jsonPayload: unknown | null;
    exportUrl: string | null;
    isRunning: boolean;
    isPulling: boolean;
    isExpanded: boolean;
}

export interface DatabaseBrowserState {
    selectedTable: TableId;
    tableCache: TableCache;
    isLoading: boolean;
}

export interface AppState {
    activePage: PageId;
    diluAgent: DiluAgentState;
    databaseBrowser: DatabaseBrowserState;
}

interface AppStateContextValue {
    state: AppState;
    setActivePage: (page: PageId) => void;
    updateDiluAgent: (updates: Partial<DiluAgentState>) => void;
    updateDatabaseBrowser: (updates: Partial<DatabaseBrowserState>) => void;
    setTableData: (tableId: TableId, data: TableData) => void;
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
    isRunning: false,
    isPulling: false,
    isExpanded: false,
};

const DEFAULT_DATABASE_BROWSER_STATE: DatabaseBrowserState = {
    selectedTable: "sessions",
    tableCache: {
        sessions: null,
        livertox: null,
        drugs: null,
    },
    isLoading: false,
};

const DEFAULT_APP_STATE: AppState = {
    activePage: "dili-agent",
    diluAgent: DEFAULT_DILU_AGENT_STATE,
    databaseBrowser: DEFAULT_DATABASE_BROWSER_STATE,
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

    const updateDatabaseBrowser = useCallback((updates: Partial<DatabaseBrowserState>) => {
        setState((prev) => ({
            ...prev,
            databaseBrowser: { ...prev.databaseBrowser, ...updates },
        }));
    }, []);

    const setTableData = useCallback((tableId: TableId, data: TableData) => {
        setState((prev) => ({
            ...prev,
            databaseBrowser: {
                ...prev.databaseBrowser,
                tableCache: {
                    ...prev.databaseBrowser.tableCache,
                    [tableId]: data,
                },
            },
        }));
    }, []);

    const value = useMemo<AppStateContextValue>(
        () => ({
            state,
            setActivePage,
            updateDiluAgent,
            updateDatabaseBrowser,
            setTableData,
        }),
        [state, setActivePage, updateDiluAgent, updateDatabaseBrowser, setTableData],
    );

    return (
        <AppStateContext.Provider value={value}>
            {children}
        </AppStateContext.Provider>
    );
}
