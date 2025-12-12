import React, { useEffect } from "react";
import { TableId, useAppState } from "../context/AppStateContext";
import { DataTable } from "../components/DataTable";
import { fetchTableData } from "../services/api";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const TABLE_OPTIONS: { id: TableId; label: string }[] = [
    { id: "sessions", label: "Clinical Sessions" },
    { id: "livertox", label: "LiverTox Data" },
    { id: "drugs", label: "Drugs Catalog" },
];

// ---------------------------------------------------------------------------
// DatabaseBrowserPage
// ---------------------------------------------------------------------------
export function DatabaseBrowserPage(): React.JSX.Element {
    const { state, updateDatabaseBrowser, setTableData } = useAppState();
    const { selectedTable, tableCache, isLoading } = state.databaseBrowser;

    const currentData = tableCache[selectedTable];

    // Fetch data when table changes (only if not cached)
    useEffect(() => {
        if (currentData !== null) {
            return; // Already cached
        }

        let cancelled = false;

        const loadData = async () => {
            updateDatabaseBrowser({ isLoading: true });
            try {
                const result = await fetchTableData(selectedTable);
                if (!cancelled) {
                    setTableData(selectedTable, result);
                }
            } catch (error) {
                console.error("Failed to load table data:", error);
            } finally {
                if (!cancelled) {
                    updateDatabaseBrowser({ isLoading: false });
                }
            }
        };

        loadData();

        return () => {
            cancelled = true;
        };
    }, [selectedTable, currentData, updateDatabaseBrowser, setTableData]);

    const handleTableChange = (tableId: TableId) => {
        updateDatabaseBrowser({ selectedTable: tableId });
    };

    const handleRefresh = async () => {
        updateDatabaseBrowser({ isLoading: true });
        try {
            const result = await fetchTableData(selectedTable);
            setTableData(selectedTable, result);
        } catch (error) {
            console.error("Failed to refresh table data:", error);
        } finally {
            updateDatabaseBrowser({ isLoading: false });
        }
    };

    // Stats
    const rowCount = currentData?.totalRows ?? 0;
    const columnCount = currentData?.columns.length ?? 0;

    return (
        <main className="page-container database-browser">
            <header className="page-header">
                <p className="eyebrow">DILIGENT Data</p>
                <h1>Database Browser</h1>
                <p className="lede">Browse historical sessions, drug catalogs, and reference data.</p>
            </header>

            <div className="browser-layout">
                {/* Controls Bar with Stats */}
                <div className="browser-controls">
                    <div className="table-selector">
                        <label htmlFor="table-select" className="field-label">Select Table</label>
                        <div className="selector-row">
                            <select
                                id="table-select"
                                value={selectedTable}
                                onChange={(e) => handleTableChange(e.target.value as TableId)}
                            >
                                {TABLE_OPTIONS.map((opt) => (
                                    <option key={opt.id} value={opt.id}>{opt.label}</option>
                                ))}
                            </select>
                            <button
                                type="button"
                                className="btn btn-secondary refresh-btn"
                                onClick={handleRefresh}
                                disabled={isLoading}
                                title="Refresh data"
                            >
                                <RefreshIcon />
                            </button>
                        </div>
                    </div>

                    {/* Stats Panel - inline beside dropdown */}
                    <aside className="stats-panel">
                        <div className="stats-header">
                            <h3>Statistics</h3>
                        </div>
                        <div className="stats-body">
                            <div className="stat-item">
                                <span className="stat-label">Rows:</span>
                                <span className="stat-value">{rowCount.toLocaleString()}</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Columns:</span>
                                <span className="stat-value">{columnCount}</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Table:</span>
                                <span className="stat-value stat-table-name">
                                    {TABLE_OPTIONS.find((t) => t.id === selectedTable)?.label ?? selectedTable}
                                </span>
                            </div>
                        </div>
                    </aside>
                </div>

                {/* Data Table - Full Width */}
                <div className="browser-content">
                    <div className="table-panel">
                        <DataTable
                            columns={currentData?.columns ?? []}
                            rows={currentData?.rows ?? []}
                            isLoading={isLoading}
                            emptyMessage="No data available for this table."
                        />
                    </div>
                </div>
            </div>
        </main>
    );
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------
const RefreshIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 4 23 10 17 10" />
        <polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </svg>
);
