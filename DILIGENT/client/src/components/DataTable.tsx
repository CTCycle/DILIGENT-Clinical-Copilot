import React from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface DataTableProps {
    columns: string[];
    rows: Record<string, unknown>[];
    isLoading?: boolean;
    emptyMessage?: string;
}

// ---------------------------------------------------------------------------
// DataTable
// ---------------------------------------------------------------------------
export function DataTable({
    columns,
    rows,
    isLoading = false,
    emptyMessage = "No data available",
}: DataTableProps): React.JSX.Element {
    if (isLoading) {
        return (
            <div className="data-table-loading">
                <div className="spinner-wheel" />
                <p>Loading data...</p>
            </div>
        );
    }

    if (rows.length === 0) {
        return (
            <div className="data-table-empty">
                <p>{emptyMessage}</p>
            </div>
        );
    }

    return (
        <div className="data-table-wrapper">
            <table className="data-table">
                <thead>
                    <tr>
                        {columns.map((col) => (
                            <th key={col}>{formatColumnName(col)}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                            {columns.map((col) => (
                                <td key={col}>{formatCellValue(row[col])}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatColumnName(name: string): string {
    return name
        .replace(/_/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatCellValue(value: unknown): string {
    if (value === null || value === undefined) {
        return "—";
    }
    if (typeof value === "object") {
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    }
    return String(value);
}
