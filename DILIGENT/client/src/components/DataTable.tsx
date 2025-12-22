import React, { useRef, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface DataTableProps {
    readonly columns: ReadonlyArray<string>;
    readonly rows: ReadonlyArray<Record<string, unknown>>;
    readonly isLoading?: boolean;
    readonly emptyMessage?: string;
    readonly hasMore?: boolean;
    readonly isLoadingMore?: boolean;
    readonly onLoadMore?: () => void;
}

// ---------------------------------------------------------------------------
// DataTable
// ---------------------------------------------------------------------------
export function DataTable({
    columns,
    rows,
    isLoading = false,
    emptyMessage = "No data available",
    hasMore = false,
    isLoadingMore = false,
    onLoadMore,
}: Readonly<DataTableProps>): React.JSX.Element {
    const sentinelRef = useRef<HTMLDivElement>(null);

    // Set up IntersectionObserver for infinite scroll
    const handleObserver = useCallback(
        (entries: IntersectionObserverEntry[]) => {
            const [entry] = entries;
            if (entry.isIntersecting && hasMore && !isLoadingMore && onLoadMore) {
                onLoadMore();
            }
        },
        [hasMore, isLoadingMore, onLoadMore],
    );

    useEffect(() => {
        const sentinel = sentinelRef.current;
        if (!sentinel) return;

        const observer = new IntersectionObserver(handleObserver, {
            root: null,
            rootMargin: "200px",
            threshold: 0,
        });

        observer.observe(sentinel);

        return () => {
            observer.disconnect();
        };
    }, [handleObserver]);

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
        <div className="data-table-container">
            <div className="data-table-scroll-x">
                <div className="data-table-scroll-y">
                    <table className="data-table">
                        <thead>
                            <tr>
                                {columns.map((col) => (
                                    <th key={col}>{formatColumnName(col)}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((row) => (
                                <tr key={getRowKey(row, columns)}>
                                    {columns.map((col) => (
                                        <td key={col}>{formatCellValue(row[col])}</td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {/* Sentinel element for infinite scroll detection */}
                    <div ref={sentinelRef} className="scroll-sentinel" />
                    {isLoadingMore && (
                        <div className="data-table-loading-more">
                            <div className="spinner-wheel spinner-small" />
                            <span>Loading more rows...</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatColumnName(name: string): string {
    return name
        .replaceAll("_", " ")
        .replaceAll(/\b\w/g, (char) => char.toUpperCase());
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

function getRowKey(row: Record<string, unknown>, columns: ReadonlyArray<string>): string {
    const preferredIdColumn = columns.find((col) => /(^id$|_id$|id$)/i.test(col));
    if (preferredIdColumn) {
        const idValue = row[preferredIdColumn];
        if (idValue !== null && idValue !== undefined && idValue !== "") {
            return `${preferredIdColumn}:${String(idValue)}`;
        }
    }

    const rowKeyParts = columns.map((col) => String(row[col] ?? ""));
    return rowKeyParts.join("|");
}
