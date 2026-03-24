import React from "react";

import { useAccessKeyManager } from "../hooks/useAccessKeyManager";
import { StatusMessage } from "./StatusMessage";
import { AccessKeyProvider } from "../types";

const MASKED_KEY_LABEL = "********************";

const CloseIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
);

const LockIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="11" width="16" height="9" rx="2" />
        <path d="M8 11V8a4 4 0 1 1 8 0v3" />
    </svg>
);

const EyeIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6-10-6-10-6Z" />
        <circle cx="12" cy="12" r="3" />
    </svg>
);

const TrashIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 9v10H8V9h8Z" />
        <path d="M14.5 3h-5l-1 1H5v2h14V4h-3.5l-1-1Z" />
        <path d="M18 7H6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7Z" />
    </svg>
);

interface AccessKeyModalProps {
    readonly isOpen: boolean;
    readonly provider: AccessKeyProvider;
    readonly providerLabel: string;
    readonly onClose: () => void;
}

function obfuscateFingerprint(value: string): string {
    const fingerprint = (value || "").trim();
    if (fingerprint.length <= 10) {
        return `fp: ${fingerprint || "unknown"}`;
    }
    return `fp: ${fingerprint.slice(0, 6)}...${fingerprint.slice(-4)}`;
}

function formatTimestamp(value: string | null): string {
    if (!value) {
        return "Not used";
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return "Not used";
    }
    return parsed.toLocaleString();
}

export function AccessKeyModal({
    isOpen,
    provider,
    providerLabel,
    onClose,
}: AccessKeyModalProps): React.JSX.Element | null {
    const { state, actions } = useAccessKeyManager(provider, isOpen);
    const {
        sortedKeys,
        isLoading,
        isSaving,
        newKeyValue,
        errorMessage,
        visibleRows,
        hasKeys,
    } = state;
    const {
        setNewKeyValue,
        handleAdd,
        handleActivate,
        handleDelete,
        toggleVisibility,
    } = actions;

    if (!isOpen) {
        return null;
    }

    return (
        <div className="modal-overlay">
            <dialog className="modal-container access-key-modal" aria-modal="true" aria-labelledby="access-key-modal-title" open>
                <div className="modal-header">
                    <div className="modal-header-content">
                        <h2 className="modal-title" id="access-key-modal-title">{providerLabel} Access Keys</h2>
                        <p className="modal-subtitle">Stored encrypted at rest. Activate one key at a time for this provider.</p>
                    </div>
                    <button
                        className="modal-close"
                        type="button"
                        onClick={onClose}
                        aria-label="Close access key modal"
                    >
                        <CloseIcon />
                    </button>
                </div>

                <div className="modal-body">
                    <section className="modal-section">
                        <h3 className="modal-section-title">Add New Key</h3>
                        <div className="access-key-input-row">
                            <label className="visually-hidden" htmlFor="new-access-key-input">Access key</label>
                            <input
                                id="new-access-key-input"
                                className="access-key-input"
                                type="password"
                                placeholder="Paste access key"
                                value={newKeyValue}
                                onChange={(event) => setNewKeyValue(event.target.value)}
                                disabled={isSaving}
                            />
                            <button className="btn btn-primary access-key-add-btn" type="button" onClick={() => { void handleAdd(); }} disabled={isSaving}>
                                Add
                            </button>
                        </div>
                    </section>

                    <section className="modal-section">
                        <h3 className="modal-section-title">Stored Keys</h3>
                        {isLoading && <p className="access-key-empty">Loading keys...</p>}
                        {!isLoading && !hasKeys && <p className="access-key-empty">No keys stored for this provider.</p>}
                        {!isLoading && hasKeys && (
                            <ul className="access-key-list">
                                {sortedKeys.map((item) => (
                                    <li key={item.id} className={`access-key-row ${item.is_active ? "is-active" : ""}`}>
                                        <div className="access-key-meta">
                                            <p className="access-key-fingerprint">
                                                {visibleRows[item.id] ? obfuscateFingerprint(item.fingerprint) : MASKED_KEY_LABEL}
                                            </p>
                                            <p className="access-key-timestamp">Last used: {formatTimestamp(item.last_used_at)}</p>
                                        </div>
                                        <div className="access-key-actions">
                                            <button
                                                className={`access-key-action ${item.is_active ? "is-active" : ""}`}
                                                type="button"
                                                onClick={() => { void handleActivate(item.id); }}
                                                disabled={isSaving}
                                                title={item.is_active ? "Active key" : "Activate key"}
                                                aria-label={item.is_active ? "Active key" : "Activate key"}
                                            >
                                                <LockIcon />
                                            </button>
                                            <button
                                                className="access-key-action"
                                                type="button"
                                                onClick={() => toggleVisibility(item.id)}
                                                disabled={isSaving}
                                                title="Toggle fingerprint view"
                                                aria-label="Toggle fingerprint view"
                                            >
                                                <EyeIcon />
                                            </button>
                                            <button
                                                className="access-key-action is-danger"
                                                type="button"
                                                onClick={() => { void handleDelete(item.id); }}
                                                disabled={isSaving}
                                                title="Delete key"
                                                aria-label="Delete key"
                                            >
                                                <TrashIcon />
                                            </button>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </section>

                    <StatusMessage
                        message={errorMessage ? `[ERROR] ${errorMessage}` : ""}
                        tone="is-error"
                    />
                </div>
            </dialog>
        </div>
    );
}
