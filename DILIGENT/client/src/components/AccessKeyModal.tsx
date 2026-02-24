import React, { useEffect, useMemo, useState } from "react";

import {
    activateAccessKey,
    createAccessKey,
    deleteAccessKey,
    fetchAccessKeys,
} from "../services/api";
import { AccessKeyProvider, AccessKeyRecord } from "../types";

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
        <path d="M3 6h18" />
        <path d="M8 6V4h8v2" />
        <path d="M6 6l1 14h10l1-14" />
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
    const [keys, setKeys] = useState<AccessKeyRecord[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [newKeyValue, setNewKeyValue] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const [visibleRows, setVisibleRows] = useState<Record<number, boolean>>({});

    const hasKeys = keys.length > 0;
    const sortedKeys = useMemo(
        () => [...keys].sort((left, right) => Number(right.is_active) - Number(left.is_active) || right.id - left.id),
        [keys],
    );

    const loadKeys = async () => {
        setIsLoading(true);
        setErrorMessage("");
        try {
            const payload = await fetchAccessKeys(provider);
            setKeys(payload);
        } catch (error) {
            const message = error instanceof Error ? error.message : "Unable to load access keys.";
            setErrorMessage(message);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (!isOpen) {
            return;
        }
        void loadKeys();
    }, [isOpen, provider]);

    const handleAdd = async () => {
        const candidate = newKeyValue.trim();
        if (!candidate) {
            setErrorMessage("Please paste a key before adding.");
            return;
        }
        setIsSaving(true);
        setErrorMessage("");
        try {
            await createAccessKey(provider, candidate);
            setNewKeyValue("");
            await loadKeys();
        } catch (error) {
            const message = error instanceof Error ? error.message : "Unable to add access key.";
            setErrorMessage(message);
        } finally {
            setIsSaving(false);
        }
    };

    const handleActivate = async (keyId: number) => {
        setIsSaving(true);
        setErrorMessage("");
        try {
            const activated = await activateAccessKey(keyId);
            setKeys((current) => current.map((item) => ({
                ...item,
                is_active: item.id === activated.id,
                updated_at: item.id === activated.id ? activated.updated_at : item.updated_at,
                last_used_at: item.id === activated.id ? activated.last_used_at : item.last_used_at,
            })));
        } catch (error) {
            const message = error instanceof Error ? error.message : "Unable to activate access key.";
            setErrorMessage(message);
        } finally {
            setIsSaving(false);
        }
    };

    const handleDelete = async (keyId: number) => {
        setIsSaving(true);
        setErrorMessage("");
        try {
            await deleteAccessKey(keyId);
            setKeys((current) => current.filter((item) => item.id !== keyId));
            setVisibleRows((current) => {
                const next = { ...current };
                delete next[keyId];
                return next;
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : "Unable to delete access key.";
            setErrorMessage(message);
        } finally {
            setIsSaving(false);
        }
    };

    const toggleVisibility = (keyId: number) => {
        setVisibleRows((current) => ({ ...current, [keyId]: !current[keyId] }));
    };

    if (!isOpen) {
        return null;
    }

    return (
        <div className="modal-overlay">
            <dialog className="modal-container access-key-modal" aria-modal="true" aria-labelledby="access-key-modal-title" open>
                <div className="modal-header">
                    <div className="modal-header-content">
                        <p className="modal-title" id="access-key-modal-title">{providerLabel} Access Keys</p>
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
                        <p className="modal-section-title">Add New Key</p>
                        <div className="access-key-input-row">
                            <input
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
                        <p className="modal-section-title">Stored Keys</p>
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

                    {!!errorMessage && <p className="model-config-status-message">[ERROR] {errorMessage}</p>}
                </div>
            </dialog>
        </div>
    );
}
