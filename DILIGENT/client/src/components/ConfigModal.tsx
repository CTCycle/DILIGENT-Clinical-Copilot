import React from "react";

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------
const CloseIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
);

// ---------------------------------------------------------------------------
// ConfigModal
// ---------------------------------------------------------------------------
interface ConfigModalProps {
    readonly isOpen: boolean;
    readonly onClose: () => void;
    readonly children: React.ReactNode;
}

export function ConfigModal({ isOpen, onClose, children }: ConfigModalProps): React.JSX.Element | null {
    if (!isOpen) {
        return null;
    }

    return (
        <div
            className="modal-overlay"
        >
            <dialog className="modal-container" aria-modal="true" aria-labelledby="modal-title" open>
                <div className="modal-header">
                    <div className="modal-header-content">
                        <p className="modal-title" id="modal-title">Model Configurations</p>
                        <p className="modal-subtitle">Adjust runtime preferences for DILI analysis</p>
                    </div>
                    <button
                        className="modal-close"
                        type="button"
                        onClick={onClose}
                        aria-label="Close configuration modal"
                    >
                        <CloseIcon />
                    </button>
                </div>
                <div className="modal-body">
                    {children}
                </div>
            </dialog>
        </div>
    );
}
