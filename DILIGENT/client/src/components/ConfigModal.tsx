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
    isOpen: boolean;
    onClose: () => void;
    children: React.ReactNode;
}

export function ConfigModal({ isOpen, onClose, children }: ConfigModalProps): React.JSX.Element | null {
    if (!isOpen) {
        return null;
    }

    const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    return (
        <div className="modal-overlay" onClick={handleOverlayClick}>
            <div className="modal-container" role="dialog" aria-modal="true" aria-labelledby="modal-title">
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
            </div>
        </div>
    );
}
