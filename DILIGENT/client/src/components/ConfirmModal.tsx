import React from "react";

interface ConfirmModalProps {
    readonly isOpen: boolean;
    readonly title: string;
    readonly message: string;
    readonly confirmLabel: string;
    readonly cancelLabel: string;
    readonly onConfirm: () => void;
    readonly onCancel: () => void;
}

export function ConfirmModal({
    isOpen,
    title,
    message,
    confirmLabel,
    cancelLabel,
    onConfirm,
    onCancel,
}: ConfirmModalProps): React.JSX.Element | null {
    if (!isOpen) {
        return null;
    }

    return (
        <div className="modal-overlay">
            <dialog
                className="modal-container"
                aria-modal="true"
                aria-labelledby="confirm-modal-title"
                aria-describedby="confirm-modal-message"
                open
            >
                <div className="modal-header">
                    <div className="modal-header-content">
                        <h2 className="modal-title" id="confirm-modal-title">{title}</h2>
                        <p className="modal-subtitle" id="confirm-modal-message">{message}</p>
                    </div>
                </div>
                <div className="modal-footer">
                    <div className="confirm-modal-actions">
                        <button className="btn btn-secondary" type="button" onClick={onCancel}>
                            {cancelLabel}
                        </button>
                        <button className="btn btn-primary" type="button" onClick={onConfirm}>
                            {confirmLabel}
                        </button>
                    </div>
                </div>
            </dialog>
        </div>
    );
}
