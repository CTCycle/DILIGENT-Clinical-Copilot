import React from "react";
import { ModalShell } from "./ModalShell";

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
    return (
        <ModalShell
            isOpen={isOpen}
            ariaLabelledBy="confirm-modal-title"
            ariaDescribedBy="confirm-modal-message"
            title={title}
            subtitle={message}
            titleId="confirm-modal-title"
            footer={(
                <div className="confirm-modal-actions">
                    <button className="btn btn-secondary" type="button" onClick={onCancel}>
                        {cancelLabel}
                    </button>
                    <button className="btn btn-primary" type="button" onClick={onConfirm}>
                        {confirmLabel}
                    </button>
                </div>
            )}
        >
            <span id="confirm-modal-message" className="visually-hidden">
                {message}
            </span>
        </ModalShell>
    );
}
