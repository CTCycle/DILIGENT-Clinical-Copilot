import React from "react";

type ModalShellProps = {
  readonly isOpen: boolean;
  readonly ariaLabelledBy?: string;
  readonly ariaDescribedBy?: string;
  readonly ariaLabel?: string;
  readonly title: string;
  readonly subtitle?: string;
  readonly titleId?: string;
  readonly dialogClassName?: string;
  readonly closeLabel?: string;
  readonly onClose?: () => void;
  readonly closeIcon?: React.ReactNode;
  readonly children: React.ReactNode;
  readonly footer?: React.ReactNode;
};

export function ModalShell({
  isOpen,
  ariaLabelledBy,
  ariaDescribedBy,
  ariaLabel,
  title,
  subtitle,
  titleId,
  dialogClassName = "modal-container",
  closeLabel,
  onClose,
  closeIcon,
  children,
  footer,
}: ModalShellProps): React.JSX.Element | null {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay">
      <dialog
        className={dialogClassName}
        aria-modal="true"
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        aria-label={ariaLabel}
        open
      >
        <div className="modal-header">
          <div className="modal-header-content">
            <h2 className="modal-title" id={titleId}>{title}</h2>
            {subtitle ? <p className="modal-subtitle">{subtitle}</p> : null}
          </div>
          {onClose ? (
            <button
              className="modal-close"
              type="button"
              onClick={onClose}
              aria-label={closeLabel ?? "Close modal"}
            >
              {closeIcon}
            </button>
          ) : null}
        </div>
        <div className="modal-body">{children}</div>
        {footer ? <div className="modal-footer">{footer}</div> : null}
      </dialog>
    </div>
  );
}
