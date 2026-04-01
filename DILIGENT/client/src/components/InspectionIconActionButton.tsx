import React from "react";

type InspectionIconActionButtonProps = {
  ariaLabel: string;
  title: string;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "danger";
  children: React.ReactNode;
};

export function InspectionIconActionButton({
  ariaLabel,
  title,
  onClick,
  disabled = false,
  variant = "default",
  children,
}: InspectionIconActionButtonProps): React.JSX.Element {
  const variantClass =
    variant === "primary" ? "is-primary" : variant === "danger" ? "is-danger" : "";
  const className = variantClass
    ? `inspection-icon-button ${variantClass}`
    : "inspection-icon-button";

  return (
    <button
      type="button"
      className={className}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      title={title}
    >
      {children}
    </button>
  );
}
