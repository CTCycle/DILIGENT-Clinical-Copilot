import React from "react";

const KeyIcon = (): React.JSX.Element => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="8" cy="15" r="3" />
    <path d="M11 15h10" />
    <path d="M18 12v6" />
  </svg>
);

type SelectableProviderAccessCardProps = {
  readonly variant: "selectable";
  readonly label: string;
  readonly isActive: boolean;
  readonly disabled: boolean;
  readonly onSelect: () => void;
  readonly onManageKeys: () => void;
  readonly manageKeyAriaLabel: string;
};

type CompactProviderAccessCardProps = {
  readonly variant: "compact";
  readonly label: string;
  readonly hint: string;
  readonly disabled: boolean;
  readonly onManageKeys: () => void;
  readonly manageKeyAriaLabel: string;
};

type ProviderAccessCardProps =
  | SelectableProviderAccessCardProps
  | CompactProviderAccessCardProps;

export function ProviderAccessCard(
  props: ProviderAccessCardProps,
): React.JSX.Element {
  const isCompact = props.variant === "compact";
  const className = [
    "model-config-provider-card",
    !isCompact && props.isActive ? "is-active" : "",
    isCompact ? "model-config-provider-card-compact" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={className}>
      {isCompact ? (
        <button
          className="model-config-provider-button"
          type="button"
          onClick={props.onManageKeys}
          disabled={props.disabled}
        >
          <span>{props.label}</span>
          <span className="model-config-provider-hint">{props.hint}</span>
        </button>
      ) : (
        <button
          className="model-config-provider-button"
          type="button"
          onClick={props.onSelect}
          disabled={props.disabled}
        >
          <span>{props.label}</span>
        </button>
      )}
      <button
        className="model-config-provider-key"
        type="button"
        onClick={props.onManageKeys}
        disabled={props.disabled}
        aria-label={props.manageKeyAriaLabel}
      >
        <KeyIcon />
      </button>
    </div>
  );
}
