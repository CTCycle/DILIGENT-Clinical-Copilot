import React from "react";

interface BooleanToggleProps {
  readonly id: string;
  readonly label: string;
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
}

export function BooleanToggle({
  id,
  label,
  checked,
  onChange,
}: BooleanToggleProps): React.JSX.Element {
  return (
    <div className="toggle-row">
      <span className="toggle-label">{label}</span>
      <label className="toggle-switch">
        <span className="visually-hidden">{label}</span>
        <input
          type="checkbox"
          id={id}
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="toggle-track" aria-hidden="true">
          <span className="toggle-thumb" />
        </span>
      </label>
    </div>
  );
}

