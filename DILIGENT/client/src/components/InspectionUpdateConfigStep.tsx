import React from "react";

type InspectionUpdateConfigStepProps = {
  values: Record<string, unknown>;
  allowedFields: string[];
  disabled: boolean;
  onChange: (field: string, value: unknown) => void;
  onNext: () => void;
};

function parseRawValue(rawValue: string, current: unknown): unknown {
  if (typeof current === "number") {
    const numeric = Number.parseFloat(rawValue);
    return Number.isFinite(numeric) ? numeric : current;
  }
  if (typeof current === "boolean") {
    return rawValue === "true";
  }
  return rawValue;
}

export function InspectionUpdateConfigStep({
  values,
  allowedFields,
  disabled,
  onChange,
  onNext,
}: InspectionUpdateConfigStepProps): React.JSX.Element {
  return (
    <div className="inspection-widget">
      <h4>Step 1 · Configuration</h4>
      <div className="inspection-controls inspection-controls-sessions">
        {allowedFields.map((field) => {
          const value = values[field];
          if (typeof value === "boolean") {
            return (
              <label key={field} className="field checkbox">
                <input
                  type="checkbox"
                  checked={value}
                  onChange={(event) => onChange(field, event.target.checked)}
                  disabled={disabled}
                />
                <span className="field-label">{field}</span>
              </label>
            );
          }
          return (
            <label key={field} className="field">
              <span className="field-label">{field}</span>
              <input
                type={typeof value === "number" ? "number" : "text"}
                value={value === null || value === undefined ? "" : String(value)}
                onChange={(event) => onChange(field, parseRawValue(event.target.value, value))}
                disabled={disabled}
              />
            </label>
          );
        })}
      </div>
      <div className="inspection-pager-actions">
        <button
          type="button"
          className="btn btn-primary inspection-mini-btn"
          onClick={onNext}
          disabled={disabled}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

