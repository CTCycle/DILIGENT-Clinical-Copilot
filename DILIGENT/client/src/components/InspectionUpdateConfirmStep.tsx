import React from "react";

type InspectionUpdateConfirmStepProps = {
  targetLabel: string;
  values: Record<string, unknown>;
  disabled: boolean;
  onBack: () => void;
  onStart: () => void;
};

export function InspectionUpdateConfirmStep({
  targetLabel,
  values,
  disabled,
  onBack,
  onStart,
}: InspectionUpdateConfirmStepProps): React.JSX.Element {
  return (
    <div className="inspection-widget">
      <h4>Step 2 · Confirm and Run</h4>
      <p className="inspection-loading-note">Target: {targetLabel}</p>
      <div className="inspection-scroll-frame inspection-scroll-frame-compact">
        <table className="inspection-table inspection-table-dense">
          <thead>
            <tr>
              <th>Setting</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(values).map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{String(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="inspection-pager-actions">
        <button type="button" className="btn btn-secondary inspection-mini-btn" onClick={onBack} disabled={disabled}>
          Back
        </button>
        <button type="button" className="btn btn-primary inspection-mini-btn" onClick={onStart} disabled={disabled}>
          Start update
        </button>
      </div>
    </div>
  );
}

