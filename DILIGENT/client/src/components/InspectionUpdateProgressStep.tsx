import React from "react";

import { InspectionUpdateJobState } from "../hooks/useInspectionUpdateJob";
import { InspectionJobPanel } from "./InspectionJobPanel";

type InspectionUpdateProgressStepProps = {
  job: InspectionUpdateJobState;
  fallbackMessage: string;
  onCancel: () => void;
  onReset: () => void;
};

export function InspectionUpdateProgressStep({
  job,
  fallbackMessage,
  onCancel,
  onReset,
}: InspectionUpdateProgressStepProps): React.JSX.Element {
  return (
    <div className="inspection-widget">
      <h4>Step 3 · Progress</h4>
      <InspectionJobPanel job={job} fallbackMessage={fallbackMessage} />
      <div className="inspection-pager-actions">
        {job.running && (
          <button type="button" className="btn btn-secondary inspection-mini-btn" onClick={onCancel}>
            Cancel
          </button>
        )}
        {!job.running && (
          <button type="button" className="btn btn-secondary inspection-mini-btn" onClick={onReset}>
            New run
          </button>
        )}
      </div>
    </div>
  );
}

