import React from "react";

import { InspectionUpdateJobState } from "../hooks/useInspectionUpdateJob";

type InspectionJobPanelProps = {
  job: InspectionUpdateJobState;
  fallbackMessage: string;
};

export function InspectionJobPanel({ job, fallbackMessage }: InspectionJobPanelProps): React.JSX.Element | null {
  if (!job.running && !job.message && !job.error) {
    return null;
  }

  const clampedProgress = Math.max(0, Math.min(100, job.progress));
  const message = job.error || job.message || fallbackMessage;

  return (
    <div className="inspection-job-panel">
      <div className="inspection-job-bar-track">
        <div className="inspection-job-bar-fill" style={{ width: `${clampedProgress}%` }} />
      </div>
      <p className="inspection-job-message">{message}</p>
    </div>
  );
}
