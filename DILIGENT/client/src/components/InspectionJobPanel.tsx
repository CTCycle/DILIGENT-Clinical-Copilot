import React from "react";

import { InspectionUpdateJobState } from "../hooks/useInspectionUpdateJob";

type InspectionJobPanelProps = {
  job: InspectionUpdateJobState;
  fallbackMessage: string;
};

export function InspectionJobPanel({ job, fallbackMessage }: InspectionJobPanelProps): React.JSX.Element | null {
  if (!job.running && !job.message && !job.error && !job.summary) {
    return null;
  }

  const clampedProgress = Math.max(0, Math.min(100, job.progress));
  const message = job.error || job.message || fallbackMessage;
  const phaseText =
    job.stepIndex && job.stepCount
      ? `Step ${job.stepIndex} of ${job.stepCount}${job.phase ? ` · ${job.phase}` : ""}`
      : job.phase;

  return (
    <div className="inspection-job-panel" aria-live="polite">
      <div className="inspection-job-bar-track">
        <div className="inspection-job-bar-fill" style={{ width: `${clampedProgress}%` }} />
      </div>
      {phaseText && <p className="inspection-job-message">{phaseText}</p>}
      <p className="inspection-job-message">{message}</p>
      {job.summary && (
        <pre className="inspection-job-message">{JSON.stringify(job.summary, null, 2)}</pre>
      )}
    </div>
  );
}
