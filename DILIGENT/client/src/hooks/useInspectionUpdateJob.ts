import { useCallback, useEffect, useRef, useState } from "react";

import { JobCancelResponse, JobStartResponse, JobStatus, InspectionUpdateJobStatusResponse } from "../types";

export type InspectionUpdateJobState = {
  jobId: string | null;
  running: boolean;
  progress: number;
  status: JobStatus | null;
  message: string | null;
  error: string | null;
};

type UseInspectionUpdateJobParams = {
  startJob: () => Promise<JobStartResponse>;
  fetchStatus: (jobId: string) => Promise<InspectionUpdateJobStatusResponse>;
  cancelJob: (jobId: string) => Promise<JobCancelResponse>;
  onCompleted?: () => Promise<void> | void;
  startMessage: string;
  startErrorMessage: string;
  cancelErrorMessage: string;
  pollErrorMessage: string;
};

type UseInspectionUpdateJobResult = {
  state: InspectionUpdateJobState;
  stopPolling: () => void;
  triggerUpdate: () => Promise<void>;
};

const DEFAULT_INSPECTION_JOB_STATE: InspectionUpdateJobState = {
  jobId: null,
  running: false,
  progress: 0,
  status: null,
  message: null,
  error: null,
};

function isRunningStatus(status: JobStatus): boolean {
  return status === "pending" || status === "running";
}

function resolveErrorDescription(error: unknown, fallbackMessage: string): string {
  return error instanceof Error ? error.message : fallbackMessage;
}

export function useInspectionUpdateJob({
  startJob,
  fetchStatus,
  cancelJob,
  onCompleted,
  startMessage,
  startErrorMessage,
  cancelErrorMessage,
  pollErrorMessage,
}: UseInspectionUpdateJobParams): UseInspectionUpdateJobResult {
  const [state, setState] = useState<InspectionUpdateJobState>(DEFAULT_INSPECTION_JOB_STATE);
  const timerRef = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);

  const stopPolling = useCallback((): void => {
    if (timerRef.current !== null) {
      globalThis.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const pollJob = useCallback(async (jobId: string, intervalMs: number): Promise<void> => {
    try {
      const payload = await fetchStatus(jobId);
      const running = isRunningStatus(payload.status);
      const progressMessage = payload.result?.progress_message ?? null;

      setState({
        jobId,
        running,
        progress: payload.progress,
        status: payload.status,
        message: progressMessage,
        error: payload.status === "failed" ? payload.error : null,
      });

      if (running) {
        timerRef.current = globalThis.setTimeout(() => {
          void pollJob(jobId, intervalMs);
        }, intervalMs);
        return;
      }

      stopPolling();
      if (payload.status === "completed") {
        await onCompleted?.();
      }
    } catch (error) {
      stopPolling();
      setState((previous) => ({
        ...previous,
        running: false,
        status: "failed",
        error: resolveErrorDescription(error, pollErrorMessage),
      }));
    }
  }, [fetchStatus, onCompleted, pollErrorMessage, stopPolling]);

  const triggerUpdate = useCallback(async (): Promise<void> => {
    if (state.running && state.jobId) {
      try {
        await cancelJob(state.jobId);
        setState((previous) => ({
          ...previous,
          message: "Cancellation requested",
          error: null,
        }));
      } catch (error) {
        setState((previous) => ({
          ...previous,
          error: resolveErrorDescription(error, cancelErrorMessage),
        }));
      }
      return;
    }

    try {
      const start = await startJob();
      const intervalMs = Math.max(250, Math.round(start.poll_interval * 1000));
      setState({
        jobId: start.job_id,
        running: true,
        progress: 1,
        status: start.status,
        message: startMessage,
        error: null,
      });
      stopPolling();
      await pollJob(start.job_id, intervalMs);
    } catch (error) {
      setState((previous) => ({
        ...previous,
        running: false,
        error: resolveErrorDescription(error, startErrorMessage),
      }));
    }
  }, [
    cancelErrorMessage,
    cancelJob,
    pollJob,
    startErrorMessage,
    startJob,
    startMessage,
    state.jobId,
    state.running,
    stopPolling,
  ]);

  return {
    state,
    stopPolling,
    triggerUpdate,
  };
}
