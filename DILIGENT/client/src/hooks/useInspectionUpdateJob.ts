import { useCallback, useEffect, useRef, useState } from "react";

import { JobCancelResponse, JobStartResponse, JobStatus, InspectionUpdateJobStatusResponse } from "../types";

export type InspectionUpdateJobState = {
  jobId: string | null;
  running: boolean;
  progress: number;
  status: JobStatus | null;
  phase: string | null;
  stepIndex: number | null;
  stepCount: number | null;
  message: string | null;
  summary: Record<string, unknown> | null;
  error: string | null;
};

type UseInspectionUpdateJobParams = {
  startJob: (payload?: Record<string, unknown>) => Promise<JobStartResponse>;
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
  triggerUpdate: (payload?: Record<string, unknown>) => Promise<void>;
};

const DEFAULT_INSPECTION_JOB_STATE: InspectionUpdateJobState = {
  jobId: null,
  running: false,
  progress: 0,
  status: null,
  phase: null,
  stepIndex: null,
  stepCount: null,
  message: null,
  summary: null,
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
      const phase = typeof payload.result?.phase === "string" ? payload.result.phase : null;
      const stepIndex =
        typeof payload.result?.step_index === "number" ? payload.result.step_index : null;
      const stepCount =
        typeof payload.result?.step_count === "number" ? payload.result.step_count : null;
      const summary =
        payload.result?.summary && typeof payload.result.summary === "object"
          ? (payload.result.summary as Record<string, unknown>)
          : null;

      setState({
        jobId,
        running,
        progress: payload.progress,
        status: payload.status,
        phase,
        stepIndex,
        stepCount,
        message: progressMessage,
        summary,
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

  const triggerUpdate = useCallback(async (payload?: Record<string, unknown>): Promise<void> => {
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
      const start = await startJob(payload);
      const intervalMs = Math.max(250, Math.round(start.poll_interval * 1000));
      setState({
        jobId: start.job_id,
        running: true,
        progress: 1,
        status: start.status,
        phase: null,
        stepIndex: null,
        stepCount: null,
        message: startMessage,
        summary: null,
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
