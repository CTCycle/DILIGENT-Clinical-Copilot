import { useCallback, useEffect, useRef } from "react";

import { fetchModelPullJobStatus, startModelPullJob } from "../services/api";
import { JobStatus, OllamaPullJobResult } from "../types";

type LoadModelConfigOptions = {
  readonly syncDraft?: boolean;
};

interface UseModelPullActionsProps {
  readonly setPulling: (isPulling: boolean) => void;
  readonly setStatusMessage: (message: string) => void;
  readonly loadModelConfig: (options?: LoadModelConfigOptions) => Promise<void>;
  readonly setModelPullProgress: (
    modelName: string,
    progress: ModelPullProgressState | null,
  ) => void;
}

interface UseModelPullActionsResult {
  readonly pullModelByName: (requestedModelName: string) => Promise<void>;
  readonly installRequiredModels: (modelNames: readonly string[]) => Promise<void>;
}

export type ModelPullProgressState = {
  progress: number;
  status: JobStatus;
  message: string;
};

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = [
  "completed",
  "failed",
  "cancelled",
];

function isTerminalStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

function resolvePullProgressMessage(
  modelName: string,
  status: JobStatus,
  result: OllamaPullJobResult | null,
): string {
  const progressMessage = result?.progress_message;
  if (typeof progressMessage === "string" && progressMessage.trim()) {
    return progressMessage;
  }
  if (status === "completed") {
    return `Model '${modelName}' is available locally.`;
  }
  if (status === "cancelled") {
    return `Pull cancelled for '${modelName}'.`;
  }
  if (status === "failed") {
    return `Pull failed for '${modelName}'.`;
  }
  return `Pulling '${modelName}' from Ollama...`;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, milliseconds);
  });
}

function resolveSuccessMessage(models: readonly string[]): string {
  if (models.length === 1) {
    return `[INFO] Model available locally: ${models[0]}.`;
  }
  return `[INFO] Models available locally: ${models.join(", ")}.`;
}

export function useModelPullActions({
  setPulling,
  setStatusMessage,
  loadModelConfig,
  setModelPullProgress,
}: UseModelPullActionsProps): UseModelPullActionsResult {
  const isMountedRef = useRef(true);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const updateModelPullProgress = useCallback(
    (modelName: string, progress: ModelPullProgressState | null) => {
      if (!isMountedRef.current) {
        return;
      }
      setModelPullProgress(modelName, progress);
    },
    [setModelPullProgress],
  );

  const pollPullJob = useCallback(
    async (
      modelName: string,
      jobId: string,
      pollIntervalMs: number,
    ): Promise<void> => {
      const safeIntervalMs = Math.max(250, pollIntervalMs);
      const requestTimeoutSeconds = Math.min(
        30,
        Math.max(5, Math.ceil((safeIntervalMs / 1000) * 4)),
      );

      while (true) {
        const payload = await fetchModelPullJobStatus(jobId, requestTimeoutSeconds);
        const progress = Math.max(0, Math.min(100, payload.progress));
        const message = resolvePullProgressMessage(
          modelName,
          payload.status,
          payload.result,
        );

        updateModelPullProgress(modelName, {
          progress,
          status: payload.status,
          message,
        });

        if (isTerminalStatus(payload.status)) {
          if (payload.status === "completed") {
            return;
          }
          const errorMessage =
            payload.error?.trim() ||
            resolvePullProgressMessage(modelName, payload.status, payload.result);
          throw new Error(`[ERROR] ${errorMessage}`);
        }

        await delay(safeIntervalMs);
      }
    },
    [updateModelPullProgress],
  );

  const runPull = useCallback(
    async (models: readonly string[], startMessage: string): Promise<void> => {
      const requestedModels = Array.from(
        new Set(models.map((model) => model.trim()).filter((model) => model.length > 0)),
      );
      if (!requestedModels.length) {
        return;
      }

      setPulling(true);
      setStatusMessage(startMessage);
      let pullFailed = false;
      let failureMessage = "";
      const completedModels: string[] = [];

      try {
        for (const modelName of requestedModels) {
          updateModelPullProgress(modelName, {
            progress: 1,
            status: "pending",
            message: `Starting pull for '${modelName}'...`,
          });

          try {
            const start = await startModelPullJob(modelName);
            const intervalMs = Math.max(250, Math.round(start.poll_interval * 1000));
            await pollPullJob(modelName, start.job_id, intervalMs);
            completedModels.push(modelName);
          } catch (error) {
            pullFailed = true;
            const description =
              error instanceof Error ? error.message : `Failed to pull '${modelName}'.`;
            failureMessage = description.startsWith("[ERROR]")
              ? description
              : `[ERROR] ${description}`;
            break;
          } finally {
            updateModelPullProgress(modelName, null);
          }
        }

        if (!pullFailed) {
          setStatusMessage(resolveSuccessMessage(completedModels));
        }
      } finally {
        try {
          await loadModelConfig({ syncDraft: false });
        } catch (error) {
          const description =
            error instanceof Error ? error.message : "Unable to refresh model catalog.";
          if (!pullFailed) {
            pullFailed = true;
            failureMessage = `[ERROR] ${description}`;
          }
        }

        if (pullFailed) {
          setStatusMessage(failureMessage);
        }
        setPulling(false);
      }
    },
    [
      loadModelConfig,
      pollPullJob,
      setPulling,
      setStatusMessage,
      updateModelPullProgress,
    ],
  );

  const pullModelByName = useCallback(
    async (requestedModelName: string): Promise<void> => {
      const candidate = requestedModelName.trim();
      if (!candidate) {
        setStatusMessage("[ERROR] Enter a model name to pull from Ollama.");
        return;
      }
      await runPull([candidate], `[INFO] Pulling '${candidate}' from Ollama...`);
    },
    [runPull, setStatusMessage],
  );

  const installRequiredModels = useCallback(
    async (modelNames: readonly string[]): Promise<void> => {
      if (!modelNames.length) {
        return;
      }
      await runPull(
        modelNames,
        `[INFO] Installing required models: ${modelNames.join(", ")}.`,
      );
    },
    [runPull],
  );

  return {
    pullModelByName,
    installRequiredModels,
  };
}
