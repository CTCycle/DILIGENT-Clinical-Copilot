import { Injectable, inject } from '@angular/core';

import { JobStatus } from '../models/types';
import { resolvePollIntervalMs } from './clinical-api';
import { JobPollingService } from './job-polling.service';
import {
  fetchModelPullJobStatus,
  startModelPullJob,
} from './model-config-api';

export type ModelPullProgressState = {
  progress: number;
  status: JobStatus;
  message: string;
};

export type ModelPullProgressCallback = (
  modelName: string,
  progress: ModelPullProgressState | null,
) => void;

export type ModelPullJobResult = {
  completedModels: string[];
};

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ['completed', 'failed', 'cancelled'];

@Injectable({ providedIn: 'root' })
export class ModelPullJobService {
  private readonly jobPolling = inject(JobPollingService);

  async pullModels(
    modelNames: readonly string[],
    onProgress: ModelPullProgressCallback,
  ): Promise<ModelPullJobResult> {
    const requested = Array.from(new Set(modelNames.map((model) => model.trim()).filter(Boolean)));
    const completedModels: string[] = [];

    for (const modelName of requested) {
      onProgress(modelName, {
        progress: 1,
        status: 'pending',
        message: `Starting pull for '${modelName}'...`,
      });

      try {
        const start = await startModelPullJob(modelName);
        const intervalMs = resolvePollIntervalMs(start.poll_interval);
        await this.pollPullJob(modelName, start.job_id, intervalMs, onProgress);
        completedModels.push(modelName);
      } catch (error) {
        const description = error instanceof Error ? error.message : `Failed to pull '${modelName}'.`;
        throw new Error(description.startsWith('[ERROR]') ? description : `[ERROR] ${description}`);
      } finally {
        onProgress(modelName, null);
      }
    }

    return { completedModels };
  }

  private async pollPullJob(
    modelName: string,
    jobId: string,
    intervalMs: number,
    onProgress: ModelPullProgressCallback,
  ): Promise<void> {
    const safeIntervalMs = Math.max(250, intervalMs);
    const requestTimeoutSeconds = Math.min(30, Math.max(5, Math.ceil((safeIntervalMs / 1000) * 4)));

    await this.jobPolling.run({
      intervalMs: safeIntervalMs,
      pollStep: async () => {
        const payload = await fetchModelPullJobStatus(jobId, requestTimeoutSeconds);
        const progress = Math.max(0, Math.min(100, payload.progress));
        const message = this.resolvePullProgressMessage(modelName, payload.status, payload.result?.progress_message);

        onProgress(modelName, {
          progress,
          status: payload.status,
          message,
        });

        if (!TERMINAL_JOB_STATUSES.includes(payload.status)) {
          return true;
        }
        if (payload.status === 'completed') {
          return false;
        }
        const errorMessage = payload.error?.trim() || message;
        throw new Error(`[ERROR] ${errorMessage}`);
      },
    });
  }

  private resolvePullProgressMessage(
    modelName: string,
    status: JobStatus,
    progressMessage: string | undefined,
  ): string {
    if (typeof progressMessage === 'string' && progressMessage.trim()) {
      return progressMessage;
    }
    if (status === 'completed') {
      return `Model '${modelName}' is available locally.`;
    }
    if (status === 'cancelled') {
      return `Pull cancelled for '${modelName}'.`;
    }
    if (status === 'failed') {
      return `Pull failed for '${modelName}'.`;
    }
    return `Pulling '${modelName}' from Ollama...`;
  }
}
