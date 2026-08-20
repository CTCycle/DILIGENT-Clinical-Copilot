import { API_BASE_URL } from "../constants";
import {
  RevisionArtifactListResponse,
  RevisionClinicalReviewUpdateRequest,
  RevisionClinicalReviewUpdateResponse,
  RevisionJobStatusResponse,
  RevisionPipelineStepListResponse,
  SessionRevisionRequest,
} from "../models/revision-types";
import { JobCancelResponse, JobStartResponse } from "../models/types";
import { requestJson } from "./http-api";

export async function startSessionRevisionJob(
  sessionId: number,
  payload: SessionRevisionRequest,
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/revision/jobs`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchSessionRevisionJobStatus(
  jobId: string,
): Promise<RevisionJobStatusResponse> {
  return requestJson<RevisionJobStatusResponse>(
    `${API_BASE_URL}/inspection/sessions/revision/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET" },
  );
}

export async function cancelSessionRevisionJob(
  jobId: string,
): Promise<JobCancelResponse> {
  return requestJson<JobCancelResponse>(
    `${API_BASE_URL}/inspection/sessions/revision/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE" },
  );
}

export async function fetchRevisionPipelineSteps(
  pipelineRunId: string,
): Promise<RevisionPipelineStepListResponse> {
  return requestJson<RevisionPipelineStepListResponse>(
    `${API_BASE_URL}/inspection/sessions/revision/pipeline-runs/${encodeURIComponent(pipelineRunId)}/steps`,
    { method: "GET" },
  );
}

export async function fetchRevisionArtifacts(
  sessionId: number,
  versionId: number,
): Promise<RevisionArtifactListResponse> {
  return requestJson<RevisionArtifactListResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/versions/${encodeURIComponent(String(versionId))}/artifacts`,
    { method: "GET" },
  );
}

export async function updateRevisionClinicalReview(
  sessionId: number,
  versionId: number,
  payload: RevisionClinicalReviewUpdateRequest,
): Promise<RevisionClinicalReviewUpdateResponse> {
  return requestJson<RevisionClinicalReviewUpdateResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/versions/${encodeURIComponent(String(versionId))}/clinical-review`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}
