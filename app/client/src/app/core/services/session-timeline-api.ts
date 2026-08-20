import { API_BASE_URL } from "../constants";
import {
  InspectionDeleteResponse,
  InspectionSessionTimeline,
  InspectionSessionTimelineListResponse,
  InspectionSessionTimelineRequest,
  InspectionTimelineJobStatusResponse,
} from "../models/inspection-types";
import { JobStartResponse } from "../models/types";
import { requestJson } from "./http-api";

const TIMELINE_REQUEST_TIMEOUT_SECONDS = 360;
const INSPECTION_JOB_STATUS_TIMEOUT_SECONDS = 20;

export async function fetchInspectionSessionTimelineList(
  sessionId: number,
): Promise<InspectionSessionTimelineListResponse> {
  return requestJson<InspectionSessionTimelineListResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timelines`,
    { method: "GET" },
    TIMELINE_REQUEST_TIMEOUT_SECONDS,
  );
}

export async function fetchInspectionSessionTimelineById(
  sessionId: number,
  timelineId: number,
): Promise<InspectionSessionTimeline> {
  return requestJson<InspectionSessionTimeline>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timelines/${encodeURIComponent(String(timelineId))}`,
    { method: "GET" },
    TIMELINE_REQUEST_TIMEOUT_SECONDS,
  );
}

export async function startInspectionSessionTimelineJob(
  sessionId: number,
  payload: InspectionSessionTimelineRequest = {},
): Promise<JobStartResponse> {
  return requestJson<JobStartResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timeline-jobs`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export async function fetchInspectionSessionTimelineJobStatus(
  sessionId: number,
  jobId: string,
): Promise<InspectionTimelineJobStatusResponse> {
  return requestJson<InspectionTimelineJobStatusResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timeline-jobs/${encodeURIComponent(jobId)}`,
    { method: "GET", headers: { "Cache-Control": "no-store" } },
    INSPECTION_JOB_STATUS_TIMEOUT_SECONDS,
  );
}

export async function deleteInspectionSessionTimeline(
  sessionId: number,
  timelineId: number,
): Promise<InspectionDeleteResponse> {
  return requestJson<InspectionDeleteResponse>(
    `${API_BASE_URL}/inspection/sessions/${encodeURIComponent(String(sessionId))}/timelines/${encodeURIComponent(String(timelineId))}`,
    { method: "DELETE" },
  );
}
