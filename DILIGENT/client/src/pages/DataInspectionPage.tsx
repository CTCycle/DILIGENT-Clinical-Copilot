import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRxNavUpdateJob,
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
  deleteInspectionSession,
  fetchInspectionLiverToxCatalog,
  fetchInspectionLiverToxExcerpt,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRxNavAliases,
  fetchInspectionRxNavCatalog,
  fetchInspectionRxNavUpdateJobStatus,
  fetchInspectionSessionReport,
  fetchInspectionSessions,
  startInspectionLiverToxUpdateJob,
  startInspectionRxNavUpdateJob,
} from "../services/api";
import {
  InspectionDateFilterMode,
  InspectionDrugAliasesResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxItem,
  InspectionRxNavItem,
  InspectionSessionItem,
  InspectionSessionStatus,
  JobStatus,
} from "../types";

const PAGE_LIMIT = 10;
const SEARCH_DEBOUNCE_MS = 250;

type UpdateJobState = {
  jobId: string | null;
  running: boolean;
  progress: number;
  status: JobStatus | null;
  message: string | null;
  error: string | null;
};

const DEFAULT_JOB_STATE: UpdateJobState = {
  jobId: null,
  running: false,
  progress: 0,
  status: null,
  message: null,
  error: null,
};

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  useEffect(() => {
    const timeoutId = globalThis.setTimeout(() => setDebouncedValue(value), delayMs);
    return () => globalThis.clearTimeout(timeoutId);
  }, [value, delayMs]);
  return debouncedValue;
}

function formatDateTime(value: string | null): string {
  if (!value) return "N/A";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatDuration(seconds: number | null): string {
  if (typeof seconds !== "number" || Number.isNaN(seconds) || seconds < 0) return "N/A";
  const rounded = Math.round(seconds);
  if (rounded < 60) return `${rounded}s`;
  return `${Math.floor(rounded / 60)}m ${rounded % 60}s`;
}

function statusLabel(value: InspectionSessionStatus): string {
  return value === "failed" ? "Failed" : "Successful";
}

function resolveExcerptFallbackMessage(error: unknown): string {
  const description =
    error instanceof Error ? error.message.toLowerCase() : "";
  if (description.includes("status 404") || description.includes("not found")) {
    return "No LiverTox excerpt is available for this drug.";
  }
  return "Unable to load this LiverTox excerpt right now.";
}

function ViewIcon(): React.JSX.Element {
  return (
    <svg className="inspection-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 5C7 5 2.73 8.11 1 12c1.73 3.89 6 7 11 7s9.27-3.11 11-7c-1.73-3.89-6-7-11-7Zm0 11a4 4 0 1 1 0-8 4 4 0 0 1 0 8Z" />
    </svg>
  );
}

function DeleteIcon(): React.JSX.Element {
  return (
    <svg className="inspection-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M16 9v10H8V9h8Zm-1.5-6h-5l-1 1H5v2h14V4h-3.5l-1-1ZM18 7H6v12c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7Z" />
    </svg>
  );
}

function ModifyIcon(): React.JSX.Element {
  return (
    <svg className="inspection-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="m3 17.25 9.81-9.82 2.76 2.76L5.76 20H3v-2.75Zm14.71-8.04-2.92-2.92 1.41-1.41a2 2 0 0 1 2.83 0l.09.09a2 2 0 0 1 0 2.83l-1.41 1.41Z" />
    </svg>
  );
}

export function DataInspectionPage(): React.JSX.Element {
  const [sessionItems, setSessionItems] = useState<InspectionSessionItem[]>([]);
  const [sessionTotal, setSessionTotal] = useState(0);
  const [sessionOffset, setSessionOffset] = useState(0);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [sessionSearchInput, setSessionSearchInput] = useState("");
  const [sessionStatusFilter, setSessionStatusFilter] = useState<InspectionSessionStatus | "all">("all");
  const [sessionDateMode, setSessionDateMode] = useState<InspectionDateFilterMode | "none">("none");
  const [sessionDate, setSessionDate] = useState("");
  const sessionSearch = useDebouncedValue(sessionSearchInput, SEARCH_DEBOUNCE_MS);

  const [rxnavItems, setRxnavItems] = useState<InspectionRxNavItem[]>([]);
  const [rxnavTotal, setRxnavTotal] = useState(0);
  const [rxnavOffset, setRxnavOffset] = useState(0);
  const [rxnavLoading, setRxnavLoading] = useState(false);
  const [rxnavError, setRxnavError] = useState<string | null>(null);
  const [rxnavSearchInput, setRxnavSearchInput] = useState("");
  const rxnavSearch = useDebouncedValue(rxnavSearchInput, SEARCH_DEBOUNCE_MS);

  const [livertoxItems, setLivertoxItems] = useState<InspectionLiverToxItem[]>([]);
  const [livertoxTotal, setLivertoxTotal] = useState(0);
  const [livertoxOffset, setLivertoxOffset] = useState(0);
  const [livertoxLoading, setLivertoxLoading] = useState(false);
  const [livertoxError, setLivertoxError] = useState<string | null>(null);
  const [livertoxSearchInput, setLivertoxSearchInput] = useState("");
  const livertoxSearch = useDebouncedValue(livertoxSearchInput, SEARCH_DEBOUNCE_MS);

  const [reportSession, setReportSession] = useState<InspectionSessionItem | null>(null);
  const [reportContent, setReportContent] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  const [aliasData, setAliasData] = useState<InspectionDrugAliasesResponse | null>(null);
  const [aliasLoading, setAliasLoading] = useState(false);
  const [aliasError, setAliasError] = useState<string | null>(null);

  const [excerptData, setExcerptData] = useState<InspectionLiverToxExcerptResponse | null>(null);
  const [excerptLoading, setExcerptLoading] = useState(false);
  const [excerptError, setExcerptError] = useState<string | null>(null);

  const [rxnavJob, setRxnavJob] = useState<UpdateJobState>(DEFAULT_JOB_STATE);
  const [livertoxJob, setLivertoxJob] = useState<UpdateJobState>(DEFAULT_JOB_STATE);
  const rxnavTimer = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);
  const livertoxTimer = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null);

  const stopRxnavPolling = useCallback(() => {
    if (rxnavTimer.current !== null) {
      globalThis.clearTimeout(rxnavTimer.current);
      rxnavTimer.current = null;
    }
  }, []);
  const stopLivertoxPolling = useCallback(() => {
    if (livertoxTimer.current !== null) {
      globalThis.clearTimeout(livertoxTimer.current);
      livertoxTimer.current = null;
    }
  }, []);
  const closeAliasModal = useCallback(() => {
    setAliasData(null);
    setAliasError(null);
    setAliasLoading(false);
  }, []);
  const closeExcerptModal = useCallback(() => {
    setExcerptData(null);
    setExcerptError(null);
    setExcerptLoading(false);
  }, []);

  const loadSessions = useCallback(async () => {
    setSessionLoading(true);
    setSessionError(null);
    try {
      const payload = await fetchInspectionSessions({
        search: sessionSearch,
        status: sessionStatusFilter === "all" ? undefined : sessionStatusFilter,
        date_mode: sessionDateMode === "none" ? undefined : sessionDateMode,
        date: sessionDateMode === "none" ? undefined : sessionDate || undefined,
        offset: sessionOffset,
        limit: PAGE_LIMIT,
      });
      setSessionItems(payload.items);
      setSessionTotal(payload.total);
    } catch (error) {
      setSessionItems([]);
      setSessionTotal(0);
      setSessionError(error instanceof Error ? error.message : "Failed to load sessions.");
    } finally {
      setSessionLoading(false);
    }
  }, [sessionDate, sessionDateMode, sessionOffset, sessionSearch, sessionStatusFilter]);

  const loadRxnav = useCallback(async () => {
    setRxnavLoading(true);
    setRxnavError(null);
    try {
      const payload = await fetchInspectionRxNavCatalog({ search: rxnavSearch, offset: rxnavOffset, limit: PAGE_LIMIT });
      setRxnavItems(payload.items);
      setRxnavTotal(payload.total);
    } catch (error) {
      setRxnavItems([]);
      setRxnavTotal(0);
      setRxnavError(error instanceof Error ? error.message : "Failed to load RxNav data.");
    } finally {
      setRxnavLoading(false);
    }
  }, [rxnavOffset, rxnavSearch]);

  const loadLivertox = useCallback(async () => {
    setLivertoxLoading(true);
    setLivertoxError(null);
    try {
      const payload = await fetchInspectionLiverToxCatalog({ search: livertoxSearch, offset: livertoxOffset, limit: PAGE_LIMIT });
      setLivertoxItems(payload.items);
      setLivertoxTotal(payload.total);
    } catch (error) {
      setLivertoxItems([]);
      setLivertoxTotal(0);
      setLivertoxError(error instanceof Error ? error.message : "Failed to load LiverTox data.");
    } finally {
      setLivertoxLoading(false);
    }
  }, [livertoxOffset, livertoxSearch]);

  useEffect(() => setSessionOffset(0), [sessionSearch, sessionStatusFilter, sessionDateMode, sessionDate]);
  useEffect(() => setRxnavOffset(0), [rxnavSearch]);
  useEffect(() => setLivertoxOffset(0), [livertoxSearch]);
  useEffect(() => { void loadSessions(); }, [loadSessions]);
  useEffect(() => { void loadRxnav(); }, [loadRxnav]);
  useEffect(() => { void loadLivertox(); }, [loadLivertox]);
  useEffect(() => () => { stopRxnavPolling(); stopLivertoxPolling(); }, [stopLivertoxPolling, stopRxnavPolling]);

  const pollRxnav = useCallback(async (jobId: string, intervalMs: number) => {
    try {
      const payload = await fetchInspectionRxNavUpdateJobStatus(jobId);
      const running = payload.status === "pending" || payload.status === "running";
      const progressMessage = payload.result && typeof payload.result.progress_message === "string" ? payload.result.progress_message : null;
      setRxnavJob({
        jobId,
        running,
        progress: payload.progress,
        status: payload.status,
        message: progressMessage,
        error: payload.status === "failed" ? payload.error : null,
      });
      if (running) {
        rxnavTimer.current = globalThis.setTimeout(() => { void pollRxnav(jobId, intervalMs); }, intervalMs);
      } else {
        stopRxnavPolling();
        if (payload.status === "completed") void loadRxnav();
      }
    } catch (error) {
      stopRxnavPolling();
      setRxnavJob((prev) => ({ ...prev, running: false, status: "failed", error: error instanceof Error ? error.message : "RxNav polling failed." }));
    }
  }, [loadRxnav, stopRxnavPolling]);

  const pollLivertox = useCallback(async (jobId: string, intervalMs: number) => {
    try {
      const payload = await fetchInspectionLiverToxUpdateJobStatus(jobId);
      const running = payload.status === "pending" || payload.status === "running";
      const progressMessage = payload.result && typeof payload.result.progress_message === "string" ? payload.result.progress_message : null;
      setLivertoxJob({
        jobId,
        running,
        progress: payload.progress,
        status: payload.status,
        message: progressMessage,
        error: payload.status === "failed" ? payload.error : null,
      });
      if (running) {
        livertoxTimer.current = globalThis.setTimeout(() => { void pollLivertox(jobId, intervalMs); }, intervalMs);
      } else {
        stopLivertoxPolling();
        if (payload.status === "completed") void loadLivertox();
      }
    } catch (error) {
      stopLivertoxPolling();
      setLivertoxJob((prev) => ({ ...prev, running: false, status: "failed", error: error instanceof Error ? error.message : "LiverTox polling failed." }));
    }
  }, [loadLivertox, stopLivertoxPolling]);

  const sessionStatusClass = useMemo(() => (status: InspectionSessionStatus) => (status === "failed" ? "is-failed" : "is-successful"), []);

  const renderPager = (total: number, offset: number, setOffset: (value: number) => void) => (
    <div className="inspection-pager">
      <span className="inspection-pager-range">
        {total > 0 ? offset + 1 : 0}-{total > 0 ? Math.min(total, offset + PAGE_LIMIT) : 0} of {total}
      </span>
      <div className="inspection-pager-actions">
        <button type="button" className="btn btn-secondary inspection-mini-btn" onClick={() => setOffset(Math.max(0, offset - PAGE_LIMIT))} disabled={offset <= 0}>Previous</button>
        <button type="button" className="btn btn-secondary inspection-mini-btn" onClick={() => setOffset(offset + PAGE_LIMIT)} disabled={offset + PAGE_LIMIT >= total}>Next</button>
      </div>
    </div>
  );

  if (reportSession !== null) {
    return (
      <main className="page-container inspection-page">
        <section className="inspection-report-view">
          <div className="inspection-report-header">
            <div>
              <p className="eyebrow">Recorded Session Report</p>
              <h1>Session #{reportSession.session_id}</h1>
              <p className="inspection-report-subtitle">{reportSession.patient_name || "Unknown patient"} · {formatDateTime(reportSession.session_timestamp)}</p>
            </div>
            <button type="button" className="btn btn-secondary" onClick={() => { setReportSession(null); setReportContent(""); setReportError(null); }}>Back</button>
          </div>
          <div className="report-shell inspection-report-shell">
            <div className="report-content is-expanded">
              {reportLoading && <div className="report-placeholder">Loading report...</div>}
              {!reportLoading && reportError && <div className="inspection-error-text">{reportError}</div>}
              {!reportLoading && !reportError && reportContent && <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown">{reportContent}</ReactMarkdown>}
              {!reportLoading && !reportError && !reportContent && <div className="report-placeholder">No report found for this session.</div>}
            </div>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="page-container inspection-page">
      <header className="page-header"><p className="eyebrow">Data Inspection</p><h1>Session and Knowledge Catalog</h1><p className="lede">Review recorded DILI sessions alongside RxNav and LiverTox knowledge records.</p></header>
      <section className="inspection-layout">
        <div className="inspection-column inspection-column-left">
          <div className="inspection-widget-header">
            <div>
              <h2>Sessions</h2>
              <p>Dense catalog of recorded DILI sessions</p>
            </div>
            <div className="inspection-controls inspection-controls-sessions">
              <input type="search" className="inspection-search" placeholder="Search sessions..." value={sessionSearchInput} onChange={(event) => setSessionSearchInput(event.target.value)} aria-label="Search sessions" />
              <div className="inspection-toggle-group" role="group" aria-label="Status filter">
                {(["all", "successful", "failed"] as const).map((value) => (
                  <button key={value} type="button" className={`inspection-toggle-pill ${sessionStatusFilter === value ? "is-active" : ""}`} onClick={() => setSessionStatusFilter(value)}>
                    {value === "all" ? "All" : value === "successful" ? "Successful" : "Failed"}
                  </button>
                ))}
              </div>
              <select className="inspection-select" value={sessionDateMode} onChange={(event) => setSessionDateMode(event.target.value as InspectionDateFilterMode | "none")} aria-label="Date filter mode">
                <option value="none">Date filter</option>
                <option value="before">Before</option>
                <option value="after">After</option>
                <option value="exact">Exact</option>
              </select>
              <input type="date" className="inspection-date" value={sessionDate} onChange={(event) => setSessionDate(event.target.value)} disabled={sessionDateMode === "none"} aria-label="Session date filter" />
            </div>
          </div>
          <div className="inspection-scroll-frame">
            <table className="inspection-table inspection-table-dense">
              <thead><tr><th>#</th><th>Patient</th><th>Date</th><th>Status</th><th>Total time</th><th aria-label="Actions" /></tr></thead>
              <tbody>
                {sessionItems.map((row, index) => (
                  <tr key={row.session_id}>
                    <td>{sessionOffset + index + 1}</td>
                    <td>{row.patient_name || "Unknown"}</td>
                    <td>{formatDateTime(row.session_timestamp)}</td>
                    <td><span className={`inspection-status-chip ${sessionStatusClass(row.status)}`}>{statusLabel(row.status)}</span></td>
                    <td>{formatDuration(row.total_duration)}</td>
                    <td className="inspection-actions-cell">
                      <button type="button" className="inspection-icon-button is-primary" onClick={() => { setReportLoading(true); setReportSession(row); setReportError(null); setReportContent(""); void fetchInspectionSessionReport(row.session_id).then((payload) => setReportContent(payload.report)).catch((error) => setReportError(error instanceof Error ? error.message : "Failed to load report.")).finally(() => setReportLoading(false)); }} aria-label={`View report for session ${row.session_id}`} title={`View report for session ${row.session_id}`}><ViewIcon /></button>
                      <button type="button" className="inspection-icon-button is-danger" onClick={() => { if (!globalThis.confirm("Delete this recorded session and report data?")) return; void deleteInspectionSession(row.session_id).then(() => { if (sessionItems.length === 1 && sessionOffset > 0) setSessionOffset(Math.max(0, sessionOffset - PAGE_LIMIT)); else void loadSessions(); }).catch((error) => setSessionError(error instanceof Error ? error.message : "Failed to delete session.")); }} aria-label={`Delete session ${row.session_id}`} title={`Delete session ${row.session_id}`}><DeleteIcon /></button>
                      <button type="button" className="inspection-icon-button" disabled aria-label="Modify session (not implemented)" title="Modify session (not implemented)"><ModifyIcon /></button>
                    </td>
                  </tr>
                ))}
                {!sessionLoading && sessionItems.length === 0 && <tr><td colSpan={6} className="inspection-empty-row">No sessions found.</td></tr>}
              </tbody>
            </table>
            {sessionLoading && <p className="inspection-loading-note">Loading sessions...</p>}
            {sessionError && <p className="inspection-error-text">{sessionError}</p>}
          </div>
          {renderPager(sessionTotal, sessionOffset, setSessionOffset)}
        </div>

        <div className="inspection-vertical-separator" aria-hidden="true" />

        <div className="inspection-column inspection-column-right">
          <div className="inspection-widget">
            <div className="inspection-widget-header">
              <div><h2>RxNav Data</h2><p>Canonical catalog with aliases and alternative names</p></div>
              <div className="inspection-controls inspection-controls-knowledge">
                <input type="search" className="inspection-search" placeholder="Search RxNav..." value={rxnavSearchInput} onChange={(event) => setRxnavSearchInput(event.target.value)} aria-label="Search RxNav data" />
                <button type="button" className="btn btn-primary inspection-mini-btn" onClick={() => {
                  if (rxnavJob.running && rxnavJob.jobId) {
                    void cancelInspectionRxNavUpdateJob(rxnavJob.jobId).then(() => setRxnavJob((prev) => ({ ...prev, message: "Cancellation requested", error: null }))).catch((error) => setRxnavJob((prev) => ({ ...prev, error: error instanceof Error ? error.message : "Failed to request cancellation." })));
                    return;
                  }
                  void startInspectionRxNavUpdateJob().then((start) => {
                    const intervalMs = Math.max(250, Math.round(start.poll_interval * 1000));
                    setRxnavJob({ jobId: start.job_id, running: true, progress: 1, status: start.status, message: "Initializing RxNav update", error: null });
                    stopRxnavPolling();
                    void pollRxnav(start.job_id, intervalMs);
                  }).catch((error) => setRxnavJob((prev) => ({ ...prev, running: false, error: error instanceof Error ? error.message : "Failed to start RxNav update." })));
                }}>{rxnavJob.running ? "Stop" : "Update"}</button>
              </div>
            </div>
            {(rxnavJob.running || rxnavJob.message || rxnavJob.error) && (
              <div className="inspection-job-panel">
                <div className="inspection-job-bar-track"><div className="inspection-job-bar-fill" style={{ width: `${Math.max(0, Math.min(100, rxnavJob.progress))}%` }} /></div>
                <p className="inspection-job-message">{rxnavJob.error || rxnavJob.message || "Updating RxNav..."}</p>
              </div>
            )}
            <div className="inspection-scroll-frame inspection-scroll-frame-compact">
              <table className="inspection-table inspection-table-dense">
                <thead><tr><th>Drug</th><th>Last update</th><th aria-label="Actions" /></tr></thead>
                <tbody>
                  {rxnavItems.map((row) => (
                    <tr key={row.drug_id}>
                      <td>{row.drug_name}</td>
                      <td>{row.last_update || "N/A"}</td>
                      <td className="inspection-actions-cell">
                        <button type="button" className="inspection-icon-button is-primary" onClick={() => { setAliasLoading(true); setAliasError(null); setAliasData(null); void fetchInspectionRxNavAliases(row.drug_id).then((payload) => setAliasData(payload)).catch((error) => setAliasError(error instanceof Error ? error.message : "Failed to load aliases.")).finally(() => setAliasLoading(false)); }} aria-label={`View aliases for ${row.drug_name}`} title={`View aliases for ${row.drug_name}`}><ViewIcon /></button>
                        <button type="button" className="inspection-icon-button is-danger" onClick={() => { if (!globalThis.confirm("Delete this drug and unlink historical references?")) return; void deleteInspectionRxNavDrug(row.drug_id).then(() => { if (rxnavItems.length === 1 && rxnavOffset > 0) setRxnavOffset(Math.max(0, rxnavOffset - PAGE_LIMIT)); else void loadRxnav(); }).catch((error) => setRxnavError(error instanceof Error ? error.message : "Failed to delete RxNav entry.")); }} aria-label={`Delete ${row.drug_name}`} title={`Delete ${row.drug_name}`}><DeleteIcon /></button>
                        <button type="button" className="inspection-icon-button" disabled aria-label="Modify RxNav entry (not implemented)" title="Modify RxNav entry (not implemented)"><ModifyIcon /></button>
                      </td>
                    </tr>
                  ))}
                  {!rxnavLoading && rxnavItems.length === 0 && <tr><td colSpan={3} className="inspection-empty-row">No RxNav rows found.</td></tr>}
                </tbody>
              </table>
              {rxnavLoading && <p className="inspection-loading-note">Loading RxNav data...</p>}
              {rxnavError && <p className="inspection-error-text">{rxnavError}</p>}
            </div>
            {renderPager(rxnavTotal, rxnavOffset, setRxnavOffset)}
          </div>

          <div className="inspection-widget">
            <div className="inspection-widget-header">
              <div><h2>LiverTox Data</h2><p>Monograph excerpts and update metadata</p></div>
              <div className="inspection-controls inspection-controls-knowledge">
                <input type="search" className="inspection-search" placeholder="Search LiverTox..." value={livertoxSearchInput} onChange={(event) => setLivertoxSearchInput(event.target.value)} aria-label="Search LiverTox data" />
                <button type="button" className="btn btn-primary inspection-mini-btn" onClick={() => {
                  if (livertoxJob.running && livertoxJob.jobId) {
                    void cancelInspectionLiverToxUpdateJob(livertoxJob.jobId).then(() => setLivertoxJob((prev) => ({ ...prev, message: "Cancellation requested", error: null }))).catch((error) => setLivertoxJob((prev) => ({ ...prev, error: error instanceof Error ? error.message : "Failed to request cancellation." })));
                    return;
                  }
                  void startInspectionLiverToxUpdateJob().then((start) => {
                    const intervalMs = Math.max(250, Math.round(start.poll_interval * 1000));
                    setLivertoxJob({ jobId: start.job_id, running: true, progress: 1, status: start.status, message: "Initializing LiverTox update", error: null });
                    stopLivertoxPolling();
                    void pollLivertox(start.job_id, intervalMs);
                  }).catch((error) => setLivertoxJob((prev) => ({ ...prev, running: false, error: error instanceof Error ? error.message : "Failed to start LiverTox update." })));
                }}>{livertoxJob.running ? "Stop" : "Update"}</button>
              </div>
            </div>
            {(livertoxJob.running || livertoxJob.message || livertoxJob.error) && (
              <div className="inspection-job-panel">
                <div className="inspection-job-bar-track"><div className="inspection-job-bar-fill" style={{ width: `${Math.max(0, Math.min(100, livertoxJob.progress))}%` }} /></div>
                <p className="inspection-job-message">{livertoxJob.error || livertoxJob.message || "Updating LiverTox..."}</p>
              </div>
            )}
            <div className="inspection-scroll-frame inspection-scroll-frame-compact">
              <table className="inspection-table inspection-table-dense">
                <thead><tr><th>Drug</th><th>Last update</th><th aria-label="Actions" /></tr></thead>
                <tbody>
                  {livertoxItems.map((row) => (
                    <tr key={row.drug_id}>
                      <td>{row.drug_name}</td>
                      <td>{row.last_update || "N/A"}</td>
                      <td className="inspection-actions-cell">
                        <button
                          type="button"
                          className="inspection-icon-button is-primary"
                          onClick={() => {
                            setExcerptLoading(true);
                            setExcerptError(null);
                            setExcerptData(null);
                            void fetchInspectionLiverToxExcerpt(row.drug_id)
                              .then((payload) => setExcerptData(payload))
                              .catch((error) => setExcerptError(resolveExcerptFallbackMessage(error)))
                              .finally(() => setExcerptLoading(false));
                          }}
                          aria-label={`View excerpt for ${row.drug_name}`}
                          title={`View excerpt for ${row.drug_name}`}
                        >
                          <ViewIcon />
                        </button>
                        <button type="button" className="inspection-icon-button is-danger" onClick={() => { if (!globalThis.confirm("Delete this LiverTox entry and unlink historical references?")) return; void deleteInspectionLiverToxDrug(row.drug_id).then(() => { if (livertoxItems.length === 1 && livertoxOffset > 0) setLivertoxOffset(Math.max(0, livertoxOffset - PAGE_LIMIT)); else void loadLivertox(); }).catch((error) => setLivertoxError(error instanceof Error ? error.message : "Failed to delete LiverTox entry.")); }} aria-label={`Delete ${row.drug_name}`} title={`Delete ${row.drug_name}`}><DeleteIcon /></button>
                        <button type="button" className="inspection-icon-button" disabled aria-label="Modify LiverTox entry (not implemented)" title="Modify LiverTox entry (not implemented)"><ModifyIcon /></button>
                      </td>
                    </tr>
                  ))}
                  {!livertoxLoading && livertoxItems.length === 0 && <tr><td colSpan={3} className="inspection-empty-row">No LiverTox rows found.</td></tr>}
                </tbody>
              </table>
              {livertoxLoading && <p className="inspection-loading-note">Loading LiverTox data...</p>}
              {livertoxError && <p className="inspection-error-text">{livertoxError}</p>}
            </div>
            {renderPager(livertoxTotal, livertoxOffset, setLivertoxOffset)}
          </div>
        </div>
      </section>

      {(aliasData || aliasLoading || aliasError) && (
        <div className="modal-overlay" onClick={closeAliasModal}>
          <dialog className="modal-container inspection-modal" open onClick={(event) => event.stopPropagation()} aria-modal="true" aria-label="RxNav aliases modal">
            <div className="modal-header">
              <div className="modal-header-content"><h2 className="modal-title">RxNav Aliases</h2><p className="modal-subtitle">{aliasData?.drug_name || "Loading aliases..."}</p></div>
              <button type="button" className="modal-close" onClick={closeAliasModal} aria-label="Close aliases modal">X</button>
            </div>
            <div className="modal-body">
              {aliasLoading && <p className="inspection-loading-note">Loading aliases...</p>}
              {!aliasLoading && aliasError && <p className="inspection-error-text">{aliasError}</p>}
              {!aliasLoading && !aliasError && aliasData && aliasData.groups.map((group) => (
                <section key={group.source} className="inspection-alias-group">
                  <h3>{group.source}</h3>
                  <ul>{group.aliases.map((entry) => <li key={`${group.source}-${entry.alias}-${entry.alias_kind}`}><span className="inspection-alias-label">{entry.alias}</span><span className="inspection-alias-kind">{entry.alias_kind}</span></li>)}</ul>
                </section>
              ))}
            </div>
          </dialog>
        </div>
      )}

      {(excerptData || excerptLoading || excerptError) && (
        <div className="modal-overlay" onClick={closeExcerptModal}>
          <dialog className="modal-container inspection-modal inspection-modal-large" open onClick={(event) => event.stopPropagation()} aria-modal="true" aria-label="LiverTox excerpt modal">
            <div className="modal-header">
              <div className="modal-header-content"><h2 className="modal-title">LiverTox Excerpt</h2><p className="modal-subtitle">{excerptData?.drug_name || (excerptError ? "Excerpt unavailable" : "Loading excerpt...")}</p></div>
              <button type="button" className="modal-close" onClick={closeExcerptModal} aria-label="Close excerpt modal">X</button>
            </div>
            <div className="modal-body inspection-excerpt-body">
              {excerptLoading && <p className="inspection-loading-note">Loading excerpt...</p>}
              {!excerptLoading && excerptError && <p className="inspection-empty-note">{excerptError}</p>}
              {!excerptLoading && !excerptError && excerptData && <pre className="inspection-excerpt-text">{excerptData.excerpt}</pre>}
            </div>
          </dialog>
        </div>
      )}
    </main>
  );
}
