import React, { Dispatch, SetStateAction, useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  cancelInspectionLiverToxUpdateJob,
  cancelInspectionRagUpdateJob,
  cancelInspectionRxNavUpdateJob,
  deleteInspectionLiverToxDrug,
  deleteInspectionRxNavDrug,
  deleteInspectionSession,
  fetchInspectionLiverToxUpdateConfig,
  fetchInspectionLiverToxCatalog,
  fetchInspectionLiverToxExcerpt,
  fetchInspectionLiverToxUpdateJobStatus,
  fetchInspectionRagDocuments,
  fetchInspectionRagUpdateConfig,
  fetchInspectionRagUpdateJobStatus,
  fetchInspectionRagVectorStore,
  fetchInspectionRxNavUpdateConfig,
  fetchInspectionRxNavAliases,
  fetchInspectionRxNavCatalog,
  fetchInspectionRxNavUpdateJobStatus,
  fetchInspectionSessionReport,
  fetchInspectionSessions,
  startInspectionRagUpdateJob,
  startInspectionLiverToxUpdateJob,
  startInspectionRxNavUpdateJob,
} from "../services/api";
import { InspectionIconActionButton } from "../components/InspectionIconActionButton";
import { InspectionUpdateWizard } from "../components/InspectionUpdateWizard";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { useInspectionUpdateJob } from "../hooks/useInspectionUpdateJob";
import {
  InspectionDateFilterMode,
  InspectionDrugAliasesResponse,
  InspectionLiverToxExcerptResponse,
  InspectionLiverToxItem,
  InspectionRagDocumentRow,
  InspectionRagVectorStoreSummary,
  InspectionRagOverrideRequest,
  InspectionLiverToxOverrideRequest,
  InspectionRxNavOverrideRequest,
  InspectionRxNavItem,
  InspectionSessionItem,
  InspectionSessionStatus,
} from "../types";

const PAGE_LIMIT = 10;
const SEARCH_DEBOUNCE_MS = 250;
type InspectionViewId = "sessions" | "rxnav" | "livertox" | "rag";

const INSPECTION_VIEW_CONFIG: Record<InspectionViewId, { label: string; description: string }> = {
  sessions: {
    label: "Sessions",
    description: "Review recorded DILI sessions, patient metadata, runtime status, and generated reports.",
  },
  rxnav: {
    label: "Drug Catalog",
    description: "Inspect the canonical RxNav catalog, aliases, and update progress for indexed drugs.",
  },
  livertox: {
    label: "LiverTox data",
    description: "Browse LiverTox monograph excerpts and recency data used by the clinical copilot.",
  },
  rag: {
    label: "RAG",
    description: "Inspect RAG source documents, LanceDB status, and run embeddings updates.",
  },
};
const INSPECTION_VIEW_ORDER: readonly InspectionViewId[] = ["sessions", "rxnav", "livertox", "rag"];

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

function sessionFilterLabel(value: "all" | "successful" | "failed"): string {
  if (value === "all") return "All";
  if (value === "successful") return "Successful";
  return "Failed";
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

function SessionsNavIcon(): React.JSX.Element {
  return (
    <svg className="inspection-nav-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5h16v3H4V5Zm0 5h16v3H4v-3Zm0 5h10v3H4v-3Z" />
    </svg>
  );
}

function RxNavNavIcon(): React.JSX.Element {
  return (
    <svg className="inspection-nav-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 6h16v2H4V6Zm0 5h10v2H4v-2Zm0 5h8v2H4v-2Zm13.4-1.4 2.6 2.6-1.4 1.4-2.6-2.6V15h-1.4v-2h3v1.6Z" />
    </svg>
  );
}

function LiverToxNavIcon(): React.JSX.Element {
  return (
    <svg className="inspection-nav-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3c4.42 0 8 3.13 8 7 0 3.4-2.75 6.24-6.4 6.88-.27 2.16-1.61 3.72-3.6 4.12v-4.14C6.6 16.45 4 13.53 4 10c0-3.87 3.58-7 8-7Zm0 2c-3.31 0-6 2.24-6 5s2.69 5 6 5 6-2.24 6-5-2.69-5-6-5Z" />
    </svg>
  );
}

function RagNavIcon(): React.JSX.Element {
  return (
    <svg className="inspection-nav-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 4h16v4H4V4Zm0 6h16v4H4v-4Zm0 6h9v4H4v-4Zm12 0h4v4h-4v-4Z" />
    </svg>
  );
}

function CloseIcon(): React.JSX.Element {
  return (
    <svg className="inspection-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M18.3 5.7a1 1 0 0 0-1.4 0L12 10.6 7.1 5.7a1 1 0 1 0-1.4 1.4l4.9 4.9-4.9 4.9a1 1 0 1 0 1.4 1.4l4.9-4.9 4.9 4.9a1 1 0 0 0 1.4-1.4L13.4 12l4.9-4.9a1 1 0 0 0 0-1.4Z" />
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
  const [ragDocuments, setRagDocuments] = useState<InspectionRagDocumentRow[]>([]);
  const [ragVectorStore, setRagVectorStore] = useState<InspectionRagVectorStoreSummary | null>(null);
  const [ragTotal, setRagTotal] = useState(0);
  const [ragOffset, setRagOffset] = useState(0);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState<string | null>(null);
  const [ragSearchInput, setRagSearchInput] = useState("");
  const ragSearch = useDebouncedValue(ragSearchInput, SEARCH_DEBOUNCE_MS);

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
  const [activeView, setActiveView] = useState<InspectionViewId>("sessions");
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

  const loadRag = useCallback(async () => {
    setRagLoading(true);
    setRagError(null);
    try {
      const [documentsPayload, vectorPayload] = await Promise.all([
        fetchInspectionRagDocuments(),
        fetchInspectionRagVectorStore(),
      ]);
      const normalizedSearch = ragSearch.trim().toLowerCase();
      const filtered = documentsPayload.items.filter((item) => {
        if (!normalizedSearch) {
          return true;
        }
        const haystack = `${item.file_name} ${item.path} ${item.extension}`.toLowerCase();
        return haystack.includes(normalizedSearch);
      });
      const paged = filtered.slice(ragOffset, ragOffset + PAGE_LIMIT);
      setRagDocuments(paged);
      setRagTotal(filtered.length);
      setRagVectorStore(vectorPayload);
    } catch (error) {
      setRagDocuments([]);
      setRagTotal(0);
      setRagVectorStore(null);
      setRagError(error instanceof Error ? error.message : "Failed to load RAG inspection data.");
    } finally {
      setRagLoading(false);
    }
  }, [ragOffset, ragSearch]);

  const rxnavUpdateJob = useInspectionUpdateJob({
    startJob: (payload) =>
      startInspectionRxNavUpdateJob((payload || {}) as InspectionRxNavOverrideRequest),
    fetchStatus: fetchInspectionRxNavUpdateJobStatus,
    cancelJob: cancelInspectionRxNavUpdateJob,
    onCompleted: loadRxnav,
    startMessage: "Initializing RxNav update",
    startErrorMessage: "Failed to start RxNav update.",
    cancelErrorMessage: "Failed to request cancellation.",
    pollErrorMessage: "RxNav polling failed.",
  });

  const livertoxUpdateJob = useInspectionUpdateJob({
    startJob: (payload) =>
      startInspectionLiverToxUpdateJob((payload || {}) as InspectionLiverToxOverrideRequest),
    fetchStatus: fetchInspectionLiverToxUpdateJobStatus,
    cancelJob: cancelInspectionLiverToxUpdateJob,
    onCompleted: loadLivertox,
    startMessage: "Initializing LiverTox update",
    startErrorMessage: "Failed to start LiverTox update.",
    cancelErrorMessage: "Failed to request cancellation.",
    pollErrorMessage: "LiverTox polling failed.",
  });

  const rxnavJob = rxnavUpdateJob.state;
  const livertoxJob = livertoxUpdateJob.state;
  const ragUpdateJob = useInspectionUpdateJob({
    startJob: (payload) =>
      startInspectionRagUpdateJob((payload || {}) as InspectionRagOverrideRequest),
    fetchStatus: fetchInspectionRagUpdateJobStatus,
    cancelJob: cancelInspectionRagUpdateJob,
    onCompleted: loadRag,
    startMessage: "Initializing RAG embeddings update",
    startErrorMessage: "Failed to start RAG update.",
    cancelErrorMessage: "Failed to request cancellation.",
    pollErrorMessage: "RAG polling failed.",
  });
  const ragJob = ragUpdateJob.state;
  const activeViewConfig = INSPECTION_VIEW_CONFIG[activeView];
  const activeTabId = `inspection-tab-${activeView}`;

  const focusInspectionTab = (view: InspectionViewId) => {
    globalThis.requestAnimationFrame(() => {
      const element = globalThis.document.getElementById(`inspection-tab-${view}`);
      if (element instanceof HTMLButtonElement) {
        element.focus();
      }
    });
  };

  const handleTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, view: InspectionViewId) => {
    const currentIndex = INSPECTION_VIEW_ORDER.indexOf(view);
    if (currentIndex < 0) {
      return;
    }
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      const nextIndex = (currentIndex + 1) % INSPECTION_VIEW_ORDER.length;
      const nextView = INSPECTION_VIEW_ORDER[nextIndex];
      setActiveView(nextView);
      focusInspectionTab(nextView);
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      const previousIndex = (currentIndex - 1 + INSPECTION_VIEW_ORDER.length) % INSPECTION_VIEW_ORDER.length;
      const previousView = INSPECTION_VIEW_ORDER[previousIndex];
      setActiveView(previousView);
      focusInspectionTab(previousView);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      const firstView = INSPECTION_VIEW_ORDER[0];
      setActiveView(firstView);
      focusInspectionTab(firstView);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      const lastView = INSPECTION_VIEW_ORDER[INSPECTION_VIEW_ORDER.length - 1];
      setActiveView(lastView);
      focusInspectionTab(lastView);
    }
  };

  useEffect(() => setSessionOffset(0), [sessionSearch, sessionStatusFilter, sessionDateMode, sessionDate]);
  useEffect(() => setRxnavOffset(0), [rxnavSearch]);
  useEffect(() => setLivertoxOffset(0), [livertoxSearch]);
  useEffect(() => setRagOffset(0), [ragSearch]);
  useEffect(() => { void loadSessions(); }, [loadSessions]);
  useEffect(() => { void loadRxnav(); }, [loadRxnav]);
  useEffect(() => { void loadLivertox(); }, [loadLivertox]);
  useEffect(() => { void loadRag(); }, [loadRag]);

  const sessionStatusClass = useMemo(() => (status: InspectionSessionStatus) => (status === "failed" ? "is-failed" : "is-successful"), []);

  const renderPager = (
    total: number,
    offset: number,
    setOffset: Dispatch<SetStateAction<number>>,
  ) => (
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
      <main className="page-container inspection-page inspection-page-report">
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

  const renderSessionsView = (): React.JSX.Element => (
    <section className="inspection-view-content">
      <div className="inspection-widget">
        <div className="inspection-widget-header inspection-widget-header-controls-only">
          <div className="inspection-controls inspection-controls-sessions">
            <div className="inspection-controls-sessions-main">
              <input type="search" className="inspection-search" placeholder="Search sessions..." value={sessionSearchInput} onChange={(event) => setSessionSearchInput(event.target.value)} aria-label="Search sessions" />
              <div className="inspection-toggle-group">
                {(["all", "successful", "failed"] as const).map((value) => (
                  <button key={value} type="button" className={`inspection-toggle-pill ${sessionStatusFilter === value ? "is-active" : ""}`} onClick={() => setSessionStatusFilter(value)}>
                    {sessionFilterLabel(value)}
                  </button>
                ))}
              </div>
              <select className="inspection-select" value={sessionDateMode} onChange={(event) => setSessionDateMode(event.target.value as InspectionDateFilterMode | "none")} aria-label="Date filter mode">
                <option value="none">Date filter</option>
                <option value="before">Before</option>
                <option value="after">After</option>
                <option value="exact">Exact</option>
              </select>
            </div>
            <div className="inspection-controls-sessions-date">
              <input type="date" className="inspection-date" value={sessionDate} onChange={(event) => setSessionDate(event.target.value)} disabled={sessionDateMode === "none"} aria-label="Session date filter" />
            </div>
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
                    <InspectionIconActionButton variant="primary" onClick={() => { setReportLoading(true); setReportSession(row); setReportError(null); setReportContent(""); void fetchInspectionSessionReport(row.session_id).then((payload) => setReportContent(payload.report)).catch((error) => setReportError(error instanceof Error ? error.message : "Failed to load report.")).finally(() => setReportLoading(false)); }} ariaLabel={`View report for session ${row.session_id}`} title={`View report for session ${row.session_id}`}><ViewIcon /></InspectionIconActionButton>
                    <InspectionIconActionButton variant="danger" onClick={() => { if (!globalThis.confirm("Delete this recorded session and report data?")) return; void deleteInspectionSession(row.session_id).then(() => { if (sessionItems.length === 1 && sessionOffset > 0) { setSessionOffset(Math.max(0, sessionOffset - PAGE_LIMIT)); } else { void loadSessions(); } }).catch((error) => setSessionError(error instanceof Error ? error.message : "Failed to delete session.")); }} ariaLabel={`Delete session ${row.session_id}`} title={`Delete session ${row.session_id}`}><DeleteIcon /></InspectionIconActionButton>
                    <InspectionIconActionButton disabled ariaLabel="Modify session (not implemented)" title="Modify session (not implemented)"><ModifyIcon /></InspectionIconActionButton>
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
    </section>
  );

  const renderRxnavView = (): React.JSX.Element => (
    <section className="inspection-view-content">
      <div className="inspection-widget">
        <div className="inspection-widget-header inspection-widget-header-controls-only">
          <div className="inspection-controls inspection-controls-knowledge">
            <input type="search" className="inspection-search" placeholder="Search RxNav..." value={rxnavSearchInput} onChange={(event) => setRxnavSearchInput(event.target.value)} aria-label="Search RxNav data" />
          </div>
        </div>
        <InspectionUpdateWizard
          targetLabel="RxNav catalog"
          fallbackMessage="Updating RxNav..."
          loadConfig={fetchInspectionRxNavUpdateConfig}
          startJob={rxnavUpdateJob.triggerUpdate}
          job={rxnavJob}
        />
        <div className="inspection-scroll-frame inspection-scroll-frame-compact">
          <table className="inspection-table inspection-table-dense">
            <thead><tr><th>Drug</th><th>Last update</th><th aria-label="Actions" /></tr></thead>
            <tbody>
              {rxnavItems.map((row) => (
                <tr key={row.drug_id}>
                  <td>{row.drug_name}</td>
                  <td>{row.last_update || "N/A"}</td>
                  <td className="inspection-actions-cell">
                    <InspectionIconActionButton variant="primary" onClick={() => { setAliasLoading(true); setAliasError(null); setAliasData(null); void fetchInspectionRxNavAliases(row.drug_id).then((payload) => setAliasData(payload)).catch((error) => setAliasError(error instanceof Error ? error.message : "Failed to load aliases.")).finally(() => setAliasLoading(false)); }} ariaLabel={`View aliases for ${row.drug_name}`} title={`View aliases for ${row.drug_name}`}><ViewIcon /></InspectionIconActionButton>
                    <InspectionIconActionButton variant="danger" onClick={() => { if (!globalThis.confirm("Delete this drug and unlink historical references?")) return; void deleteInspectionRxNavDrug(row.drug_id).then(() => { if (rxnavItems.length === 1 && rxnavOffset > 0) { setRxnavOffset(Math.max(0, rxnavOffset - PAGE_LIMIT)); } else { void loadRxnav(); } }).catch((error) => setRxnavError(error instanceof Error ? error.message : "Failed to delete RxNav entry.")); }} ariaLabel={`Delete ${row.drug_name}`} title={`Delete ${row.drug_name}`}><DeleteIcon /></InspectionIconActionButton>
                    <InspectionIconActionButton disabled ariaLabel="Modify RxNav entry (not implemented)" title="Modify RxNav entry (not implemented)"><ModifyIcon /></InspectionIconActionButton>
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
    </section>
  );

  const renderLivertoxView = (): React.JSX.Element => (
    <section className="inspection-view-content">
      <div className="inspection-widget">
        <div className="inspection-widget-header inspection-widget-header-controls-only">
          <div className="inspection-controls inspection-controls-knowledge">
            <input type="search" className="inspection-search" placeholder="Search LiverTox..." value={livertoxSearchInput} onChange={(event) => setLivertoxSearchInput(event.target.value)} aria-label="Search LiverTox data" />
          </div>
        </div>
        <InspectionUpdateWizard
          targetLabel="LiverTox monographs"
          fallbackMessage="Updating LiverTox..."
          loadConfig={fetchInspectionLiverToxUpdateConfig}
          startJob={livertoxUpdateJob.triggerUpdate}
          job={livertoxJob}
        />
        <div className="inspection-scroll-frame inspection-scroll-frame-compact">
          <table className="inspection-table inspection-table-dense">
            <thead><tr><th>Drug</th><th>Last update</th><th aria-label="Actions" /></tr></thead>
            <tbody>
              {livertoxItems.map((row) => (
                <tr key={row.drug_id}>
                  <td>{row.drug_name}</td>
                  <td>{row.last_update || "N/A"}</td>
                  <td className="inspection-actions-cell">
                    <InspectionIconActionButton variant="primary" onClick={() => {
                      setExcerptLoading(true);
                      setExcerptError(null);
                      setExcerptData(null);
                      void fetchInspectionLiverToxExcerpt(row.drug_id)
                        .then((payload) => setExcerptData(payload))
                        .catch((error) => setExcerptError(resolveExcerptFallbackMessage(error)))
                        .finally(() => setExcerptLoading(false));
                    }} ariaLabel={`View excerpt for ${row.drug_name}`} title={`View excerpt for ${row.drug_name}`}><ViewIcon /></InspectionIconActionButton>
                    <InspectionIconActionButton variant="danger" onClick={() => { if (!globalThis.confirm("Delete this LiverTox entry and unlink historical references?")) return; void deleteInspectionLiverToxDrug(row.drug_id).then(() => { if (livertoxItems.length === 1 && livertoxOffset > 0) { setLivertoxOffset(Math.max(0, livertoxOffset - PAGE_LIMIT)); } else { void loadLivertox(); } }).catch((error) => setLivertoxError(error instanceof Error ? error.message : "Failed to delete LiverTox entry.")); }} ariaLabel={`Delete ${row.drug_name}`} title={`Delete ${row.drug_name}`}><DeleteIcon /></InspectionIconActionButton>
                    <InspectionIconActionButton disabled ariaLabel="Modify LiverTox entry (not implemented)" title="Modify LiverTox entry (not implemented)"><ModifyIcon /></InspectionIconActionButton>
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
    </section>
  );

  const renderRagView = (): React.JSX.Element => (
    <section className="inspection-view-content">
      <div className="inspection-widget">
        <div className="inspection-widget-header inspection-widget-header-controls-only">
          <div className="inspection-controls inspection-controls-knowledge">
            <input
              type="search"
              className="inspection-search"
              placeholder="Search RAG documents..."
              value={ragSearchInput}
              onChange={(event) => setRagSearchInput(event.target.value)}
              aria-label="Search RAG documents"
            />
          </div>
        </div>
        <div className="inspection-scroll-frame inspection-scroll-frame-compact">
          <table className="inspection-table inspection-table-dense">
            <thead>
              <tr>
                <th>LanceDB path</th>
                <th>Collection</th>
                <th>Embeddings</th>
                <th>Documents</th>
                <th>Dimension</th>
                <th>Index ready</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{ragVectorStore?.vector_db_path || "N/A"}</td>
                <td>{ragVectorStore?.collection_name || "N/A"}</td>
                <td>{ragVectorStore?.embedding_count ?? 0}</td>
                <td>{ragVectorStore?.distinct_document_count ?? 0}</td>
                <td>{ragVectorStore?.embedding_dimension ?? "N/A"}</td>
                <td>{ragVectorStore?.index_ready ? "Yes" : "No"}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <InspectionUpdateWizard
          targetLabel="RAG embeddings"
          fallbackMessage="Updating RAG embeddings..."
          loadConfig={fetchInspectionRagUpdateConfig}
          startJob={ragUpdateJob.triggerUpdate}
          job={ragJob}
        />
        <div className="inspection-scroll-frame inspection-scroll-frame-compact">
          <table className="inspection-table inspection-table-dense">
            <thead>
              <tr>
                <th>File</th>
                <th>Extension</th>
                <th>Size</th>
                <th>Last modified</th>
                <th>Supported</th>
              </tr>
            </thead>
            <tbody>
              {ragDocuments.map((row) => (
                <tr key={row.path}>
                  <td>{row.file_name}</td>
                  <td>{row.extension}</td>
                  <td>{row.file_size}</td>
                  <td>{formatDateTime(row.last_modified)}</td>
                  <td>{row.supported_for_ingestion ? "Yes" : "No"}</td>
                </tr>
              ))}
              {!ragLoading && ragDocuments.length === 0 && (
                <tr><td colSpan={5} className="inspection-empty-row">No RAG documents found.</td></tr>
              )}
            </tbody>
          </table>
          {ragLoading && <p className="inspection-loading-note">Loading RAG data...</p>}
          {ragError && <p className="inspection-error-text">{ragError}</p>}
        </div>
        {renderPager(ragTotal, ragOffset, setRagOffset)}
      </div>
    </section>
  );

  return (
    <main className="page-container inspection-page inspection-page-workbench">
      <header className="page-header"><p className="eyebrow">Data Inspection</p><h1>Session and Knowledge Catalog</h1><p className="lede">Review recorded DILI sessions alongside RxNav and LiverTox knowledge records.</p></header>
      <section className="inspection-layout">
        <aside className="inspection-toolbar" aria-label="Data inspection views">
          <p className="inspection-toolbar-eyebrow">Views</p>
          <div className="inspection-toolbar-list" role="tablist" aria-label="Inspection datasets">
            <button
              id="inspection-tab-sessions"
              type="button"
              className={`inspection-toolbar-item ${activeView === "sessions" ? "is-active" : ""}`}
              onClick={() => setActiveView("sessions")}
              onKeyDown={(event) => handleTabKeyDown(event, "sessions")}
              role="tab"
              aria-selected={activeView === "sessions"}
              aria-controls="inspection-active-view-panel"
              tabIndex={activeView === "sessions" ? 0 : -1}
            >
              <SessionsNavIcon />
              <span>{INSPECTION_VIEW_CONFIG.sessions.label}</span>
            </button>
            <button
              id="inspection-tab-rxnav"
              type="button"
              className={`inspection-toolbar-item ${activeView === "rxnav" ? "is-active" : ""}`}
              onClick={() => setActiveView("rxnav")}
              onKeyDown={(event) => handleTabKeyDown(event, "rxnav")}
              role="tab"
              aria-selected={activeView === "rxnav"}
              aria-controls="inspection-active-view-panel"
              tabIndex={activeView === "rxnav" ? 0 : -1}
            >
              <RxNavNavIcon />
              <span>{INSPECTION_VIEW_CONFIG.rxnav.label}</span>
            </button>
            <button
              id="inspection-tab-livertox"
              type="button"
              className={`inspection-toolbar-item ${activeView === "livertox" ? "is-active" : ""}`}
              onClick={() => setActiveView("livertox")}
              onKeyDown={(event) => handleTabKeyDown(event, "livertox")}
              role="tab"
              aria-selected={activeView === "livertox"}
              aria-controls="inspection-active-view-panel"
              tabIndex={activeView === "livertox" ? 0 : -1}
            >
              <LiverToxNavIcon />
              <span>{INSPECTION_VIEW_CONFIG.livertox.label}</span>
            </button>
            <button
              id="inspection-tab-rag"
              type="button"
              className={`inspection-toolbar-item ${activeView === "rag" ? "is-active" : ""}`}
              onClick={() => setActiveView("rag")}
              onKeyDown={(event) => handleTabKeyDown(event, "rag")}
              role="tab"
              aria-selected={activeView === "rag"}
              aria-controls="inspection-active-view-panel"
              tabIndex={activeView === "rag" ? 0 : -1}
            >
              <RagNavIcon />
              <span>{INSPECTION_VIEW_CONFIG.rag.label}</span>
            </button>
          </div>
        </aside>

        <section
          id="inspection-active-view-panel"
          className="inspection-active-view"
          role="tabpanel"
          aria-labelledby={activeTabId}
        >
          <div className="inspection-active-view-header">
            <h2>{activeViewConfig.label}</h2>
            <p>{activeViewConfig.description}</p>
          </div>
          <div className="inspection-view-stage">
            {activeView === "sessions" && renderSessionsView()}
            {activeView === "rxnav" && renderRxnavView()}
            {activeView === "livertox" && renderLivertoxView()}
            {activeView === "rag" && renderRagView()}
          </div>
        </section>
      </section>

      {(aliasData || aliasLoading || aliasError) && (
        <div className="modal-overlay">
          <dialog className="modal-container inspection-modal" open aria-modal="true" aria-label="RxNav aliases modal">
            <div className="modal-header">
              <div className="modal-header-content"><h2 className="modal-title">RxNav Aliases</h2><p className="modal-subtitle">{aliasData?.drug_name || "Loading aliases..."}</p></div>
              <button type="button" className="modal-close" onClick={closeAliasModal} aria-label="Close aliases modal"><CloseIcon /></button>
            </div>
            <div className="modal-body">
              {aliasLoading && <p className="inspection-loading-note">Loading aliases...</p>}
              {!aliasLoading && aliasError && <p className="inspection-error-text">{aliasError}</p>}
              {!aliasLoading && !aliasError && aliasData?.groups.map((group) => (
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
        <div className="modal-overlay">
          <dialog className="modal-container inspection-modal inspection-modal-large" open aria-modal="true" aria-label="LiverTox excerpt modal">
            <div className="modal-header">
              <div className="modal-header-content"><h2 className="modal-title">LiverTox Excerpt</h2><p className="modal-subtitle">{excerptData?.drug_name || (excerptError ? "Excerpt unavailable" : "Loading excerpt...")}</p></div>
              <button type="button" className="modal-close" onClick={closeExcerptModal} aria-label="Close excerpt modal"><CloseIcon /></button>
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
