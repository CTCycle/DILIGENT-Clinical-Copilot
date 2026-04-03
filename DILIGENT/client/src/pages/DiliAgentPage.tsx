import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { BooleanToggle } from "../components/BooleanToggle";
import {
    DEFAULT_FORM_STATE,
    REPORT_EXPORT_FILENAME,
} from "../constants";
import { useAppState } from "../context/AppStateContext";
import { useObjectUrlLifecycle } from "../hooks/useObjectUrlLifecycle";
import {
    cancelClinicalJob,
    pollClinicalJobStatus,
    startClinicalJob,
} from "../services/api";
import { JobStatus, JobStatusResponse } from "../types";
import {
    buildClinicalPayload,
    createDownloadUrl,
    normalizeVisitDateInput,
} from "../utils";

const todayIso = new Date().toISOString().slice(0, 10);
const DEFAULT_POLL_INTERVAL_MS = 1000;

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------
const CopyIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
);

const ExpandIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="15 3 21 3 21 9" />
        <polyline points="9 21 3 21 3 15" />
        <line x1="21" y1="3" x2="14" y2="10" />
        <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
);

const DownloadIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
);

function isTerminalJobStatus(status: JobStatus | null): boolean {
    return status === "completed" || status === "failed" || status === "cancelled";
}

// ---------------------------------------------------------------------------
// DiliAgentPage
// ---------------------------------------------------------------------------
export function DiliAgentPage(): React.JSX.Element {
    const { state, updateDiliAgent } = useAppState();
    const {
        settings,
        form,
        message,
        exportUrl,
        jobId,
        jobProgress,
        jobStatus,
        jobStageMessage,
        isRunning,
        isExpanded,
    } = state.diliAgent;

    const pollerRef = useRef<{ stop: () => void } | null>(null);
    const [isCancelling, setIsCancelling] = useState(false);
    const { revokeObjectUrl } = useObjectUrlLifecycle(exportUrl);

    // Cleanup poller on unmount
    useEffect(() => {
        return () => {
            if (pollerRef.current) {
                pollerRef.current.stop();
                pollerRef.current = null;
            }
        };
    }, []);

    // ---------------------------------------------------------------------------
    // Handlers
    // ---------------------------------------------------------------------------
    const resetOutputs = () => {
        revokeObjectUrl(exportUrl);
        updateDiliAgent({
            message: "",
            exportUrl: null,
            jobId: null,
            jobProgress: 0,
            jobStatus: null,
            jobStage: null,
            jobStageMessage: null,
        });
    };

    const handleFormChange = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => {
        updateDiliAgent({ form: { ...form, [key]: value } });
    };

    const handleVisitDateChange = (value: string) => {
        handleFormChange("visitDate", normalizeVisitDateInput(value));
    };

    const stopPoller = useCallback(() => {
        if (pollerRef.current) {
            pollerRef.current.stop();
            pollerRef.current = null;
        }
    }, []);

    const handlePollingError = useCallback(
        (pollError: string) => {
            stopPoller();
            setIsCancelling(false);
            updateDiliAgent({
                message: `[ERROR] ${pollError}`,
                exportUrl: null,
                jobStage: null,
                jobStageMessage: null,
                isRunning: false,
            });
        },
        [stopPoller, updateDiliAgent],
    );

    const handleJobStatusUpdate = useCallback(
        (status: JobStatusResponse) => {
            const terminalStatus = isTerminalJobStatus(status.status);
            const stage =
                status.result && typeof status.result.progress_stage === "string"
                    ? status.result.progress_stage
                    : null;
            const stageMessage =
                status.result && typeof status.result.progress_message === "string"
                    ? status.result.progress_message
                    : null;
            const resolvedProgress =
                typeof status.progress === "number" ? status.progress : 0;
            updateDiliAgent({
                jobProgress: terminalStatus ? 100 : resolvedProgress,
                jobStatus: status.status,
                jobStage: stage,
                jobStageMessage: stageMessage,
                isRunning: !terminalStatus,
            });

            if (!terminalStatus) {
                return;
            }

            stopPoller();
            setIsCancelling(false);

            if (status.status === "completed") {
                const report =
                    typeof status.result?.report === "string"
                        ? status.result.report
                        : "";
                const newExportUrl = report
                    ? createDownloadUrl(report, REPORT_EXPORT_FILENAME)
                    : null;
                updateDiliAgent({
                    message: report || "[INFO] Clinical analysis completed.",
                    exportUrl: newExportUrl,
                    jobStage: null,
                    jobStageMessage: null,
                });
            } else if (status.status === "failed") {
                const errorMessage = status.error
                    ? `[ERROR] ${status.error}`
                    : "[ERROR] Clinical analysis failed.";
                updateDiliAgent({
                    message: errorMessage,
                    exportUrl: null,
                    jobStage: null,
                    jobStageMessage: null,
                });
            } else if (status.status === "cancelled") {
                updateDiliAgent({
                    message: "[INFO] Clinical analysis cancelled.",
                    exportUrl: null,
                    jobStage: null,
                    jobStageMessage: null,
                });
            }
        },
        [stopPoller, updateDiliAgent],
    );

    const startPolling = useCallback(
        (jobIdToPoll: string, intervalMs: number) => {
            stopPoller();
            pollerRef.current = pollClinicalJobStatus(
                jobIdToPoll,
                intervalMs,
                handleJobStatusUpdate,
                handlePollingError,
            );
        },
        [handleJobStatusUpdate, handlePollingError, stopPoller],
    );

    const executeRunSession = async () => {
        setIsCancelling(false);
        updateDiliAgent({ isRunning: true });
        resetOutputs();
        stopPoller();
        try {
            const payload = buildClinicalPayload(form, settings);
            const startResult = await startClinicalJob(payload);
            updateDiliAgent({
                jobId: startResult.job_id,
                jobProgress: 0,
                jobStatus: startResult.status,
                jobStage: "session_initialization",
                jobStageMessage: "Initializing clinical session",
            });
            const intervalMs = startResult.poll_interval * 1000;
            startPolling(startResult.job_id, intervalMs);
        } catch (error) {
            const description =
                error instanceof Error ? error.message : "Unexpected error";
            updateDiliAgent({
                message: `[ERROR] ${description}`,
                exportUrl: null,
                jobStage: null,
                jobStageMessage: null,
                isRunning: false,
            });
        } finally {
            if (!pollerRef.current) {
                updateDiliAgent({ isRunning: false });
            }
        }
    };

    useEffect(() => {
        if (!isRunning || !jobId || isTerminalJobStatus(jobStatus) || pollerRef.current) {
            return;
        }
        startPolling(jobId, DEFAULT_POLL_INTERVAL_MS);
    }, [isRunning, jobId, jobStatus, startPolling]);

    const handleRunSession = async () => {
        const missingMessageByField: Array<[keyof typeof form, string]> = [
            ["anamnesis", "[ERROR] Provide the anamnesis."],
            ["visitDate", "[ERROR] Provide the visit date."],
            ["drugs", "[ERROR] Provide current drugs."],
            [
                "laboratoryAnalysis",
                "[ERROR] Provide laboratory data sufficient to determine hepatotoxicity pattern, ideally dated ALT or AST, ALP, and bilirubin.",
            ],
        ];
        const firstMissing = missingMessageByField.find(
            ([field]) => !String(form[field]).trim(),
        );
        if (firstMissing) {
            resetOutputs();
            updateDiliAgent({
                isRunning: false,
                message: firstMissing[1],
                exportUrl: null,
                jobStage: null,
                jobStageMessage: null,
            });
            return;
        }
        await executeRunSession();
    };

    const handleStopSession = async () => {
        if (!jobId) {
            return;
        }
        setIsCancelling(true);
        try {
            await cancelClinicalJob(jobId);
            updateDiliAgent({
                message: "[INFO] Cancellation requested. Waiting for worker shutdown...",
            });
        } catch (error) {
            const description =
                error instanceof Error ? error.message : "Failed to request cancellation.";
            updateDiliAgent({
                message: `[ERROR] ${description}`,
            });
        } finally {
            setIsCancelling(false);
        }
    };

    const handleClear = () => {
        revokeObjectUrl(exportUrl);
        updateDiliAgent({
            form: { ...DEFAULT_FORM_STATE },
            message: "",
            exportUrl: null,
            jobId: null,
            jobProgress: 0,
            jobStatus: null,
            jobStage: null,
            jobStageMessage: null,
        });
    };

    const handleDownload = () => {
        if (!exportUrl) return;
        const anchor = document.createElement("a");
        anchor.href = exportUrl;
        anchor.download = REPORT_EXPORT_FILENAME;
        anchor.click();
    };

    const handleCopyReport = async () => {
        if (message) {
            try {
                await navigator.clipboard.writeText(message);
            } catch (err) {
                console.error("Failed to copy:", err);
            }
        }
    };

    const handleToggleExpand = () => {
        updateDiliAgent({ isExpanded: !isExpanded });
    };

    const recordedDateLabel = form.visitDate
        ? new Date(`${form.visitDate}T00:00:00`).toLocaleDateString(undefined, {
            year: "numeric",
            month: "long",
            day: "numeric",
        })
        : "Not set";
    const patientNameLabel = form.patientName.trim() || "Unnamed patient";

    // ---------------------------------------------------------------------------
    // Render
    // ---------------------------------------------------------------------------
    const spinner = (
        <div className="session-spinner" role="status" aria-live="polite">
            <div className="spinner-wheel" />
            <p className="spinner-label">
                {`${jobStageMessage || "Generating report"}${
                    jobProgress > 0 ? `... ${Math.round(jobProgress)}%` : "..."
                }`}
            </p>
        </div>
    );

    const reportContent = (() => {
        if (isRunning && !isTerminalJobStatus(jobStatus)) {
            return spinner;
        }
        if (message) {
            return (
                <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown">
                    {message}
                </ReactMarkdown>
            );
        }
        return (
            <div className="report-placeholder">
                No report generated yet. Run analysis to see results.
            </div>
        );
    })();
    return (
        <main className="page-container stitch-dili-page">
            <header className="stitch-dili-heading">
                <h1>DILI Clinical Assessment</h1>
                <p>Patient Intake and Context</p>
            </header>

            <div className="stitch-dili-grid">
                <div className="stitch-dili-main">
                    <section className="stitch-dili-card">
                        <div className="stitch-dili-card-title">
                            <h2>Clinical Inputs</h2>
                        </div>
                        <div className="stitch-dili-input-stack">
                            <div className="stitch-dili-input-group">
                                <h3>Anamnesis</h3>
                                <textarea
                                    id="anamnesis"
                                    placeholder="Describe patient history, symptoms onset, and clinical observations..."
                                    value={form.anamnesis}
                                    onChange={(e) => handleFormChange("anamnesis", e.target.value)}
                                />
                            </div>

                            <div className="stitch-dili-input-group">
                                <h3>Current Drugs</h3>
                                <textarea
                                    id="drugs"
                                    placeholder="List medications, dosages, frequency, and duration of use..."
                                    value={form.drugs}
                                    onChange={(e) => handleFormChange("drugs", e.target.value)}
                                />
                            </div>

                            <div className="stitch-dili-input-group">
                                <h3>Laboratory Analysis</h3>
                                <textarea
                                    id="laboratory-analysis"
                                    placeholder="Input raw lab results (ALT, AST, ALP, Bilirubin, etc.)..."
                                    value={form.laboratoryAnalysis}
                                    onChange={(e) => handleFormChange("laboratoryAnalysis", e.target.value)}
                                />
                            </div>
                        </div>
                    </section>
                </div>

                <aside className="stitch-dili-sidebar">
                    <section className="stitch-dili-card stitch-dili-patient">
                        <div className="stitch-dili-patient-top">
                            <div className="stitch-dili-avatar" aria-hidden="true">
                                {patientNameLabel.charAt(0).toUpperCase()}
                            </div>
                            <div>
                                <h3>{patientNameLabel}</h3>
                                <p>Recorded: {recordedDateLabel}</p>
                            </div>
                        </div>

                        <div className="stitch-dili-inline-fields">
                            <div className="field">
                                <label className="field-label" htmlFor="patient-name">Patient Name</label>
                                <input
                                    id="patient-name"
                                    type="text"
                                    placeholder="e.g., Marco Rossi"
                                    value={form.patientName}
                                    onChange={(e) => handleFormChange("patientName", e.target.value)}
                                />
                            </div>
                            <div className="field">
                                <label className="field-label" htmlFor="visit-date">Visit Date</label>
                                <input
                                    id="visit-date"
                                    type="date"
                                    max={todayIso}
                                    value={form.visitDate}
                                    onChange={(e) => handleVisitDateChange(e.target.value)}
                                />
                            </div>
                        </div>

                        <div className="stitch-dili-controls">
                            <p>Retrieval Controls</p>
                            <BooleanToggle
                                id="use-rag"
                                label="Internal RAG"
                                checked={form.useRag}
                                onChange={(checked) => handleFormChange("useRag", checked)}
                            />
                            <BooleanToggle
                                id="use-web-search"
                                label="Web Search"
                                checked={form.useWebSearch}
                                onChange={(checked) => handleFormChange("useWebSearch", checked)}
                            />
                        </div>

                        <div className="stitch-dili-actions">
                            <button
                                className="btn btn-primary"
                                type="button"
                                disabled={isCancelling}
                                onClick={() => { void (isRunning ? handleStopSession() : handleRunSession()); }}
                            >
                                {isRunning ? (isCancelling ? "Stopping..." : "Stop analysis") : "Run DILI analysis"}
                            </button>
                            <button className="stitch-dili-clear" type="button" onClick={handleClear} disabled={isRunning}>
                                Clear all
                            </button>
                        </div>
                    </section>

                    <div className="stitch-dili-tip">
                        "Ensure laboratory results include units for accurate drug-induced liver injury risk profiling."
                    </div>
                </aside>
            </div>

            <section className="report-section stitch-dili-report">
                <div className="report-shell">
                    <div className="report-header">
                        <p className="report-eyebrow">Report Output</p>
                    </div>

                    <div className="report-toolbar">
                        <button
                            className="toolbar-btn"
                            type="button"
                            onClick={handleCopyReport}
                            disabled={!message}
                            title="Copy to clipboard"
                        >
                            <CopyIcon />
                            <span>Copy</span>
                        </button>
                        <button
                            className="toolbar-btn"
                            type="button"
                            onClick={handleToggleExpand}
                            disabled={!message}
                            title={isExpanded ? "Collapse" : "Expand"}
                        >
                            <ExpandIcon />
                            <span>{isExpanded ? "Collapse" : "Expand"}</span>
                        </button>
                        <button
                            className="toolbar-btn"
                            type="button"
                            onClick={handleDownload}
                            disabled={!exportUrl}
                            title="Download markdown file"
                        >
                            <DownloadIcon />
                            <span>Download markdown</span>
                        </button>
                    </div>

                    <div className={`report-content ${isExpanded ? "is-expanded" : ""}`}>
                        {reportContent}
                    </div>
                </div>
            </section>
        </main>
    );
}

