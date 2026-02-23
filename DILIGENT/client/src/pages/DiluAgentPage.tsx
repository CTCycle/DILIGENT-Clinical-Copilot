import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ConfirmModal } from "../components/ConfirmModal";
import {
    DEFAULT_SETTINGS,
    REPORT_EXPORT_FILENAME,
} from "../constants";
import { useAppState } from "../context/AppStateContext";
import {
    pollClinicalJobStatus,
    startClinicalJob,
} from "../services/api";
import {
    buildClinicalPayload,
    createDownloadUrl,
    normalizeVisitDateInput,
} from "../utils";

const todayIso = new Date().toISOString().slice(0, 10);

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

function isTerminalJobStatus(status: string | null): boolean {
    return status === "completed" || status === "failed" || status === "cancelled";
}

// ---------------------------------------------------------------------------
// DiluAgentPage
// ---------------------------------------------------------------------------
export function DiluAgentPage(): React.JSX.Element {
    const { state, updateDiluAgent } = useAppState();
    const {
        settings,
        form,
        message,
        jsonPayload,
        exportUrl,
        jobProgress,
        jobStatus,
        jobStageMessage,
        isRunning,
        isExpanded,
    } = state.diluAgent;

    const pollerRef = useRef<{ stop: () => void } | null>(null);
    const [isMissingLabsModalOpen, setIsMissingLabsModalOpen] = useState(false);

    // Cleanup export URL on unmount
    useEffect(() => {
        return () => {
            if (exportUrl) {
                URL.revokeObjectURL(exportUrl);
            }
            if (pollerRef.current) {
                pollerRef.current.stop();
                pollerRef.current = null;
            }
        };
    }, [exportUrl]);

    // ---------------------------------------------------------------------------
    // Handlers
    // ---------------------------------------------------------------------------
    const resetOutputs = () => {
        if (exportUrl) {
            URL.revokeObjectURL(exportUrl);
        }
        updateDiluAgent({
            message: "",
            jsonPayload: null,
            exportUrl: null,
            jobId: null,
            jobProgress: 0,
            jobStatus: null,
            jobStage: null,
            jobStageMessage: null,
        });
    };

    const handleFormChange = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => {
        updateDiluAgent({ form: { ...form, [key]: value } });
    };

    const handleVisitDateChange = (value: string) => {
        handleFormChange("visitDate", normalizeVisitDateInput(value));
    };

    const executeRunSession = async (allowMissingLabs: boolean | null) => {
        updateDiluAgent({ isRunning: true });
        resetOutputs();
        if (pollerRef.current) {
            pollerRef.current.stop();
            pollerRef.current = null;
        }
        try {
            const payload = buildClinicalPayload(form, settings, allowMissingLabs);
            const startResult = await startClinicalJob(payload);
            updateDiluAgent({
                jobId: startResult.job_id,
                jobProgress: 0,
                jobStatus: startResult.status,
                jobStage: "session_initialization",
                jobStageMessage: "Initializing clinical session",
            });
            const intervalMs = startResult.poll_interval * 1000;

            pollerRef.current = pollClinicalJobStatus(
                startResult.job_id,
                intervalMs,
                (status) => {
                    const terminalStatus = isTerminalJobStatus(status.status);
                    const stage =
                        status.result && typeof status.result.progress_stage === "string"
                            ? status.result.progress_stage
                            : null;
                    const stageMessage =
                        status.result && typeof status.result.progress_message === "string"
                            ? status.result.progress_message
                            : null;
                    updateDiluAgent({
                        jobProgress: status.progress ?? 0,
                        jobStatus: status.status,
                        jobStage: stage,
                        jobStageMessage: stageMessage,
                        isRunning: !terminalStatus,
                    });

                    if (terminalStatus) {
                        if (pollerRef.current) {
                            pollerRef.current.stop();
                            pollerRef.current = null;
                        }
                    }

                    if (!terminalStatus) {
                        return;
                    }

                    if (status.status === "completed") {
                        const report =
                            typeof status.result?.report === "string"
                                ? status.result.report
                                : "";
                        const newExportUrl = report
                            ? createDownloadUrl(report, REPORT_EXPORT_FILENAME)
                            : null;
                        updateDiluAgent({
                            message: report || "[INFO] Clinical analysis completed.",
                            jsonPayload: status.result,
                            exportUrl: newExportUrl,
                            jobStage: null,
                            jobStageMessage: null,
                        });
                    } else if (status.status === "failed") {
                        const errorMessage = status.error
                            ? `[ERROR] ${status.error}`
                            : "[ERROR] Clinical analysis failed.";
                        updateDiluAgent({
                            message: errorMessage,
                            jsonPayload: status.result,
                            exportUrl: null,
                            jobStage: null,
                            jobStageMessage: null,
                        });
                    } else if (status.status === "cancelled") {
                        updateDiluAgent({
                            message: "[INFO] Clinical analysis cancelled.",
                            jsonPayload: status.result,
                            exportUrl: null,
                            jobStage: null,
                            jobStageMessage: null,
                        });
                    }
                },
                (pollError) => {
                    if (pollerRef.current) {
                        pollerRef.current.stop();
                        pollerRef.current = null;
                    }
                    updateDiluAgent({
                        message: `[ERROR] ${pollError}`,
                        jsonPayload: null,
                        exportUrl: null,
                        jobStage: null,
                        jobStageMessage: null,
                        isRunning: false,
                    });
                },
            );
        } catch (error) {
            const description =
                error instanceof Error ? error.message : "Unexpected error";
            updateDiluAgent({
                message: `[ERROR] ${description}`,
                jsonPayload: null,
                exportUrl: null,
                jobStage: null,
                jobStageMessage: null,
                isRunning: false,
            });
        } finally {
            if (!pollerRef.current) {
                updateDiluAgent({ isRunning: false });
            }
        }
    };

    const handleRunSession = async () => {
        const cleanedDrugs = form.drugs.trim();
        if (!cleanedDrugs) {
            resetOutputs();
            updateDiluAgent({
                isRunning: false,
                message: "[ERROR] At least one therapy drug is required for DILI analysis.",
                jsonPayload: null,
                exportUrl: null,
                jobStage: null,
                jobStageMessage: null,
            });
            return;
        }

        const missingLabs = [form.alt, form.altMax, form.alp, form.alpMax].some(
            (value) => !value.trim(),
        );
        if (missingLabs) {
            setIsMissingLabsModalOpen(true);
            return;
        }

        await executeRunSession(null);
    };

    const handleConfirmMissingLabs = async () => {
        setIsMissingLabsModalOpen(false);
        await executeRunSession(true);
    };

    const handleCancelMissingLabs = () => {
        setIsMissingLabsModalOpen(false);
    };

    const handleClear = () => {
        if (exportUrl) {
            URL.revokeObjectURL(exportUrl);
        }
        updateDiluAgent({
            settings: DEFAULT_SETTINGS,
            form: {
                patientName: "",
                visitDate: "",
                anamnesis: "",
                drugs: "",
                alt: "",
                altMax: "",
                alp: "",
                alpMax: "",
                useRag: false,
            },
            message: "",
            jsonPayload: null,
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
        updateDiluAgent({ isExpanded: !isExpanded });
    };

    // ---------------------------------------------------------------------------
    // Render
    // ---------------------------------------------------------------------------
    const spinner = (
        <div className="session-spinner">
            <div className="spinner-wheel" />
            <p className="spinner-label">
                {`${jobStageMessage || "Generating report"}${
                    jobProgress > 0 ? `... ${Math.round(jobProgress)}%` : "..."
                }`}
            </p>
        </div>
    );

    const jsonText = useMemo(() => {
        if (jsonPayload === null || jsonPayload === undefined) return "";
        try {
            return JSON.stringify(jsonPayload, null, 2);
        } catch {
            return `${jsonPayload}`;
        }
    }, [jsonPayload]);

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
        <>
            <main className="page-container">
                <header className="page-header">
                    <p className="eyebrow">DILIGENT Clinical Copilot</p>
                    <h1>Drug-Induced Liver Injury analysis</h1>
                    <p className="lede">
                        Provide clinical context and lab data to generate a structured hepatotoxicity assessment.
                    </p>
                </header>

                <div className="main-form-grid">
                    {/* Clinical Inputs Column */}
                    <section className="card clinical-inputs">
                        <div className="card-header">
                            <h2>Clinical Context</h2>
                            <p className="helper">Summarize history, current therapy, and key liver labs for this visit.</p>
                        </div>

                        {/* Section 1: Clinical Context */}
                        <div className="clinical-section">
                            <div className="field">
                                <label className="field-label" htmlFor="anamnesis">Anamnesis</label>
                                <textarea
                                    id="anamnesis"
                                    placeholder="Patient history and clinical observations..."
                                    value={form.anamnesis}
                                    onChange={(e) => handleFormChange("anamnesis", e.target.value)}
                                />
                                <span className="field-helper">
                                    Include relevant exams and previous lab results when available.
                                </span>
                            </div>
                            <div className="field">
                                <label className="field-label" htmlFor="drugs">Current Drugs</label>
                                <textarea
                                    id="drugs"
                                    placeholder="Medication list with dosages..."
                                    value={form.drugs}
                                    onChange={(e) => handleFormChange("drugs", e.target.value)}
                                />
                                <span className="field-helper">List current therapies, dosage and schedule.</span>
                            </div>
                        </div>

                        {/* Section 2: Lab Values */}
                        <div className="clinical-section">
                            <p className="section-title">Lab values</p>
                            <div className="lab-layout">
                                <div className="lab-widget">
                                    <div className="lab-row">
                                        <div className="field compact-field">
                                            <label className="field-label" htmlFor="alt">ALT (U/L)</label>
                                            <input
                                                id="alt"
                                                type="text"
                                                placeholder="189"
                                                value={form.alt}
                                                onChange={(e) => handleFormChange("alt", e.target.value)}
                                            />
                                        </div>
                                        <div className="field compact-field">
                                            <label className="field-label" htmlFor="alt-max">ALT Max (U/L)</label>
                                            <input
                                                id="alt-max"
                                                type="text"
                                                placeholder="47"
                                                value={form.altMax}
                                                onChange={(e) => handleFormChange("altMax", e.target.value)}
                                            />
                                            <span className="field-helper">Upper limit of normal</span>
                                        </div>
                                    </div>
                                    <div className="lab-row">
                                        <div className="field compact-field">
                                            <label className="field-label" htmlFor="alp">ALP (U/L)</label>
                                            <input
                                                id="alp"
                                                type="text"
                                                placeholder="140"
                                                value={form.alp}
                                                onChange={(e) => handleFormChange("alp", e.target.value)}
                                            />
                                        </div>
                                        <div className="field compact-field">
                                            <label className="field-label" htmlFor="alp-max">ALP Max (U/L)</label>
                                            <input
                                                id="alp-max"
                                                type="text"
                                                placeholder="150"
                                                value={form.alpMax}
                                                onChange={(e) => handleFormChange("alpMax", e.target.value)}
                                            />
                                            <span className="field-helper">Upper limit of normal</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="lab-controls-panel">
                                    <div className="advanced-options">
                                        <p className="advanced-options-header">Evidence Retrieval</p>
                                        <div className="toggle-row">
                                            <span className="toggle-label">Enable RAG for supporting evidence</span>
                                            <label className="toggle-switch">
                                                <span className="visually-hidden">Enable RAG for supporting evidence</span>
                                                <input
                                                    type="checkbox"
                                                    id="use-rag"
                                                    checked={form.useRag}
                                                    onChange={(e) => handleFormChange("useRag", e.target.checked)}
                                                />
                                                <span className="toggle-track" aria-hidden="true">
                                                    <span className="toggle-thumb" />
                                                </span>
                                            </label>
                                        </div>
                                    </div>
                                    <div className="lab-controls-future" aria-hidden="true" />
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Patient Information Column */}
                    <section className="card patient-info">
                        <div className="card-header">
                            <h2>Patient Information</h2>
                            <p className="helper">Basic demographics and visit data.</p>
                        </div>

                        {/* Basic Fields */}
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

                        {/* Action Buttons */}
                        <div className="action-stack">
                            <button
                                className="btn btn-primary"
                                type="button"
                                disabled={isRunning}
                                onClick={handleRunSession}
                            >
                                {isRunning ? "Running..." : "Run DILI analysis"}
                            </button>
                            <button
                                className="btn btn-secondary"
                                type="button"
                                disabled={!exportUrl}
                                onClick={handleDownload}
                            >
                                Download report
                            </button>
                            <button className="btn btn-tertiary" type="button" onClick={handleClear}>
                                Clear all
                            </button>
                        </div>
                    </section>

                    {/* Report Output Section */}
                    <section className="report-section">
                        <div
                            className="report-shell"
                            style={isExpanded ? { maxHeight: "none" } : undefined}
                        >
                            <div className="report-header">
                                <p className="report-eyebrow">Session output</p>
                                <h2>Report Output</h2>
                                <p className="report-subtitle">Markdown rendering of the clinical report.</p>
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

                            <div
                                className="report-content"
                                style={isExpanded ? { maxHeight: "none" } : undefined}
                            >
                                {reportContent}
                            </div>
                        </div>
                    </section>
                </div>

                {jsonPayload !== null && (
                    <section className="card json-card">
                        <div className="card-header">
                            <h2>JSON Response</h2>
                            <p className="helper">Raw payload returned by the backend for troubleshooting.</p>
                        </div>
                        <pre className="code-block">{jsonText}</pre>
                    </section>
                )}
            </main>
            <ConfirmModal
                isOpen={isMissingLabsModalOpen}
                title="Missing ALT/ALP labs"
                message="ALT/ALP values are missing; pattern will be undetermined. Continue anyway?"
                confirmLabel="Continue anyway"
                cancelLabel="Cancel"
                onConfirm={() => { void handleConfirmMissingLabs(); }}
                onCancel={handleCancelMissingLabs}
            />
        </>
    );
}
