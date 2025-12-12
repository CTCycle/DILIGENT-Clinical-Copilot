import React, { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    CLOUD_PROVIDERS,
    DEFAULT_SETTINGS,
    PARSING_MODEL_CHOICES,
    REPORT_EXPORT_FILENAME,
} from "../constants";
import { useAppState } from "../context/AppStateContext";
import { pullModels, runClinicalSession } from "../services/api";
import { RuntimeSettings } from "../types";
import {
    buildClinicalPayload,
    createDownloadUrl,
    normalizeVisitDateInput,
    resolveCloudSelection,
} from "../utils";
import { ConfigModal } from "../components/ConfigModal";

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

const SettingsIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
);

// ---------------------------------------------------------------------------
// DiluAgentPage
// ---------------------------------------------------------------------------
export function DiluAgentPage(): React.JSX.Element {
    const { state, updateDiluAgent } = useAppState();
    const {
        settings,
        form,
        cloudSelection,
        message,
        jsonPayload,
        exportUrl,
        isRunning,
        isPulling,
        isExpanded,
    } = state.diluAgent;

    const [configModalOpen, setConfigModalOpen] = useState(false);

    const cloudEnabled = settings.useCloudServices;
    const pullDisabled = cloudEnabled || isPulling;

    // Cleanup export URL on unmount
    useEffect(() => {
        return () => {
            if (exportUrl) {
                URL.revokeObjectURL(exportUrl);
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
        updateDiluAgent({ message: "", jsonPayload: null, exportUrl: null });
    };

    const handleSettingsChange = (next: Partial<RuntimeSettings>) => {
        const merged = { ...settings, ...next };
        const selection = resolveCloudSelection(merged.provider, merged.cloudModel);
        updateDiluAgent({
            settings: {
                ...merged,
                provider: selection.provider || DEFAULT_SETTINGS.provider,
                cloudModel: selection.model,
            },
            cloudSelection: { provider: selection.provider, model: selection.model },
        });
    };

    const handleUseCloudChange = (value: boolean) => {
        handleSettingsChange({ useCloudServices: value });
    };

    const handleProviderChange = (provider: string) => {
        const selection = resolveCloudSelection(provider, settings.cloudModel);
        updateDiluAgent({
            cloudSelection: { provider: selection.provider, model: selection.model },
        });
        handleSettingsChange({
            provider: selection.provider,
            cloudModel: selection.model,
        });
    };

    const handleCloudModelChange = (model: string) => {
        const selection = resolveCloudSelection(settings.provider, model);
        updateDiluAgent({
            cloudSelection: { provider: selection.provider, model: selection.model },
        });
        handleSettingsChange({ cloudModel: selection.model });
    };

    const handleFormChange = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) => {
        updateDiluAgent({ form: { ...form, [key]: value } });
    };

    const handleVisitDateChange = (value: string) => {
        handleFormChange("visitDate", normalizeVisitDateInput(value));
    };

    const handleRunSession = async () => {
        updateDiluAgent({ isRunning: true });
        resetOutputs();
        try {
            const payload = buildClinicalPayload(form, settings);
            const result = await runClinicalSession(payload);
            let newExportUrl: string | null = null;
            if (result.message.trim()) {
                newExportUrl = createDownloadUrl(result.message, REPORT_EXPORT_FILENAME);
            }
            updateDiluAgent({
                message: result.message,
                jsonPayload: result.json,
                exportUrl: newExportUrl,
            });
        } finally {
            updateDiluAgent({ isRunning: false });
        }
    };

    const handlePullModels = async () => {
        if (cloudEnabled) return;
        updateDiluAgent({ isPulling: true });
        resetOutputs();
        try {
            const result = await pullModels([settings.parsingModel, settings.clinicalModel]);
            updateDiluAgent({ message: result.message, jsonPayload: result.json });
        } finally {
            updateDiluAgent({ isPulling: false });
        }
    };

    const handleClear = () => {
        if (exportUrl) {
            URL.revokeObjectURL(exportUrl);
        }
        const defaultSelection = resolveCloudSelection(
            DEFAULT_SETTINGS.provider,
            DEFAULT_SETTINGS.cloudModel,
        );
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
            cloudSelection: { provider: defaultSelection.provider, model: defaultSelection.model },
            message: "",
            jsonPayload: null,
            exportUrl: null,
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
            <p className="spinner-label">Generating report...</p>
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

    return (
        <>
            {/* Config Modal */}
            <ConfigModal isOpen={configModalOpen} onClose={() => setConfigModalOpen(false)}>
                {/* Group 1: Execution Mode */}
                <div className="modal-section">
                    <p className="modal-section-title">Execution Mode</p>
                    <label className="field checkbox">
                        <input
                            type="checkbox"
                            id="use-cloud-services"
                            checked={cloudEnabled}
                            onChange={(e) => handleUseCloudChange(e.target.checked)}
                        />
                        <span className="field-label">Use Cloud Services</span>
                    </label>
                </div>

                {/* Group 2: Cloud Configuration */}
                <div className="modal-section">
                    <p className="modal-section-title">Cloud Configuration</p>
                    <div className="field">
                        <label className="field-label" htmlFor="cloud-service">Cloud Service</label>
                        <select
                            id="cloud-service"
                            value={cloudSelection.provider}
                            onChange={(e) => handleProviderChange(e.target.value)}
                            disabled={!cloudEnabled}
                        >
                            {CLOUD_PROVIDERS.map((provider) => (
                                <option key={provider} value={provider}>{provider}</option>
                            ))}
                        </select>
                    </div>
                    <div className="field">
                        <label className="field-label" htmlFor="cloud-model">Cloud Model</label>
                        <select
                            id="cloud-model"
                            value={cloudSelection.model ?? ""}
                            onChange={(e) => handleCloudModelChange(e.target.value)}
                            disabled={!cloudEnabled}
                        >
                            {(CLOUD_MODEL_CHOICES[cloudSelection.provider] || []).map((model) => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Group 3: Local Configuration */}
                <div className="modal-section">
                    <p className="modal-section-title">Local Configuration</p>
                    <div className="field">
                        <label className="field-label" htmlFor="parsing-model">Parsing Model</label>
                        <select
                            id="parsing-model"
                            value={settings.parsingModel}
                            onChange={(e) => handleSettingsChange({ parsingModel: e.target.value })}
                            disabled={cloudEnabled}
                        >
                            {PARSING_MODEL_CHOICES.map((model) => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                    <div className="field">
                        <label className="field-label" htmlFor="clinical-model">Clinical Model</label>
                        <select
                            id="clinical-model"
                            value={settings.clinicalModel}
                            onChange={(e) => handleSettingsChange({ clinicalModel: e.target.value })}
                            disabled={cloudEnabled}
                        >
                            {CLINICAL_MODEL_CHOICES.map((model) => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Group 4: Advanced */}
                <div className="modal-section">
                    <p className="modal-section-title">Advanced</p>
                    <div className="field">
                        <label className="field-label" htmlFor="temperature">Temperature (Ollama)</label>
                        <input
                            id="temperature"
                            type="number"
                            min={0}
                            max={2}
                            step={0.05}
                            value={settings.temperature}
                            onChange={(e) =>
                                handleSettingsChange({ temperature: Number.parseFloat(e.target.value) || 0 })
                            }
                            disabled={cloudEnabled}
                        />
                    </div>
                    <label className="field checkbox">
                        <input
                            type="checkbox"
                            id="enable-reasoning"
                            checked={settings.reasoning}
                            onChange={(e) => handleSettingsChange({ reasoning: e.target.checked })}
                            disabled={cloudEnabled}
                        />
                        <span className="field-label">Enable SDL/Reasoning (Ollama)</span>
                    </label>
                </div>

                {/* Pull Models Button */}
                <div className="modal-footer">
                    <button
                        className="btn btn-primary"
                        type="button"
                        disabled={pullDisabled}
                        onClick={handlePullModels}
                    >
                        {isPulling ? "Pulling models..." : "Pull Selected Models"}
                    </button>
                </div>
            </ConfigModal>

            {/* Page Content */}
            <main className="page-container">
                <header className="page-header">
                    <p className="eyebrow">DILIGENT Clinical Copilot</p>
                    <h1>Drug-Induced Liver Injury analysis</h1>
                    <p className="lede">
                        Provide clinical context and lab data to generate a structured hepatotoxicity assessment.
                    </p>
                    <button
                        className="btn btn-secondary config-trigger"
                        type="button"
                        onClick={() => setConfigModalOpen(true)}
                    >
                        <SettingsIcon />
                        <span>Model Configurations</span>
                    </button>
                </header>

                <div className="main-form-grid">
                    {/* Clinical Inputs Column */}
                    <section className="card clinical-inputs">
                        <div className="card-header">
                            <h2>Clinical Inputs</h2>
                            <p className="helper">Describe the clinical picture and therapies for this visit.</p>
                        </div>

                        {/* Section 1: Clinical Context */}
                        <div className="clinical-section">
                            <p className="section-title">Clinical context</p>
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
                            <div className="lab-grid">
                                <div className="field">
                                    <label className="field-label" htmlFor="alt">ALT (U/L)</label>
                                    <input
                                        id="alt"
                                        type="text"
                                        placeholder="189"
                                        value={form.alt}
                                        onChange={(e) => handleFormChange("alt", e.target.value)}
                                    />
                                </div>
                                <div className="field">
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
                                <div className="field">
                                    <label className="field-label" htmlFor="alp">ALP (U/L)</label>
                                    <input
                                        id="alp"
                                        type="text"
                                        placeholder="140"
                                        value={form.alp}
                                        onChange={(e) => handleFormChange("alp", e.target.value)}
                                    />
                                </div>
                                <div className="field">
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
                    </section>

                    {/* Patient Information Column */}
                    <section className="card patient-info">
                        <div className="card-header">
                            <h2>Patient Information</h2>
                            <p className="helper">Basic demographics and visit data.</p>
                        </div>

                        {/* Advanced Options Block */}
                        <div className="advanced-options">
                            <p className="advanced-options-header">Evidence Retrieval</p>
                            <div className="toggle-row">
                                <span className="toggle-label">Enable RAG for supporting evidence</span>
                                <label className="toggle-switch">
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
                                {isRunning ? (
                                    spinner
                                ) : message ? (
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown">
                                        {message}
                                    </ReactMarkdown>
                                ) : (
                                    <div className="report-placeholder">
                                        No report generated yet. Run analysis to see results.
                                    </div>
                                )}
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
        </>
    );
}
