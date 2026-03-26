import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { BooleanToggle } from "../components/BooleanToggle";
import { ConfirmModal } from "../components/ConfirmModal";
import { DEFAULT_FORM_STATE, DEFAULT_SETTINGS, REPORT_EXPORT_FILENAME } from "../constants";
import { useAppState } from "../context/AppStateContext";
import { pollClinicalJobStatus, startClinicalJob } from "../services/api";
import { JobStatus, JobStatusResponse } from "../types";
import { buildClinicalPayload, normalizeVisitDateInput } from "../utils";

const TODAY_ISO = new Date().toISOString().slice(0, 10);
const DEFAULT_POLL_INTERVAL_MS = 1000;
const REPORT_SECTIONS = [
  { id: "patient-context", title: "Patient Context", navLabel: "Patient Context" },
  { id: "drug-exposure", title: "Drug Exposure", navLabel: "Drug Exposure" },
  { id: "lab-evaluation", title: "Lab Evaluation", navLabel: "Lab Evaluation" },
  { id: "causality-assessment", title: "Causality Assessment", navLabel: "Causality" },
  { id: "conclusion", title: "Conclusion", navLabel: "Conclusion" },
] as const;

type SectionId = (typeof REPORT_SECTIONS)[number]["id"];
type RiskClassification = "High" | "Possible" | "Unlikely";

type DrugAnalysisItem = {
  name: string;
  score: number;
  risk: RiskClassification;
  rationale: string;
};

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

const PdfIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <path d="M8 13h2a1 1 0 0 1 0 2H8v3" />
    <path d="M14 18v-5h2a1.5 1.5 0 0 1 0 3h-2" />
  </svg>
);

const LinkIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10 13a5 5 0 0 0 7.54.54l2.91-2.91a5 5 0 0 0-7.07-7.07L11.68 5.3" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-2.91 2.91a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </svg>
);

function isTerminalJobStatus(status: JobStatus | null): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function parsePositiveNumber(rawValue: string): number | null {
  const normalized = rawValue.trim();
  if (!normalized) return null;
  const parsed = Number.parseFloat(normalized);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return parsed;
}

function formatRatio(value: number | null): string {
  return value === null ? "--" : value.toFixed(1);
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function classifyRiskByScore(score: number): RiskClassification {
  if (score >= 0.72) return "High";
  if (score >= 0.45) return "Possible";
  return "Unlikely";
}

function riskClassName(risk: RiskClassification): string {
  return risk.toLowerCase();
}

function normalizeReportText(value: string): string {
  return value.trim().replace(/^\[(ERROR|INFO)\]\s*/, "");
}

function isStatusMessage(value: string): boolean {
  return value.startsWith("[ERROR]") || value.startsWith("[INFO]");
}

function formatVisitDateLabel(visitDate: string): string {
  if (!visitDate) return "Not provided";
  const parsed = new Date(`${visitDate}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return "Not provided";
  return parsed.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}

function trendGlyph(ratio: number | null): string {
  if (ratio === null) return "-";
  if (ratio > 1.1) return "^";
  if (ratio < 0.9) return "v";
  return "=";
}

export function DiliAgentPage(): React.JSX.Element {
  const { state, updateDiliAgent } = useAppState();
  const { settings, form, message, jobId, jobProgress, jobStatus, jobStageMessage, isRunning } = state.diliAgent;

  const pollerRef = useRef<{ stop: () => void } | null>(null);
  const reportContentRef = useRef<HTMLDivElement | null>(null);

  const [isMissingLabsModalOpen, setIsMissingLabsModalOpen] = useState(false);
  const [showValidation, setShowValidation] = useState(false);
  const [selectedDrug, setSelectedDrug] = useState<string | null>(null);
  const [collapsedSections, setCollapsedSections] = useState<Record<SectionId, boolean>>({
    "patient-context": false,
    "drug-exposure": false,
    "lab-evaluation": false,
    "causality-assessment": false,
    conclusion: false,
  });

  useEffect(() => {
    return () => {
      if (pollerRef.current) {
        pollerRef.current.stop();
        pollerRef.current = null;
      }
    };
  }, []);

  const resetOutputs = () => {
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
      const stage = status.result && typeof status.result.progress_stage === "string" ? status.result.progress_stage : null;
      const stageMessage = status.result && typeof status.result.progress_message === "string" ? status.result.progress_message : null;
      const resolvedProgress = typeof status.progress === "number" ? status.progress : 0;
      updateDiliAgent({
        jobProgress: terminalStatus ? 100 : resolvedProgress,
        jobStatus: status.status,
        jobStage: stage,
        jobStageMessage: stageMessage,
        isRunning: !terminalStatus,
      });

      if (!terminalStatus) return;
      stopPoller();

      if (status.status === "completed") {
        const report = typeof status.result?.report === "string" ? status.result.report : "";
        updateDiliAgent({
          message: report || "[INFO] Clinical analysis completed.",
          exportUrl: null,
          jobStage: null,
          jobStageMessage: null,
        });
      } else if (status.status === "failed") {
        updateDiliAgent({
          message: status.error ? `[ERROR] ${status.error}` : "[ERROR] Clinical analysis failed.",
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
      pollerRef.current = pollClinicalJobStatus(jobIdToPoll, intervalMs, handleJobStatusUpdate, handlePollingError);
    },
    [handleJobStatusUpdate, handlePollingError, stopPoller],
  );

  const executeRunSession = async (allowMissingLabs: boolean | null) => {
    setShowValidation(true);
    updateDiliAgent({ isRunning: true });
    resetOutputs();
    stopPoller();
    try {
      const payload = buildClinicalPayload(form, settings, allowMissingLabs);
      const startResult = await startClinicalJob(payload);
      updateDiliAgent({
        jobId: startResult.job_id,
        jobProgress: 0,
        jobStatus: startResult.status,
        jobStage: "session_initialization",
        jobStageMessage: "Initializing clinical session",
      });
      startPolling(startResult.job_id, startResult.poll_interval * 1000);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Unexpected error";
      updateDiliAgent({
        message: `[ERROR] ${description}`,
        exportUrl: null,
        jobStage: null,
        jobStageMessage: null,
        isRunning: false,
      });
    } finally {
      if (!pollerRef.current) updateDiliAgent({ isRunning: false });
    }
  };

  useEffect(() => {
    if (!isRunning || !jobId || isTerminalJobStatus(jobStatus) || pollerRef.current) return;
    startPolling(jobId, DEFAULT_POLL_INTERVAL_MS);
  }, [isRunning, jobId, jobStatus, startPolling]);

  const handleConfirmMissingLabs = async () => {
    setIsMissingLabsModalOpen(false);
    await executeRunSession(true);
  };

  const handleCancelMissingLabs = () => setIsMissingLabsModalOpen(false);

  const handleClear = () => {
    setShowValidation(false);
    setSelectedDrug(null);
    setCollapsedSections({
      "patient-context": false,
      "drug-exposure": false,
      "lab-evaluation": false,
      "causality-assessment": false,
      conclusion: false,
    });
    updateDiliAgent({
      settings: DEFAULT_SETTINGS,
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

  const toggleSection = (id: SectionId) => {
    setCollapsedSections((current) => ({ ...current, [id]: !current[id] }));
  };

  const toggleAllSections = () => {
    const areAllExpanded = REPORT_SECTIONS.every((section) => !collapsedSections[section.id]);
    const shouldCollapse = areAllExpanded;
    setCollapsedSections({
      "patient-context": shouldCollapse,
      "drug-exposure": shouldCollapse,
      "lab-evaluation": shouldCollapse,
      "causality-assessment": shouldCollapse,
      conclusion: shouldCollapse,
    });
  };

  const drugsList = useMemo(() => {
    const fragments = form.drugs.split(/[\n,;]+/g).map((item) => item.trim()).filter((item) => item.length > 0);
    return [...new Set(fragments)];
  }, [form.drugs]);

  useEffect(() => {
    if (!drugsList.length) {
      setSelectedDrug(null);
      return;
    }
    setSelectedDrug((current) => (current && drugsList.includes(current) ? current : drugsList[0]));
  }, [drugsList]);

  const altValue = parsePositiveNumber(form.alt);
  const altMaxValue = parsePositiveNumber(form.altMax);
  const alpValue = parsePositiveNumber(form.alp);
  const alpMaxValue = parsePositiveNumber(form.alpMax);

  const altRatio = altValue && altMaxValue ? altValue / altMaxValue : null;
  const alpRatio = alpValue && alpMaxValue ? alpValue / alpMaxValue : null;
  const rRatio = altRatio && alpRatio ? altRatio / alpRatio : null;

  const overallRisk = useMemo<RiskClassification>(() => {
    if ((altRatio ?? 0) >= 5 || (alpRatio ?? 0) >= 2) return "High";
    if ((altRatio ?? 0) >= 3 || (alpRatio ?? 0) >= 1.5) return "Possible";
    return "Unlikely";
  }, [altRatio, alpRatio]);

  const labPattern = useMemo(() => {
    if (rRatio === null) return "Pattern pending (requires ALT and ALP ratios).";
    if (rRatio >= 5) return "Predominantly hepatocellular pattern.";
    if (rRatio <= 2) return "Predominantly cholestatic pattern.";
    return "Mixed liver injury pattern.";
  }, [rRatio]);

  const perDrugAnalysis = useMemo<DrugAnalysisItem[]>(() => {
    if (!drugsList.length) return [];
    const baselineScore = overallRisk === "High" ? 0.82 : overallRisk === "Possible" ? 0.58 : 0.32;
    return drugsList
      .map((drug, index) => {
        const score = Math.max(0.12, baselineScore - index * 0.09);
        const risk = classifyRiskByScore(score);
        const rationale = altRatio !== null || alpRatio !== null
          ? `Preview based on available enzymes${altRatio !== null ? `, ALT ${formatRatio(altRatio)}x ULN` : ""}${alpRatio !== null ? `, ALP ${formatRatio(alpRatio)}x ULN` : ""}.`
          : "Preview pending liver enzyme values.";
        return { name: drug, score, risk, rationale };
      })
      .sort((left, right) => right.score - left.score);
  }, [drugsList, overallRisk, altRatio, alpRatio]);

  const selectedDrugAnalysis = useMemo(
    () => (selectedDrug ? perDrugAnalysis.find((item) => item.name === selectedDrug) ?? null : null),
    [selectedDrug, perDrugAnalysis],
  );

  const fieldErrors = useMemo(() => {
    const errors: Partial<Record<keyof typeof form, string>> = {};
    const numericPairs: Array<{ key: "alt" | "altMax" | "alp" | "alpMax"; label: string }> = [
      { key: "alt", label: "ALT" },
      { key: "altMax", label: "ALT Max" },
      { key: "alp", label: "ALP" },
      { key: "alpMax", label: "ALP Max" },
    ];
    for (const { key, label } of numericPairs) {
      const rawValue = form[key].trim();
      if (!rawValue) continue;
      if (parsePositiveNumber(rawValue) === null) errors[key] = `${label} must be a positive number.`;
    }
    if (showValidation && !form.drugs.trim()) errors.drugs = "At least one therapy drug is required.";
    return errors;
  }, [form, showValidation]);

  const hasBlockingValidation = useMemo(
    () => Boolean(fieldErrors.drugs || fieldErrors.alt || fieldErrors.altMax || fieldErrors.alp || fieldErrors.alpMax),
    [fieldErrors],
  );

  const handleRunSession = async () => {
    setShowValidation(true);
    if (hasBlockingValidation) {
      updateDiliAgent({
        message: "[ERROR] Resolve highlighted input issues before generating the report.",
        exportUrl: null,
        isRunning: false,
        jobStage: null,
        jobStageMessage: null,
      });
      return;
    }
    const missingLabs = [form.alt, form.altMax, form.alp, form.alpMax].some((value) => !value.trim());
    if (missingLabs) {
      setIsMissingLabsModalOpen(true);
      return;
    }
    await executeRunSession(null);
  };

  const statusNotice = isStatusMessage(message) ? normalizeReportText(message) : "";
  const finalNarrativeReport = !isStatusMessage(message) ? message.trim() : "";
  const patientLabel = form.patientName.trim() || "Unidentified patient";
  const visitDateLabel = formatVisitDateLabel(form.visitDate);
  const labSummaryLine = altRatio !== null || alpRatio !== null
    ? `ALT ${formatRatio(altRatio)}x ULN | ALP ${formatRatio(alpRatio)}x ULN`
    : "ALT/ALP pending";

  const reportMarkdown = useMemo(() => {
    const lines: string[] = [
      "# Drug-Induced Liver Injury Report",
      "",
      "## Patient Context",
      `- **Patient:** ${patientLabel}`,
      `- **Visit date:** ${visitDateLabel}`,
      `- **Clinical context:** ${form.anamnesis.trim() || "Not provided."}`,
      "",
      "## Drug Exposure",
      `- **Selected focus drug:** ${selectedDrug ?? "Not selected"}`,
    ];
    if (perDrugAnalysis.length) {
      lines.push("", "### Per-drug analysis");
      perDrugAnalysis.forEach((item, index) => {
        lines.push(`${index + 1}. **${item.name}** - ${item.risk} (${formatPercent(item.score)} confidence). ${item.rationale}`);
      });
    } else {
      lines.push("- No therapy entries provided.");
    }
    lines.push(
      "",
      "## Lab Evaluation",
      `- **ALT:** ${form.alt.trim() || "N/A"} U/L`,
      `- **ALT ULN:** ${form.altMax.trim() || "N/A"} U/L`,
      `- **ALP:** ${form.alp.trim() || "N/A"} U/L`,
      `- **ALP ULN:** ${form.alpMax.trim() || "N/A"} U/L`,
      `- **ALT ratio:** ${formatRatio(altRatio)}x ULN`,
      `- **ALP ratio:** ${formatRatio(alpRatio)}x ULN`,
      `- **R ratio:** ${formatRatio(rRatio)}`,
      `- **Pattern:** ${labPattern}`,
      "",
      "## Causality Assessment",
      `- **Current risk classification:** ${overallRisk}`,
      `- **Evidence retrieval:** RAG ${form.useRag ? "enabled" : "disabled"}, Web search ${form.useWebSearch ? "enabled" : "disabled"}`,
      "",
      "## Conclusion",
    );
    if (finalNarrativeReport) {
      lines.push("Model-backed narrative generated.", "", "### Final Narrative", finalNarrativeReport);
    } else {
      lines.push(`Preliminary classification: **${overallRisk}**. Generate/refresh report for final model-backed reasoning.`);
    }
    return lines.join("\n");
  }, [patientLabel, visitDateLabel, form.anamnesis, form.alt, form.altMax, form.alp, form.alpMax, form.useRag, form.useWebSearch, selectedDrug, perDrugAnalysis, altRatio, alpRatio, rRatio, labPattern, overallRisk, finalNarrativeReport]);

  const handleCopyReport = async () => {
    try {
      await navigator.clipboard.writeText(reportMarkdown);
    } catch (copyError) {
      console.error("Failed to copy report:", copyError);
    }
  };

  const handleDownloadMarkdown = () => {
    const blob = new Blob([reportMarkdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = REPORT_EXPORT_FILENAME;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const handleExportPdf = () => {
    const reportMarkup = reportContentRef.current?.innerHTML;
    if (!reportMarkup) return;
    const printWindow = window.open("", "_blank", "noopener,noreferrer");
    if (!printWindow) return;
    printWindow.document.write(`
      <!doctype html>
      <html><head><meta charset="utf-8" />
      <title>Clinical Report</title>
      <style>
        body { font-family: Manrope, "Segoe UI", Arial, sans-serif; margin: 24px; color: #0f172a; line-height: 1.5; }
        h2 { margin: 0 0 8px; font-size: 24px; } h3 { margin: 18px 0 8px; font-size: 18px; }
        p { margin: 8px 0; } ul, ol { margin: 8px 0; padding-left: 20px; }
        details > summary { font-weight: 600; cursor: pointer; margin: 8px 0; }
        .dili-report-section { border: 1px solid #d0d7e2; border-radius: 12px; margin-bottom: 12px; padding: 12px; break-inside: avoid; }
        .dili-risk-badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; border: 1px solid #d0d7e2; }
      </style></head><body>${reportMarkup}</body></html>
    `);
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
  };

  const areAllSectionsExpanded = REPORT_SECTIONS.every((section) => !collapsedSections[section.id]);

  const spinner = (
    <div className="session-spinner" role="status" aria-live="polite">
      <div className="spinner-wheel" />
      <p className="spinner-label">
        {`${jobStageMessage || "Generating report"}${jobProgress > 0 ? `... ${Math.round(jobProgress)}%` : "..."}`}
      </p>
    </div>
  );

  const reportContent = isRunning && !isTerminalJobStatus(jobStatus) ? <div className="dili-loading-state">{spinner}</div> : null;

  return (
    <>
      <main className="page-container dili-workspace">
        <header className="dili-sticky-header">
          <div className="dili-sticky-header-main">
            <h1>Drug-Induced Liver Injury analysis</h1>
            <p>Narrative report workspace with progressive drafting and explicit final generation.</p>
          </div>
          <div className="dili-sticky-header-meta">
            <div className="dili-header-chip">
              <span className="dili-header-chip-label">Patient</span>
              <strong>{patientLabel}</strong>
            </div>
            <div className="dili-header-chip">
              <span className="dili-header-chip-label">Visit</span>
              <strong>{visitDateLabel}</strong>
            </div>
            <div className={`dili-risk-badge ${riskClassName(overallRisk)}`}>{overallRisk} risk</div>
            <div className="dili-header-chip">
              <span className="dili-header-chip-label">Lab snapshot</span>
              <strong>{labSummaryLine}</strong>
            </div>
          </div>
        </header>

        <div className="dili-layout">
          <section className="card dili-input-panel">
            <div className="dili-panel-header">
              <h2>Clinical Inputs</h2>
              <p>Capture context once and keep the report updated as you type.</p>
            </div>

            <div className="dili-input-group">
              <h3>Patient and Visit</h3>
              <div className="field">
                <label className="field-label" htmlFor="patient-name">Patient Name</label>
                <input
                  id="patient-name"
                  type="text"
                  placeholder="e.g., Marco Rossi"
                  value={form.patientName}
                  onChange={(event) => handleFormChange("patientName", event.target.value)}
                />
              </div>
              <div className="field">
                <label className="field-label" htmlFor="visit-date">Visit Date</label>
                <input
                  id="visit-date"
                  type="date"
                  max={TODAY_ISO}
                  value={form.visitDate}
                  onChange={(event) => handleVisitDateChange(event.target.value)}
                />
              </div>
            </div>

            <div className="dili-input-group">
              <h3>Clinical Context</h3>
              <div className="field">
                <label className="field-label" htmlFor="anamnesis">Anamnesis</label>
                <textarea
                  id="anamnesis"
                  placeholder="Patient history and clinical observations..."
                  value={form.anamnesis}
                  onChange={(event) => handleFormChange("anamnesis", event.target.value)}
                />
                <span className="field-helper">Include relevant exams and previous lab results when available.</span>
              </div>
            </div>

            <div className="dili-input-group">
              <h3>Drug Exposure</h3>
              <div className="field">
                <label className="field-label" htmlFor="drugs">Current Drugs</label>
                <textarea
                  id="drugs"
                  placeholder="One drug per line, include dosage and schedule."
                  value={form.drugs}
                  onChange={(event) => handleFormChange("drugs", event.target.value)}
                />
                <span className="field-helper">List current therapies, dosage and schedule.</span>
                {fieldErrors.drugs ? <span className="dili-inline-error" role="alert">{fieldErrors.drugs}</span> : null}
              </div>

              <div className="dili-drug-selector">
                <p className="dili-drug-selector-label">Selectable drug list</p>
                {drugsList.length ? (
                  <div className="dili-drug-selector-grid" role="listbox" aria-label="Drug list">
                    {drugsList.map((drug) => (
                      <button
                        key={drug}
                        type="button"
                        className={`dili-drug-item ${selectedDrug === drug ? "is-selected" : ""}`}
                        onClick={() => setSelectedDrug(drug)}
                        aria-selected={selectedDrug === drug}
                      >
                        {drug}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="dili-empty-note">Add therapy drugs to enable per-drug assessment.</p>
                )}
              </div>
            </div>

            <div className="dili-input-group">
              <h3>Lab Evaluation Inputs</h3>
              <div className="dili-lab-grid">
                <div className={`dili-lab-card ${(altRatio ?? 0) > 1 ? "is-abnormal" : ""}`}>
                  <div className="field compact-field">
                    <label className="field-label" htmlFor="alt">ALT (U/L)</label>
                    <input id="alt" type="text" placeholder="189" value={form.alt} onChange={(event) => handleFormChange("alt", event.target.value)} />
                    {fieldErrors.alt ? <span className="dili-inline-error" role="alert">{fieldErrors.alt}</span> : null}
                  </div>
                  <div className="field compact-field">
                    <label className="field-label" htmlFor="alt-max">ALT Max (U/L)</label>
                    <input id="alt-max" type="text" placeholder="47" value={form.altMax} onChange={(event) => handleFormChange("altMax", event.target.value)} />
                    {fieldErrors.altMax ? <span className="dili-inline-error" role="alert">{fieldErrors.altMax}</span> : null}
                  </div>
                  <p className="dili-lab-ratio">Ratio vs ULN: <strong>{formatRatio(altRatio)}x</strong><span aria-hidden="true">{trendGlyph(altRatio)}</span></p>
                </div>

                <div className={`dili-lab-card ${(alpRatio ?? 0) > 1 ? "is-abnormal" : ""}`}>
                  <div className="field compact-field">
                    <label className="field-label" htmlFor="alp">ALP (U/L)</label>
                    <input id="alp" type="text" placeholder="140" value={form.alp} onChange={(event) => handleFormChange("alp", event.target.value)} />
                    {fieldErrors.alp ? <span className="dili-inline-error" role="alert">{fieldErrors.alp}</span> : null}
                  </div>
                  <div className="field compact-field">
                    <label className="field-label" htmlFor="alp-max">ALP Max (U/L)</label>
                    <input id="alp-max" type="text" placeholder="150" value={form.alpMax} onChange={(event) => handleFormChange("alpMax", event.target.value)} />
                    {fieldErrors.alpMax ? <span className="dili-inline-error" role="alert">{fieldErrors.alpMax}</span> : null}
                  </div>
                  <p className="dili-lab-ratio">Ratio vs ULN: <strong>{formatRatio(alpRatio)}x</strong><span aria-hidden="true">{trendGlyph(alpRatio)}</span></p>
                </div>
              </div>
            </div>

            <div className="dili-input-group">
              <h3>Evidence Retrieval</h3>
              <div className="dili-toggle-stack">
                <BooleanToggle id="use-rag" label="Enable RAG for supporting evidence" checked={form.useRag} onChange={(checked) => handleFormChange("useRag", checked)} />
                <BooleanToggle id="use-web-search" label="Enable web search for supporting evidence" checked={form.useWebSearch} onChange={(checked) => handleFormChange("useWebSearch", checked)} />
              </div>
            </div>

            <div className="dili-actions">
              <button className="btn btn-primary" type="button" disabled={isRunning} onClick={handleRunSession}>
                {isRunning ? "Generating..." : "Generate / Refresh Report"}
              </button>
              <p className="dili-action-helper">Live preview updates continuously. Use this action for model-backed final output.</p>
              <button className="btn btn-secondary" type="button" onClick={handleDownloadMarkdown}>Export Markdown</button>
              <button className="btn btn-tertiary" type="button" onClick={handleClear}>Clear all</button>
            </div>
          </section>

          <section className="dili-report-panel">
            <div className="dili-report-shell">
              <div className="dili-report-header">
                <p className="report-eyebrow">Live narrative report</p>
                <h2>Clinical Report Preview</h2>
                <p className="report-subtitle">Scaffold remains visible and fills progressively as data arrives.</p>
              </div>

              <div className="dili-report-toolbar">
                <div className="dili-toolbar-main">
                  <button className="dili-toolbar-btn" type="button" onClick={handleCopyReport} title="Copy report markdown"><CopyIcon /><span>Copy</span></button>
                  <button className="dili-toolbar-btn" type="button" onClick={handleExportPdf} title="Export to PDF"><PdfIcon /><span>Export PDF</span></button>
                  <button className="dili-toolbar-btn" type="button" onClick={handleDownloadMarkdown} title="Export to markdown"><DownloadIcon /><span>Export MD</span></button>
                  <button className="dili-toolbar-btn" type="button" onClick={toggleAllSections} title={areAllSectionsExpanded ? "Collapse all sections" : "Expand all sections"}>
                    <ExpandIcon />
                    <span>{areAllSectionsExpanded ? "Collapse all" : "Expand all"}</span>
                  </button>
                </div>

                <nav className="dili-anchor-nav" aria-label="Report section navigation">
                  {REPORT_SECTIONS.map((section) => (
                    <a key={section.id} href={`#report-${section.id}`} className="dili-anchor-link"><LinkIcon /><span>{section.navLabel}</span></a>
                  ))}
                </nav>
              </div>

              {statusNotice ? <div className={`status-message ${message.startsWith("[ERROR]") ? "error" : "info"}`}>{statusNotice}</div> : null}
              {reportContent}

              <div className="dili-report-content" ref={reportContentRef}>
                <article id="report-patient-context" className="dili-report-section">
                  <header className="dili-section-header">
                    <h3>Patient Context</h3>
                    <button type="button" className="dili-section-toggle" onClick={() => toggleSection("patient-context")}>{collapsedSections["patient-context"] ? "Expand" : "Collapse"}</button>
                  </header>
                  {!collapsedSections["patient-context"] ? (
                    <>
                      <p><strong>{patientLabel}</strong> evaluated on <strong>{visitDateLabel}</strong>.</p>
                      <p>{form.anamnesis.trim() || "Clinical history pending. Add anamnesis details to enrich interpretation context."}</p>
                      <details className="dili-inline-explainer">
                        <summary>Why this matters</summary>
                        <p>Recent symptoms, prior liver disease, and competing etiologies modify DILI probability and should be documented before final conclusions.</p>
                      </details>
                    </>
                  ) : null}
                </article>

                <article id="report-drug-exposure" className="dili-report-section">
                  <header className="dili-section-header">
                    <h3>Drug Exposure</h3>
                    <button type="button" className="dili-section-toggle" onClick={() => toggleSection("drug-exposure")}>{collapsedSections["drug-exposure"] ? "Expand" : "Collapse"}</button>
                  </header>
                  {!collapsedSections["drug-exposure"] ? (
                    <>
                      <p>Focus drug: <strong>{selectedDrug || "Not selected"}</strong>.</p>
                      {perDrugAnalysis.length ? (
                        <ol className="dili-ranking-list">
                          {perDrugAnalysis.map((item) => (
                            <li key={item.name}>
                              <span className="dili-ranking-name">{item.name}</span>
                              <span className={`dili-risk-badge ${riskClassName(item.risk)}`}>{item.risk}</span>
                              <span className="dili-ranking-score">{formatPercent(item.score)}</span>
                              <p>{item.rationale}</p>
                            </li>
                          ))}
                        </ol>
                      ) : (
                        <p className="dili-empty-note">Add medications to generate per-drug analysis and ranked causality summary.</p>
                      )}
                      <details className="dili-reasoning">
                        <summary>Detailed reasoning</summary>
                        <p>Ranking is preview-only until model-backed generation. Use it to identify which agents require deeper evidence review.</p>
                      </details>
                    </>
                  ) : null}
                </article>

                <article id="report-lab-evaluation" className="dili-report-section">
                  <header className="dili-section-header">
                    <h3>Lab Evaluation</h3>
                    <button type="button" className="dili-section-toggle" onClick={() => toggleSection("lab-evaluation")}>{collapsedSections["lab-evaluation"] ? "Expand" : "Collapse"}</button>
                  </header>
                  {!collapsedSections["lab-evaluation"] ? (
                    <>
                      <p>
                        {altRatio !== null ? <span className="dili-inline-highlight">ALT {formatRatio(altRatio)}x ULN</span> : <span className="dili-empty-value">ALT ratio pending</span>}
                        {" | "}
                        {alpRatio !== null ? <span className="dili-inline-highlight">ALP {formatRatio(alpRatio)}x ULN</span> : <span className="dili-empty-value">ALP ratio pending</span>}
                      </p>
                      <p>R ratio: <strong>{formatRatio(rRatio)}</strong>. {labPattern}</p>
                      <details className="dili-inline-explainer">
                        <summary>Lab interpretation note</summary>
                        <p>Abnormal values are highlighted to prioritize review. Ratios versus ULN are used for consistent severity framing across visits.</p>
                      </details>
                    </>
                  ) : null}
                </article>

                <article id="report-causality-assessment" className="dili-report-section">
                  <header className="dili-section-header">
                    <h3>Causality Assessment</h3>
                    <button type="button" className="dili-section-toggle" onClick={() => toggleSection("causality-assessment")}>{collapsedSections["causality-assessment"] ? "Expand" : "Collapse"}</button>
                  </header>
                  {!collapsedSections["causality-assessment"] ? (
                    <>
                      <p>Current classification: <span className={`dili-risk-badge ${riskClassName(overallRisk)}`}>{overallRisk}</span></p>
                      <p>Evidence support: RAG <strong>{form.useRag ? "enabled" : "disabled"}</strong>, web search <strong>{form.useWebSearch ? "enabled" : "disabled"}</strong>.</p>
                      {selectedDrugAnalysis ? <p>Selected drug priority: <strong>{selectedDrugAnalysis.name}</strong> ({formatPercent(selectedDrugAnalysis.score)} confidence preview).</p> : null}
                      <details className="dili-reasoning">
                        <summary>Detailed reasoning</summary>
                        <p>This section combines current enzyme pattern and drug ranking preview. Final confidence is established only after the full model-backed report run.</p>
                      </details>
                    </>
                  ) : null}
                </article>

                <article id="report-conclusion" className="dili-report-section">
                  <header className="dili-section-header">
                    <h3>Conclusion</h3>
                    <button type="button" className="dili-section-toggle" onClick={() => toggleSection("conclusion")}>{collapsedSections.conclusion ? "Expand" : "Collapse"}</button>
                  </header>
                  {!collapsedSections.conclusion ? (
                    <>
                      {finalNarrativeReport ? (
                        <>
                          <p>Final model-backed narrative available. Review and export as needed.</p>
                          <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown">{finalNarrativeReport}</ReactMarkdown>
                        </>
                      ) : (
                        <p>
                          Preliminary narrative: current data supports a{" "}
                          <span className={`dili-risk-badge ${riskClassName(overallRisk)}`}>{overallRisk}</span>{" "}
                          likelihood profile. Generate/refresh for full causality reasoning.
                        </p>
                      )}
                    </>
                  ) : null}
                </article>
              </div>
            </div>
          </section>
        </div>
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
