import React, { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  CLINICAL_MODEL_CHOICES,
  CLOUD_MODEL_CHOICES,
  CLOUD_PROVIDERS,
  DEFAULT_FORM_STATE,
  DEFAULT_SETTINGS,
  PARSING_MODEL_CHOICES,
  REPORT_EXPORT_FILENAME,
} from "./constants";
import { pullModels, runClinicalSession } from "./services/api";
import { ClinicalFormState, RuntimeSettings } from "./types";
import {
  buildClinicalPayload,
  createDownloadUrl,
  normalizeVisitDateInput,
  resolveCloudSelection,
} from "./utils";

const todayIso = new Date().toISOString().slice(0, 10);

function App(): React.JSX.Element {
  const [settings, setSettings] = useState<RuntimeSettings>(DEFAULT_SETTINGS);
  const [form, setForm] = useState<ClinicalFormState>(DEFAULT_FORM_STATE);
  const [cloudSelection, setCloudSelection] = useState(
    resolveCloudSelection(DEFAULT_SETTINGS.provider, DEFAULT_SETTINGS.cloudModel),
  );
  const [message, setMessage] = useState("");
  const [jsonPayload, setJsonPayload] = useState<unknown | null>(null);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isPulling, setIsPulling] = useState(false);

  const cloudEnabled = settings.useCloudServices;
  const pullDisabled = cloudEnabled || isPulling;

  useEffect(() => {
    return () => {
      if (exportUrl) {
        URL.revokeObjectURL(exportUrl);
      }
    };
  }, [exportUrl]);

  const resetOutputs = () => {
    setMessage("");
    setJsonPayload(null);
    if (exportUrl) {
      URL.revokeObjectURL(exportUrl);
      setExportUrl(null);
    }
  };

  const handleSettingsChange = (next: Partial<RuntimeSettings>) => {
    setSettings((prev) => {
      const merged = { ...prev, ...next };
      const selection = resolveCloudSelection(
        merged.provider,
        merged.cloudModel,
      );
      setCloudSelection(selection);
      return {
        ...merged,
        provider: selection.provider || DEFAULT_SETTINGS.provider,
        cloudModel: selection.model,
      };
    });
  };

  const handleUseCloudChange = (value: boolean) => {
    handleSettingsChange({ useCloudServices: value });
  };

  const handleProviderChange = (provider: string) => {
    const selection = resolveCloudSelection(provider, settings.cloudModel);
    setCloudSelection(selection);
    handleSettingsChange({
      provider: selection.provider,
      cloudModel: selection.model,
    });
  };

  const handleCloudModelChange = (model: string) => {
    const selection = resolveCloudSelection(settings.provider, model);
    setCloudSelection(selection);
    handleSettingsChange({ cloudModel: selection.model });
  };

  const handleFormChange = <K extends keyof ClinicalFormState>(
    key: K,
    value: ClinicalFormState[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleVisitDateChange = (value: string) => {
    handleFormChange("visitDate", normalizeVisitDateInput(value));
  };

  const handleRunSession = async () => {
    setIsRunning(true);
    resetOutputs();
    try {
      const payload = buildClinicalPayload(form);
      const result = await runClinicalSession(payload);
      setMessage(result.message);
      setJsonPayload(result.json);
      if (result.message.trim()) {
        setExportUrl(createDownloadUrl(result.message, REPORT_EXPORT_FILENAME));
      }
    } finally {
      setIsRunning(false);
    }
  };

  const handlePullModels = async () => {
    if (cloudEnabled) {
      return;
    }
    setIsPulling(true);
    resetOutputs();
    try {
      const result = await pullModels([
        settings.parsingModel,
        settings.clinicalModel,
      ]);
      setMessage(result.message);
      setJsonPayload(result.json);
    } finally {
      setIsPulling(false);
    }
  };

  const handleClear = () => {
    if (exportUrl) {
      URL.revokeObjectURL(exportUrl);
    }
    setSettings(DEFAULT_SETTINGS);
    setCloudSelection(
      resolveCloudSelection(DEFAULT_SETTINGS.provider, DEFAULT_SETTINGS.cloudModel),
    );
    setForm(DEFAULT_FORM_STATE);
    setMessage("");
    setJsonPayload(null);
    setExportUrl(null);
    setSidebarOpen(false);
  };

  const handleDownload = () => {
    if (!exportUrl) {
      return;
    }
    const anchor = document.createElement("a");
    anchor.href = exportUrl;
    anchor.download = REPORT_EXPORT_FILENAME;
    anchor.click();
  };

  const spinner = (
    <div className="session-spinner">
      <div className="spinner-wheel" />
      <p className="spinner-label">Generating report...</p>
    </div>
  );

  const jsonText = useMemo(() => {
    if (jsonPayload === null || jsonPayload === undefined) {
      return "";
    }
    try {
      return JSON.stringify(jsonPayload, null, 2);
    } catch {
      return `${jsonPayload}`;
    }
  }, [jsonPayload]);

  return (
    <div className="app-shell">
      <aside className={`config-drawer ${sidebarOpen ? "open" : ""}`}>
        <div className="drawer-header">
          <div>
            <p className="drawer-title">Model Configurations</p>
            <p className="drawer-subtitle">
              Adjust runtime preferences for DILI analysis
            </p>
          </div>
          <button
            className="text-button"
            type="button"
            onClick={() => setSidebarOpen(false)}
          >
            Close
          </button>
        </div>

        <div className="drawer-section">
          <label className="field checkbox">
            <input
              type="checkbox"
              checked={form.useRag}
              onChange={(event) => handleFormChange("useRag", event.target.checked)}
            />
            <span>Use Retrieval Augmented Generation (RAG)</span>
          </label>

          <label className="field checkbox">
            <input
              type="checkbox"
              checked={cloudEnabled}
              onChange={(event) => handleUseCloudChange(event.target.checked)}
            />
            <span>Use Cloud Services</span>
          </label>
        </div>

        <div className="drawer-section">
          <p className="section-title">Cloud Configuration</p>
          <label className="field">
            <span>Cloud Service</span>
            <select
              value={cloudSelection.provider}
              onChange={(event) => handleProviderChange(event.target.value)}
              disabled={!cloudEnabled}
            >
              {CLOUD_PROVIDERS.map((provider) => (
                <option key={provider} value={provider}>
                  {provider}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Cloud Model</span>
            <select
              value={cloudSelection.model ?? ""}
              onChange={(event) => handleCloudModelChange(event.target.value)}
              disabled={!cloudEnabled}
            >
              {(CLOUD_MODEL_CHOICES[cloudSelection.provider] || []).map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="drawer-section">
          <p className="section-title">Local Configuration</p>
          <label className="field">
            <span>Parsing Model</span>
            <select
              value={settings.parsingModel}
              onChange={(event) =>
                handleSettingsChange({ parsingModel: event.target.value })
              }
              disabled={cloudEnabled}
            >
              {PARSING_MODEL_CHOICES.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Clinical Model</span>
            <select
              value={settings.clinicalModel}
              onChange={(event) =>
                handleSettingsChange({ clinicalModel: event.target.value })
              }
              disabled={cloudEnabled}
            >
              {CLINICAL_MODEL_CHOICES.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="drawer-section">
          <label className="field">
            <span>Temperature (Ollama)</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.05}
              value={settings.temperature}
              onChange={(event) =>
                handleSettingsChange({
                  temperature: Number.parseFloat(event.target.value) || 0,
                })
              }
              disabled={cloudEnabled}
            />
          </label>

          <label className="field checkbox">
            <input
              type="checkbox"
              checked={settings.reasoning}
              onChange={(event) =>
                handleSettingsChange({ reasoning: event.target.checked })
              }
              disabled={cloudEnabled}
            />
            <span>Enable SDL/Reasoning (Ollama)</span>
          </label>
        </div>

        <button
          className="primary-button"
          type="button"
          disabled={pullDisabled}
          onClick={handlePullModels}
        >
          {isPulling ? "Pulling models..." : "Pull Selected Models"}
        </button>
      </aside>

      {sidebarOpen && (
        <div className="drawer-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      <button
        className="drawer-toggle"
        type="button"
        onClick={() => setSidebarOpen(true)}
        aria-label="Open configuration"
      >
        <span>&rsaquo;</span>
      </button>

      <main className="page-container">
        <header className="page-header">
          <div>
            <p className="eyebrow">DILIGENT Clinical Copilot</p>
            <h1>Drug-Induced Liver Injury analysis</h1>
            <p className="lede">
              Provide clinical context and lab data to generate a structured
              hepatotoxicity assessment.
            </p>
          </div>
          <div className="toolbar">
            <button
              className="ghost-button"
              type="button"
              onClick={() => setSidebarOpen(true)}
            >
              Configure models
            </button>
          </div>
        </header>

        <div className="content-grid">
          <section className="card">
            <div className="card-header">
              <h2>Clinical Inputs</h2>
              <p className="helper">
                Describe the clinical picture and therapies for this visit.
              </p>
            </div>
            <label className="field">
              <span>Anamnesis</span>
              <textarea
                placeholder="Describe the clinical picture, including exams and labs when relevant..."
                value={form.anamnesis}
                onChange={(event) =>
                  handleFormChange("anamnesis", event.target.value)
                }
              />
            </label>
            <label className="field">
              <span>Current Drugs</span>
              <textarea
                placeholder="List current therapies, dosage and schedule..."
                value={form.drugs}
                onChange={(event) => handleFormChange("drugs", event.target.value)}
              />
            </label>

            <div className="two-column">
              <label className="field">
                <span>ALT</span>
                <input
                  type="text"
                  placeholder="e.g., 189 or 189 U/L"
                  value={form.alt}
                  onChange={(event) => handleFormChange("alt", event.target.value)}
                />
              </label>
              <label className="field">
                <span>ALT Max</span>
                <input
                  type="text"
                  placeholder="e.g., 47 U/L"
                  value={form.altMax}
                  onChange={(event) =>
                    handleFormChange("altMax", event.target.value)
                  }
                />
              </label>
            </div>

            <div className="two-column">
              <label className="field">
                <span>ALP</span>
                <input
                  type="text"
                  placeholder="e.g., 140 or 140 U/L"
                  value={form.alp}
                  onChange={(event) => handleFormChange("alp", event.target.value)}
                />
              </label>
              <label className="field">
                <span>ALP Max</span>
                <input
                  type="text"
                  placeholder="e.g., 150 U/L"
                  value={form.alpMax}
                  onChange={(event) =>
                    handleFormChange("alpMax", event.target.value)
                  }
                />
              </label>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <h2>Patient Information</h2>
              <p className="helper">Basic demographics and visit data.</p>
            </div>

            <label className="field">
              <span>Patient Name</span>
              <input
                type="text"
                placeholder="e.g., Marco Rossi"
                value={form.patientName}
                onChange={(event) =>
                  handleFormChange("patientName", event.target.value)
                }
              />
            </label>

            <label className="field">
              <span>Visit Date</span>
              <input
                type="date"
                max={todayIso}
                value={form.visitDate}
                onChange={(event) => handleVisitDateChange(event.target.value)}
              />
            </label>

            <div className="stacked-actions">
              <button
                className="primary-button"
                type="button"
                disabled={isRunning}
                onClick={handleRunSession}
              >
                {isRunning ? "Running..." : "Run DILI analysis"}
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={!exportUrl}
                onClick={handleDownload}
              >
                Download report
              </button>
              <button className="ghost-button" type="button" onClick={handleClear}>
                Clear all
              </button>
            </div>

            <div className="spinner-container">
              {isRunning ? spinner : <div className="placeholder-row" />}
            </div>
          </section>
        </div>

        <section className="card">
          <div className="card-header">
            <h2>Report Output</h2>
            <p className="helper">Markdown rendering of the clinical report.</p>
          </div>
          <div className="report-surface">
            {isRunning ? (
              spinner
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]} className="markdown">
                {message || "_No report generated yet._"}
              </ReactMarkdown>
            )}
          </div>
        </section>

        {jsonPayload !== null && (
          <section className="card json-card">
            <div className="card-header">
              <h2>JSON Response</h2>
              <p className="helper">
                Raw payload returned by the backend for troubleshooting.
              </p>
            </div>
            <pre className="code-block">{jsonText}</pre>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
