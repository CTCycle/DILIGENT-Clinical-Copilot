import React from "react";

import {
    CLINICAL_MODEL_CHOICES,
    CLOUD_MODEL_CHOICES,
    CLOUD_PROVIDERS,
    DEFAULT_SETTINGS,
    PARSING_MODEL_CHOICES,
} from "../constants";
import { useAppState } from "../context/AppStateContext";
import { pullModels } from "../services/api";
import { RuntimeSettings } from "../types";
import { resolveCloudSelection } from "../utils";

export function ModelConfigPage(): React.JSX.Element {
    const { state, updateDiluAgent } = useAppState();
    const {
        settings,
        cloudSelection,
        exportUrl,
        isPulling,
    } = state.diluAgent;

    const cloudEnabled = settings.useCloudServices;
    const pullDisabled = cloudEnabled || isPulling;

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
        });
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

    return (
        <main className="page-container model-config-page">
            <header className="page-header">
                <p className="eyebrow">DILIGENT Clinical Copilot</p>
                <h1>Model Configurations</h1>
                <p className="lede">Adjust runtime preferences for DILI analysis.</p>
            </header>

            <div className="model-config-first-row">
                <section className="model-config-column">
                    <p className="modal-section-title">Local (Ollama) Parameters</p>
                    <div className="field">
                        <label className="field-label" htmlFor="local-parsing-model">
                            Data Extraction Model
                        </label>
                        <select
                            id="local-parsing-model"
                            value={settings.parsingModel}
                            onChange={(e) => handleSettingsChange({ parsingModel: e.target.value })}
                            disabled={cloudEnabled}
                        >
                            {PARSING_MODEL_CHOICES.map((model) => (
                                <option key={model} value={model}>
                                    {model}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="field">
                        <label className="field-label" htmlFor="local-clinical-model">Clinical Model</label>
                        <select
                            id="local-clinical-model"
                            value={settings.clinicalModel}
                            onChange={(e) => handleSettingsChange({ clinicalModel: e.target.value })}
                            disabled={cloudEnabled}
                        >
                            {CLINICAL_MODEL_CHOICES.map((model) => (
                                <option key={model} value={model}>
                                    {model}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="field">
                        <label className="field-label" htmlFor="local-temperature">Temperature (Ollama)</label>
                        <input
                            id="local-temperature"
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
                            id="local-reasoning"
                            checked={settings.reasoning}
                            onChange={(e) => handleSettingsChange({ reasoning: e.target.checked })}
                            disabled={cloudEnabled}
                        />
                        <span className="field-label">Enable SDL/Reasoning (Ollama)</span>
                    </label>
                    <button
                        className="btn btn-primary model-config-pull-btn"
                        type="button"
                        disabled={pullDisabled}
                        onClick={handlePullModels}
                    >
                        {isPulling ? "Pulling models..." : "Pull Selected Models"}
                    </button>
                </section>

                <section className={`model-config-column ${cloudEnabled ? "" : "model-config-column-disabled"}`}>
                    <p className="modal-section-title">Cloud Provider Parameters</p>
                    <label className="field checkbox">
                        <input
                            type="checkbox"
                            id="use-cloud-providers"
                            checked={cloudEnabled}
                            onChange={(e) => handleUseCloudChange(e.target.checked)}
                        />
                        <span className="field-label">Use Cloud Providers</span>
                    </label>

                    {cloudEnabled ? (
                        <div className="model-config-cloud-fields">
                            <div className="field">
                                <label className="field-label" htmlFor="cloud-service">Cloud Service</label>
                                <select
                                    id="cloud-service"
                                    value={cloudSelection.provider}
                                    onChange={(e) => handleProviderChange(e.target.value)}
                                >
                                    {CLOUD_PROVIDERS.map((provider) => (
                                        <option key={provider} value={provider}>
                                            {provider}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div className="field">
                                <label className="field-label" htmlFor="cloud-model">Cloud Model</label>
                                <select
                                    id="cloud-model"
                                    value={cloudSelection.model ?? ""}
                                    onChange={(e) => handleCloudModelChange(e.target.value)}
                                >
                                    {(CLOUD_MODEL_CHOICES[cloudSelection.provider] || []).map((model) => (
                                        <option key={model} value={model}>
                                            {model}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    ) : (
                        <p className="helper model-config-disabled-note">
                            Cloud providers are disabled and not active.
                        </p>
                    )}
                </section>
            </div>

            <div className="model-config-second-row" aria-hidden="true" />
        </main>
    );
}
