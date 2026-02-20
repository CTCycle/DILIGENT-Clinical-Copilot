import React, { useEffect, useMemo, useState } from "react";

import { CLOUD_MODEL_CHOICES, DEFAULT_SETTINGS } from "../constants";
import { useAppState } from "../context/AppStateContext";
import {
    fetchModelConfigState,
    pullModels,
    updateModelConfigState,
} from "../services/api";
import { ModelConfigStateResponse, ModelConfigUpdateRequest, RuntimeSettings } from "../types";

const PROVIDER_LABELS: Record<string, string> = {
    openai: "OpenAI",
    gemini: "Gemini",
};

const KeyIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="8" cy="15" r="3" />
        <path d="M11 15h10" />
        <path d="M18 12v6" />
    </svg>
);

function resolveProvider(
    provider: string | null | undefined,
    cloudChoices: Record<string, string[]>,
): string {
    const normalized = (provider || "").trim().toLowerCase();
    if (normalized && cloudChoices[normalized]) {
        return normalized;
    }
    if (cloudChoices.openai) {
        return "openai";
    }
    const fallback = Object.keys(cloudChoices)[0];
    return fallback || DEFAULT_SETTINGS.provider;
}

function resolveCloudModel(
    provider: string,
    cloudModel: string | null,
    cloudChoices: Record<string, string[]>,
): string | null {
    const options = cloudChoices[provider] || [];
    if (!options.length) {
        return null;
    }
    if (cloudModel && options.includes(cloudModel)) {
        return cloudModel;
    }
    return options[0];
}

export function ModelConfigPage(): React.JSX.Element {
    const { state, updateDiluAgent } = useAppState();
    const { settings, isPulling } = state.diluAgent;

    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [localModels, setLocalModels] = useState<ModelConfigStateResponse["local_models"]>([]);
    const [cloudChoices, setCloudChoices] = useState<Record<string, string[]>>(CLOUD_MODEL_CHOICES);
    const [pullModelName, setPullModelName] = useState("");
    const [statusMessage, setStatusMessage] = useState("");
    const [openProviderMenu, setOpenProviderMenu] = useState<string | null>(null);

    const cloudEnabled = settings.useCloudServices;
    const localSelectionDisabled = cloudEnabled || isSaving || isLoading;

    const applyConfigToState = (payload: ModelConfigStateResponse) => {
        setLocalModels(payload.local_models || []);
        setCloudChoices(payload.cloud_model_choices || CLOUD_MODEL_CHOICES);

        const provider = resolveProvider(payload.llm_provider, payload.cloud_model_choices);
        const cloudModel = resolveCloudModel(provider, payload.cloud_model, payload.cloud_model_choices);
        const nextSettings: RuntimeSettings = {
            ...settings,
            useCloudServices: payload.use_cloud_services,
            provider,
            cloudModel,
            parsingModel: payload.text_extraction_model || settings.parsingModel,
            clinicalModel: payload.clinical_model || settings.clinicalModel,
            reasoning: payload.ollama_reasoning,
        };
        updateDiluAgent({ settings: nextSettings });
    };

    const loadModelConfig = async () => {
        setIsLoading(true);
        try {
            const payload = await fetchModelConfigState();
            applyConfigToState(payload);
            setStatusMessage("");
        } catch (error) {
            const description = error instanceof Error ? error.message : "Unable to load model settings.";
            setStatusMessage(`[ERROR] ${description}`);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        void loadModelConfig();
    }, []);

    const persistConfigPatch = async (patch: ModelConfigUpdateRequest) => {
        setIsSaving(true);
        try {
            const payload = await updateModelConfigState(patch);
            applyConfigToState(payload);
            setStatusMessage("");
        } catch (error) {
            const description = error instanceof Error ? error.message : "Unable to save model settings.";
            setStatusMessage(`[ERROR] ${description}`);
        } finally {
            setIsSaving(false);
        }
    };

    const providerOptions = useMemo(() => {
        const values = Object.keys(cloudChoices);
        if (values.length) {
            return values;
        }
        return ["openai", "gemini"];
    }, [cloudChoices]);

    const activeProvider = resolveProvider(settings.provider, cloudChoices);
    const activeCloudModel = resolveCloudModel(activeProvider, settings.cloudModel, cloudChoices);
    const activeCloudModels = cloudChoices[activeProvider] || [];

    const handleRoleSelection = async (role: "clinical" | "text_extraction", modelName: string) => {
        if (role === "clinical") {
            await persistConfigPatch({ clinical_model: modelName });
            return;
        }
        await persistConfigPatch({ text_extraction_model: modelName });
    };

    const handleCloudSwitchChange = async (value: boolean) => {
        await persistConfigPatch({ use_cloud_services: value });
    };

    const handleProviderChange = async (provider: string) => {
        await persistConfigPatch({ llm_provider: provider });
    };

    const handleCloudModelChange = async (modelName: string) => {
        await persistConfigPatch({ cloud_model: modelName });
    };

    const handleReasoningChange = async (enabled: boolean) => {
        await persistConfigPatch({ ollama_reasoning: enabled });
    };

    const handlePullModel = async () => {
        const candidate = pullModelName.trim();
        if (!candidate) {
            setStatusMessage("[ERROR] Enter a model name to pull from Ollama.");
            return;
        }

        updateDiluAgent({ isPulling: true });
        try {
            const result = await pullModels([candidate]);
            setStatusMessage(result.message);
            await loadModelConfig();
            if (!result.message.startsWith("[ERROR]")) {
                setPullModelName("");
            }
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

            <div className="model-config-layout">
                <section className="model-config-left-column">
                    <div className={`model-config-local-row ${localSelectionDisabled ? "is-disabled" : ""}`} aria-disabled={localSelectionDisabled}>
                        <div className="model-config-row-header">
                            <p className="modal-section-title">Local Model Preview</p>
                            <p className="helper">Select one clinical model and one text extraction model.</p>
                        </div>

                        <div className="model-config-local-cards" role="list">
                            {!localModels.length && (
                                <p className="model-config-empty-note">
                                    {isLoading ? "Loading local models..." : "No local Ollama models available."}
                                </p>
                            )}

                            {localModels.map((model) => (
                                <article key={model.name} className="model-config-local-card" role="listitem">
                                    <h3>{model.name}</h3>
                                    <p>{model.description}</p>
                                    <div className="model-config-role-controls">
                                        <label className="field checkbox">
                                            <input
                                                type="radio"
                                                name="clinical-role"
                                                checked={settings.clinicalModel === model.name}
                                                onChange={() => { void handleRoleSelection("clinical", model.name); }}
                                                disabled={localSelectionDisabled}
                                            />
                                            <span className="field-label">Clinical Model</span>
                                        </label>
                                        <label className="field checkbox">
                                            <input
                                                type="radio"
                                                name="text-extraction-role"
                                                checked={settings.parsingModel === model.name}
                                                onChange={() => { void handleRoleSelection("text_extraction", model.name); }}
                                                disabled={localSelectionDisabled}
                                            />
                                            <span className="field-label">Text Extraction Model</span>
                                        </label>
                                    </div>
                                </article>
                            ))}
                        </div>
                    </div>

                    <div className="model-config-ollama-row">
                        <p className="modal-section-title">Ollama Settings</p>
                        <label className="field checkbox">
                            <input
                                type="checkbox"
                                checked={settings.reasoning}
                                onChange={(e) => { void handleReasoningChange(e.target.checked); }}
                                disabled={isSaving || isLoading}
                            />
                            <span className="field-label">Enable SDL/Reasoning</span>
                        </label>

                        <div className="model-config-pull-widget">
                            <label className="field-label" htmlFor="pull-model-name">Pull model from Ollama</label>
                            <div className="model-config-pull-actions">
                                <input
                                    id="pull-model-name"
                                    type="text"
                                    value={pullModelName}
                                    placeholder="e.g. qwen3:8b"
                                    onChange={(e) => setPullModelName(e.target.value)}
                                    disabled={isPulling || isSaving || isLoading}
                                />
                                <button
                                    className="btn btn-primary"
                                    type="button"
                                    onClick={handlePullModel}
                                    disabled={isPulling || isSaving || isLoading}
                                >
                                    {isPulling ? "Pulling..." : "Pull Model"}
                                </button>
                            </div>
                        </div>
                    </div>
                </section>

                <section className="model-config-right-column">
                    <div className="model-config-cloud-row">
                        <p className="modal-section-title">Cloud Model Control</p>
                        <label className="field checkbox">
                            <input
                                type="checkbox"
                                checked={cloudEnabled}
                                onChange={(e) => { void handleCloudSwitchChange(e.target.checked); }}
                                disabled={isSaving || isLoading}
                            />
                            <span className="field-label">Use Cloud Models</span>
                        </label>

                        <div className={`model-config-provider-grid ${cloudEnabled ? "" : "is-disabled"}`}>
                            <div className="model-config-provider-list">
                                {providerOptions.map((provider) => (
                                    <div
                                        key={provider}
                                        className={`model-config-provider-card ${activeProvider === provider ? "is-active" : ""}`}
                                    >
                                        <button
                                            className="model-config-provider-button"
                                            type="button"
                                            onClick={() => { void handleProviderChange(provider); }}
                                            disabled={!cloudEnabled || isSaving || isLoading}
                                        >
                                            <span>{PROVIDER_LABELS[provider] || provider}</span>
                                        </button>
                                        <button
                                            className="model-config-provider-key"
                                            type="button"
                                            aria-haspopup="menu"
                                            aria-expanded={openProviderMenu === provider}
                                            onClick={() =>
                                                setOpenProviderMenu((current) => (
                                                    current === provider ? null : provider
                                                ))
                                            }
                                            disabled={!cloudEnabled || isSaving || isLoading}
                                        >
                                            <KeyIcon />
                                        </button>
                                        {openProviderMenu === provider && (
                                            <div className="model-config-provider-menu" role="menu">
                                                API key management placeholder.
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            <div className="field">
                                <label className="field-label" htmlFor="cloud-model-select">Cloud Model</label>
                                <select
                                    id="cloud-model-select"
                                    value={activeCloudModel || ""}
                                    onChange={(e) => { void handleCloudModelChange(e.target.value); }}
                                    disabled={!cloudEnabled || isSaving || isLoading || !activeCloudModels.length}
                                >
                                    {activeCloudModels.map((modelName) => (
                                        <option key={modelName} value={modelName}>
                                            {modelName}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>

                    <div className="model-config-right-footer-row">
                        <p className="modal-section-title">Cloud Status</p>
                        <p className="helper">
                            Active provider: <strong>{PROVIDER_LABELS[activeProvider] || activeProvider}</strong>
                        </p>
                        <p className="helper">
                            Active model: <strong>{activeCloudModel || "Not set"}</strong>
                        </p>
                    </div>
                </section>
            </div>

            {statusMessage && <p className="model-config-status-message">{statusMessage}</p>}
        </main>
    );
}
