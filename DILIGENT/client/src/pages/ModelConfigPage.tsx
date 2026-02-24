import React, { useEffect, useMemo, useState } from "react";

import { AccessKeyModal } from "../components/AccessKeyModal";
import { CLOUD_MODEL_CHOICES, DEFAULT_SETTINGS } from "../constants";
import { useAppState } from "../context/AppStateContext";
import {
    fetchModelConfigState,
    pullModels,
    updateModelConfigState,
} from "../services/api";
import { AccessKeyProvider, ModelConfigStateResponse, ModelConfigUpdateRequest, RuntimeSettings } from "../types";

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

const PullIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 3v11" />
        <path d="m7 10 5 5 5-5" />
        <path d="M5 21h14" />
    </svg>
);

function isAccessKeyProvider(provider: string): provider is AccessKeyProvider {
    return provider === "openai" || provider === "gemini";
}

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

function resolveAvailabilityBadgeClass(modelAvailableInOllama: boolean | undefined): string {
    return modelAvailableInOllama ? "model-config-summary-ok" : "model-config-summary-muted";
}

function resolveAvailabilityLabel(modelAvailableInOllama: boolean | undefined): string {
    if (modelAvailableInOllama === undefined) {
        return "Unknown";
    }
    return modelAvailableInOllama ? "Installed" : "Not installed";
}

export function ModelConfigPage(): React.JSX.Element {
    const { state, updateDiluAgent } = useAppState();
    const { settings, isPulling } = state.diluAgent;

    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [localModels, setLocalModels] = useState<ModelConfigStateResponse["local_models"]>([]);
    const [cloudChoices, setCloudChoices] = useState<Record<string, string[]>>(CLOUD_MODEL_CHOICES);
    const [modelSearchQuery, setModelSearchQuery] = useState("");
    const [statusMessage, setStatusMessage] = useState("");
    const [openProviderModal, setOpenProviderModal] = useState<AccessKeyProvider | null>(null);

    const cloudEnabled = settings.useCloudServices;
    const localSelectionDisabled = cloudEnabled || isSaving || isLoading;
    const ollamaControlsDisabled = cloudEnabled || isSaving || isLoading;
    const availableLocalModelCount = useMemo(
        () => localModels.filter((model) => model.available_in_ollama).length,
        [localModels],
    );
    const filteredLocalModels = useMemo(() => {
        const query = modelSearchQuery.trim().toLowerCase();
        if (!query) {
            return localModels;
        }
        return localModels.filter((model) => model.name.toLowerCase().includes(query));
    }, [localModels, modelSearchQuery]);

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
    const selectedClinicalModel = useMemo(
        () => localModels.find((model) => model.name === settings.clinicalModel) || null,
        [localModels, settings.clinicalModel],
    );
    const selectedTextExtractionModel = useMemo(
        () => localModels.find((model) => model.name === settings.parsingModel) || null,
        [localModels, settings.parsingModel],
    );
    const clinicalAvailabilityClassName = resolveAvailabilityBadgeClass(
        selectedClinicalModel?.available_in_ollama,
    );
    const clinicalAvailabilityLabel = resolveAvailabilityLabel(
        selectedClinicalModel?.available_in_ollama,
    );
    const extractionAvailabilityClassName = resolveAvailabilityBadgeClass(
        selectedTextExtractionModel?.available_in_ollama,
    );
    const extractionAvailabilityLabel = resolveAvailabilityLabel(
        selectedTextExtractionModel?.available_in_ollama,
    );

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

    const handlePullModel = async (requestedModelName: string) => {
        const candidate = requestedModelName.trim();
        if (!candidate) {
            setStatusMessage("[ERROR] Enter a model name to pull from Ollama.");
            return;
        }

        updateDiluAgent({ isPulling: true });
        setStatusMessage(`[INFO] Pulling '${candidate}' from Ollama...`);
        try {
            const result = await pullModels([candidate]);
            setStatusMessage(result.message);
            await loadModelConfig();
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
                            <div className="model-config-row-header-top">
                                <div>
                                    <p className="modal-section-title">Local Model Catalog</p>
                                    <p className="helper">
                                        Select one clinical model and one text extraction model from the full catalog.
                                        {` ${availableLocalModelCount} installed in Ollama.`}
                                    </p>
                                </div>
                                <div className="model-config-search">
                                    <label className="visually-hidden" htmlFor="model-search-by-name">Search model by name</label>
                                    <input
                                        id="model-search-by-name"
                                        type="text"
                                        value={modelSearchQuery}
                                        placeholder="Search by name..."
                                        onChange={(event) => setModelSearchQuery(event.target.value)}
                                        disabled={isLoading}
                                    />
                                </div>
                            </div>
                        </div>

                        <ul className="model-config-local-cards">
                            {!localModels.length && (
                                <li className="model-config-empty-note">
                                    {isLoading ? "Loading local model catalog..." : "No local model catalog entries available."}
                                </li>
                            )}

                            {!!localModels.length && !filteredLocalModels.length && (
                                <li className="model-config-empty-note">
                                    No models match "{modelSearchQuery.trim()}".
                                </li>
                            )}

                            {filteredLocalModels.map((model) => (
                                <li
                                    key={model.name}
                                    className={`model-config-local-card ${model.available_in_ollama ? "is-available" : ""}`}
                                >
                                    <div className="model-config-local-card-header">
                                        <h3>{model.name}</h3>
                                        <div className="model-config-local-card-actions">
                                            <span
                                                className={`model-config-availability-pill ${model.available_in_ollama ? "is-available" : "is-unavailable"}`}
                                            >
                                                {model.available_in_ollama ? "Available in Ollama" : "Not installed"}
                                            </span>
                                            {!model.available_in_ollama && (
                                                <button
                                                    className="model-config-card-pull"
                                                    type="button"
                                                    onClick={() => { void handlePullModel(model.name); }}
                                                    disabled={isPulling || ollamaControlsDisabled}
                                                    aria-label={`Pull ${model.name} from Ollama`}
                                                    title={`Pull ${model.name}`}
                                                >
                                                    <PullIcon />
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                    <p className="model-config-family">Family: {model.family}</p>
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
                                            <span className="field-label">Text Extraction</span>
                                        </label>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className={`model-config-ollama-row ${ollamaControlsDisabled ? "is-disabled" : ""}`} aria-disabled={ollamaControlsDisabled}>
                        <p className="modal-section-title">Ollama Settings</p>
                        <label className="field checkbox">
                            <input
                                type="checkbox"
                                checked={settings.reasoning}
                                onChange={(e) => { void handleReasoningChange(e.target.checked); }}
                                disabled={ollamaControlsDisabled}
                            />
                            <span className="field-label">Enable SDL/Reasoning</span>
                        </label>
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

                        <div className="model-config-provider-grid">
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
                                            onClick={() => {
                                                if (isAccessKeyProvider(provider)) {
                                                    setOpenProviderModal(provider);
                                                }
                                            }}
                                            disabled={isSaving || isLoading}
                                            aria-label={`Manage ${PROVIDER_LABELS[provider] || provider} access keys`}
                                        >
                                            <KeyIcon />
                                        </button>
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
                        <p className="modal-section-title">Current Selection Summary</p>
                        <table className="model-config-summary-table" aria-label="Current selection summary">
                            <tbody>
                                <tr>
                                    <th scope="row">Clinical</th>
                                    <td>{settings.clinicalModel || "Not set"}</td>
                                    <td className={clinicalAvailabilityClassName}>
                                        {clinicalAvailabilityLabel}
                                    </td>
                                </tr>
                                <tr>
                                    <th scope="row">Text Extraction</th>
                                    <td>{settings.parsingModel || "Not set"}</td>
                                    <td className={extractionAvailabilityClassName}>
                                        {extractionAvailabilityLabel}
                                    </td>
                                </tr>
                                <tr>
                                    <th scope="row">Provider</th>
                                    <td>{PROVIDER_LABELS[activeProvider] || activeProvider}</td>
                                    <td>{cloudEnabled ? "Cloud active" : "Cloud disabled"}</td>
                                </tr>
                                <tr>
                                    <th scope="row">Cloud Model</th>
                                    <td>{activeCloudModel || "Not set"}</td>
                                    <td>{cloudEnabled ? "In use" : "Standby"}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>
            </div>

            {statusMessage && <p className="model-config-status-message">{statusMessage}</p>}
            <AccessKeyModal
                isOpen={openProviderModal !== null}
                provider={openProviderModal ?? "openai"}
                providerLabel={PROVIDER_LABELS[openProviderModal ?? "openai"] || (openProviderModal ?? "openai")}
                onClose={() => setOpenProviderModal(null)}
            />
        </main>
    );
}
