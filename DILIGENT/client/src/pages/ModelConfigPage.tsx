import React, { useEffect, useMemo, useState } from "react";

import { AccessKeyModal } from "../components/AccessKeyModal";
import { StatusMessage, resolveStatusTone } from "../components/StatusMessage";
import { CLOUD_MODEL_CHOICES } from "../constants";
import { useAppState } from "../context/AppStateContext";
import {
    CloudModelChoices,
    buildRuntimeSettingsFromConfig,
    resolveCloudChoices,
    resolveCloudModel,
    resolveProvider,
} from "../modelConfig";
import {
    fetchModelConfigState,
    pullModels,
    updateModelConfigState,
} from "../services/api";
import {
    AccessKeyProvider,
    LocalModelCard,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
    RuntimeSettings,
} from "../types";

const PROVIDER_LABELS: Record<string, string> = {
    openai: "OpenAI",
    gemini: "Gemini",
    tavily: "Tavily",
};

type ModelFilterKey = "installed" | "reasoning" | "small" | "extraction";

type DraftRuntimeConfig = {
    useCloudServices: boolean;
    provider: string;
    cloudModel: string | null;
    clinicalModel: string;
    parsingModel: string;
};

type PersistOptions = {
    syncDraft?: boolean;
    successMessage?: string;
};

const MODEL_FILTERS: Array<{ key: ModelFilterKey; label: string }> = [
    { key: "installed", label: "Installed" },
    { key: "reasoning", label: "Reasoning" },
    { key: "small", label: "Small models" },
    { key: "extraction", label: "Extraction" },
];

const KeyIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="8" cy="15" r="3" />
        <path d="M11 15h10" />
        <path d="M18 12v6" />
    </svg>
);

function isAccessKeyProvider(provider: string): provider is AccessKeyProvider {
    return provider === "openai" || provider === "gemini" || provider === "tavily";
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

function resolveDraftFromSettings(
    runtimeSettings: RuntimeSettings,
    choices: CloudModelChoices,
): DraftRuntimeConfig {
    const provider = resolveProvider(runtimeSettings.provider, choices);
    const cloudModel = resolveCloudModel(provider, runtimeSettings.cloudModel, choices);
    return {
        useCloudServices: runtimeSettings.useCloudServices,
        provider,
        cloudModel,
        clinicalModel: runtimeSettings.clinicalModel,
        parsingModel: runtimeSettings.parsingModel,
    };
}

function parseModelSizeInBillions(name: string): number | null {
    const match = name.match(/:(\d+(?:\.\d+)?)([mb])$/i);
    if (!match) {
        return null;
    }
    const value = Number.parseFloat(match[1]);
    if (!Number.isFinite(value)) {
        return null;
    }
    return match[2].toLowerCase() === "m" ? value / 1000 : value;
}

function isReasoningModel(model: LocalModelCard): boolean {
    const value = `${model.name} ${model.family} ${model.description}`.toLowerCase();
    return value.includes("reasoning");
}

function isSmallModel(model: LocalModelCard): boolean {
    const size = parseModelSizeInBillions(model.name);
    return size !== null && size <= 4;
}

function isExtractionModel(model: LocalModelCard): boolean {
    const value = `${model.name} ${model.family} ${model.description}`.toLowerCase();
    const extractionKeywords = [
        "extract",
        "parsing",
        "parser",
        "structured",
        "compact",
        "lightweight",
        "low-latency",
        "smollm",
    ];
    return extractionKeywords.some((keyword) => value.includes(keyword)) || isSmallModel(model);
}

export function ModelConfigPage(): React.JSX.Element {
    const { state, updateDiliAgent } = useAppState();
    const { settings, isPulling } = state.diliAgent;

    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [localModels, setLocalModels] = useState<ModelConfigStateResponse["local_models"]>([]);
    const [cloudChoices, setCloudChoices] = useState<CloudModelChoices>(CLOUD_MODEL_CHOICES);
    const [modelSearchQuery, setModelSearchQuery] = useState("");
    const [statusMessage, setStatusMessage] = useState("");
    const [openProviderModal, setOpenProviderModal] = useState<AccessKeyProvider | null>(null);
    const [activeFilters, setActiveFilters] = useState<Record<ModelFilterKey, boolean>>({
        installed: false,
        reasoning: false,
        small: false,
        extraction: false,
    });
    const [draftConfig, setDraftConfig] = useState<DraftRuntimeConfig>(
        () => resolveDraftFromSettings(settings, CLOUD_MODEL_CHOICES),
    );

    const localSelectionDisabled = isSaving || isLoading;
    const ollamaControlsDisabled = isSaving || isLoading || isPulling;
    const availableLocalModelCount = useMemo(
        () => localModels.filter((model) => model.available_in_ollama).length,
        [localModels],
    );

    const filteredLocalModels = useMemo(() => {
        const query = modelSearchQuery.trim().toLowerCase();
        return localModels.filter((model) => {
            const haystack = `${model.name} ${model.family} ${model.description}`.toLowerCase();
            if (query && !haystack.includes(query)) {
                return false;
            }
            if (activeFilters.installed && !model.available_in_ollama) {
                return false;
            }
            if (activeFilters.reasoning && !isReasoningModel(model)) {
                return false;
            }
            if (activeFilters.small && !isSmallModel(model)) {
                return false;
            }
            if (activeFilters.extraction && !isExtractionModel(model)) {
                return false;
            }
            return true;
        });
    }, [activeFilters, localModels, modelSearchQuery]);

    const applyConfigToState = (
        payload: ModelConfigStateResponse,
        options: { syncDraft?: boolean } = {},
    ) => {
        setLocalModels(payload.local_models || []);
        const choices = resolveCloudChoices(payload.cloud_model_choices);
        setCloudChoices(choices);

        const nextSettings: RuntimeSettings = buildRuntimeSettingsFromConfig(payload, settings);
        updateDiliAgent({ settings: nextSettings });
        if (options.syncDraft !== false) {
            setDraftConfig(resolveDraftFromSettings(nextSettings, choices));
        }
    };

    const loadModelConfig = async (options: { syncDraft?: boolean } = {}) => {
        setIsLoading(true);
        try {
            const payload = await fetchModelConfigState();
            applyConfigToState(payload, options);
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

    const persistConfigPatch = async (patch: ModelConfigUpdateRequest, options: PersistOptions = {}) => {
        setIsSaving(true);
        try {
            const payload = await updateModelConfigState(patch);
            applyConfigToState(payload, { syncDraft: options.syncDraft });
            setStatusMessage(options.successMessage || "");
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

    const cloudProviderOptions = useMemo(() => {
        const providers = providerOptions.filter((provider) => provider === "openai" || provider === "gemini");
        if (providers.length) {
            return providers;
        }
        return ["openai", "gemini"];
    }, [providerOptions]);

    const draftProvider = resolveProvider(draftConfig.provider, cloudChoices);
    const draftCloudModel = resolveCloudModel(draftProvider, draftConfig.cloudModel, cloudChoices);
    const activeCloudModels = cloudChoices[draftProvider] || [];

    useEffect(() => {
        setDraftConfig((previous) => {
            const nextProvider = resolveProvider(previous.provider, cloudChoices);
            const nextCloudModel = resolveCloudModel(nextProvider, previous.cloudModel, cloudChoices);
            if (previous.provider === nextProvider && previous.cloudModel === nextCloudModel) {
                return previous;
            }
            return {
                ...previous,
                provider: nextProvider,
                cloudModel: nextCloudModel,
            };
        });
    }, [cloudChoices]);

    const statusTone = resolveStatusTone(statusMessage);
    const extraControlsDisabled = isSaving || isLoading;

    const selectedClinicalModel = useMemo(
        () => localModels.find((model) => model.name === draftConfig.clinicalModel) || null,
        [draftConfig.clinicalModel, localModels],
    );

    const selectedTextExtractionModel = useMemo(
        () => localModels.find((model) => model.name === draftConfig.parsingModel) || null,
        [draftConfig.parsingModel, localModels],
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

    const missingRequiredModels = useMemo(() => {
        if (draftConfig.useCloudServices) {
            return [];
        }
        const modelMap = new Map(localModels.map((model) => [model.name, model]));
        const missing = new Set<string>();
        for (const modelName of [draftConfig.clinicalModel, draftConfig.parsingModel]) {
            const candidate = modelName.trim();
            if (!candidate) {
                continue;
            }
            const localModel = modelMap.get(candidate);
            if (!localModel || !localModel.available_in_ollama) {
                missing.add(candidate);
            }
        }
        return Array.from(missing);
    }, [draftConfig.clinicalModel, draftConfig.parsingModel, draftConfig.useCloudServices, localModels]);

    const savedProvider = resolveProvider(settings.provider, cloudChoices);
    const savedCloudModel = resolveCloudModel(savedProvider, settings.cloudModel, cloudChoices);

    const hasPendingChanges =
        draftConfig.useCloudServices !== settings.useCloudServices
        || draftProvider !== savedProvider
        || (draftCloudModel || "") !== (savedCloudModel || "")
        || draftConfig.clinicalModel !== settings.clinicalModel
        || draftConfig.parsingModel !== settings.parsingModel;

    const saveDisabled = isLoading
        || isSaving
        || !hasPendingChanges
        || (draftConfig.useCloudServices && !draftCloudModel);

    const handleRoleSelection = (role: "clinical" | "text_extraction", modelName: string) => {
        if (role === "clinical") {
            setDraftConfig((previous) => ({ ...previous, clinicalModel: modelName }));
            return;
        }
        setDraftConfig((previous) => ({ ...previous, parsingModel: modelName }));
    };

    const handleCloudSwitchChange = (value: boolean) => {
        setDraftConfig((previous) => ({ ...previous, useCloudServices: value }));
    };

    const handleProviderChange = (provider: string) => {
        const resolvedProvider = resolveProvider(provider, cloudChoices);
        const cloudModel = resolveCloudModel(resolvedProvider, null, cloudChoices);
        setDraftConfig((previous) => ({
            ...previous,
            provider: resolvedProvider,
            cloudModel,
        }));
    };

    const handleCloudModelChange = (modelName: string) => {
        setDraftConfig((previous) => ({ ...previous, cloudModel: modelName || null }));
    };

    const handleReasoningChange = async (enabled: boolean) => {
        await persistConfigPatch(
            { ollama_reasoning: enabled },
            { syncDraft: false, successMessage: "Extra parameters saved." },
        );
    };

    const handlePullModel = async (requestedModelName: string) => {
        const candidate = requestedModelName.trim();
        if (!candidate) {
            setStatusMessage("[ERROR] Enter a model name to pull from Ollama.");
            return;
        }

        updateDiliAgent({ isPulling: true });
        setStatusMessage(`[INFO] Pulling '${candidate}' from Ollama...`);
        try {
            const result = await pullModels([candidate]);
            setStatusMessage(result.message);
            await loadModelConfig({ syncDraft: false });
        } finally {
            updateDiliAgent({ isPulling: false });
        }
    };

    const handleInstallRequiredModels = async () => {
        if (!missingRequiredModels.length) {
            return;
        }

        updateDiliAgent({ isPulling: true });
        setStatusMessage(`[INFO] Installing required models: ${missingRequiredModels.join(", ")}.`);
        try {
            const result = await pullModels(missingRequiredModels);
            setStatusMessage(result.message);
            await loadModelConfig({ syncDraft: false });
        } finally {
            updateDiliAgent({ isPulling: false });
        }
    };

    const handleSaveConfiguration = async () => {
        await persistConfigPatch(
            {
                use_cloud_services: draftConfig.useCloudServices,
                llm_provider: draftProvider,
                cloud_model: draftCloudModel,
                clinical_model: draftConfig.clinicalModel || null,
                text_extraction_model: draftConfig.parsingModel || null,
            },
            { successMessage: "Configuration saved." },
        );
    };

    const noLocalModelMessage = useMemo(() => {
        if (isLoading) {
            return "Loading local model catalog...";
        }
        if (!localModels.length) {
            return "No local model catalog entries available.";
        }
        if (!filteredLocalModels.length) {
            if (modelSearchQuery.trim()) {
                return `No models match "${modelSearchQuery.trim()}".`;
            }
            return "No models match the active filters.";
        }
        return "";
    }, [filteredLocalModels.length, isLoading, localModels.length, modelSearchQuery]);

    const runtimeLabel = draftConfig.useCloudServices
        ? `Cloud (${PROVIDER_LABELS[draftProvider] || draftProvider})`
        : "Local (Ollama)";

    const localInstallSummary = availableLocalModelCount
        ? `${availableLocalModelCount} installed in Ollama.`
        : "No models installed. Install a model to continue.";

    return (
        <main className="page-container model-config-page">
            <header className="page-header">
                <p className="eyebrow">DILIGENT Clinical Copilot</p>
                <h1>Model Configurations</h1>
                <p className="lede">Adjust runtime preferences for DILI analysis.</p>
            </header>

            <div className="model-config-layout">
                <div className="model-config-main-column">
                    <section className="model-config-runtime-row" aria-label="Runtime source selector">
                        <h2 className="modal-section-title">Runtime Source</h2>
                        <div className="model-config-runtime-options">
                            <label
                                className={`model-config-runtime-option ${!draftConfig.useCloudServices ? "is-active" : ""}`}
                            >
                                <input
                                    type="radio"
                                    name="runtime-source"
                                    checked={!draftConfig.useCloudServices}
                                    onChange={() => handleCloudSwitchChange(false)}
                                    disabled={isSaving || isLoading}
                                />
                                <span>
                                    <span className="field-label">Local Models (Ollama)</span>
                                    <span className="field-helper">Run both model roles using local Ollama models.</span>
                                </span>
                            </label>
                            <label
                                className={`model-config-runtime-option ${draftConfig.useCloudServices ? "is-active" : ""}`}
                            >
                                <input
                                    type="radio"
                                    name="runtime-source"
                                    checked={draftConfig.useCloudServices}
                                    onChange={() => handleCloudSwitchChange(true)}
                                    disabled={isSaving || isLoading}
                                />
                                <span>
                                    <span className="field-label">Cloud Models (OpenAI / Gemini)</span>
                                    <span className="field-helper">Use a cloud provider and configured cloud model.</span>
                                </span>
                            </label>
                        </div>
                    </section>

                    {!draftConfig.useCloudServices && (
                        <section className={`model-config-local-row ${localSelectionDisabled ? "is-disabled" : ""}`} aria-disabled={localSelectionDisabled}>
                            <div className="model-config-row-header">
                                <div className="model-config-row-header-top">
                                    <div>
                                        <h2 className="modal-section-title">Local Model Catalog</h2>
                                        <p className="model-config-caption">
                                            Choose one clinical model and one text extraction model.
                                        </p>
                                        <p className="model-config-installed-note">{localInstallSummary}</p>
                                    </div>
                                    <div className="model-config-search">
                                        <label className="visually-hidden" htmlFor="model-search-by-name">Search model by name</label>
                                        <input
                                            id="model-search-by-name"
                                            type="text"
                                            value={modelSearchQuery}
                                            placeholder="Search models..."
                                            onChange={(event) => setModelSearchQuery(event.target.value)}
                                            disabled={isLoading}
                                        />
                                    </div>
                                </div>
                                <div className="model-config-filter-row" role="group" aria-label="Model filters">
                                    <span className="model-config-filter-label">Filter:</span>
                                    {MODEL_FILTERS.map((filterOption) => (
                                        <button
                                            key={filterOption.key}
                                            className={`model-config-filter-pill ${activeFilters[filterOption.key] ? "is-active" : ""}`}
                                            type="button"
                                            onClick={() => {
                                                setActiveFilters((previous) => ({
                                                    ...previous,
                                                    [filterOption.key]: !previous[filterOption.key],
                                                }));
                                            }}
                                            disabled={isLoading}
                                        >
                                            {filterOption.label}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="model-config-selection-section">
                                <div className="model-config-selection-header">
                                    <h3 className="modal-section-title">Clinical Model</h3>
                                    <p className="model-config-selected-line">
                                        <span>Selected: <strong>{draftConfig.clinicalModel || "Not set"}</strong></span>
                                        <span className={clinicalAvailabilityClassName}>{clinicalAvailabilityLabel}</span>
                                    </p>
                                </div>
                                <ul className="model-config-model-list">
                                    {noLocalModelMessage && (
                                        <li className="model-config-empty-note">{noLocalModelMessage}</li>
                                    )}
                                    {!noLocalModelMessage && filteredLocalModels.map((model) => {
                                        const isSelected = draftConfig.clinicalModel === model.name;
                                        return (
                                            <li
                                                key={`clinical-${model.name}`}
                                                className={`model-config-model-item ${isSelected ? "is-selected" : ""}`}
                                            >
                                                <label className="model-config-model-radio">
                                                    <input
                                                        type="radio"
                                                        name="clinical-role"
                                                        checked={isSelected}
                                                        onChange={() => handleRoleSelection("clinical", model.name)}
                                                        disabled={localSelectionDisabled}
                                                    />
                                                    <span className="model-config-model-copy">
                                                        <span className="model-config-model-name">{model.name}</span>
                                                        <span className="model-config-model-description" title={model.description}>
                                                            {model.description}
                                                        </span>
                                                    </span>
                                                </label>
                                                <div className="model-config-model-item-side">
                                                    <span
                                                        className={`model-config-availability-pill ${model.available_in_ollama ? "is-available" : "is-unavailable"}`}
                                                    >
                                                        {model.available_in_ollama ? "Installed" : "Not installed"}
                                                    </span>
                                                    <div className="model-config-model-action-row">
                                                        {!model.available_in_ollama && (
                                                            <button
                                                                className="btn btn-secondary model-config-inline-btn"
                                                                type="button"
                                                                onClick={() => { void handlePullModel(model.name); }}
                                                                disabled={ollamaControlsDisabled}
                                                            >
                                                                Install
                                                            </button>
                                                        )}
                                                        <button
                                                            className={`btn ${isSelected ? "btn-tertiary" : "btn-secondary"} model-config-inline-btn`}
                                                            type="button"
                                                            onClick={() => handleRoleSelection("clinical", model.name)}
                                                            disabled={localSelectionDisabled || isSelected}
                                                        >
                                                            {isSelected ? "Selected" : "Select"}
                                                        </button>
                                                    </div>
                                                </div>
                                            </li>
                                        );
                                    })}
                                </ul>
                            </div>

                            <div className="model-config-selection-section">
                                <div className="model-config-selection-header">
                                    <h3 className="modal-section-title">Text Extraction Model</h3>
                                    <p className="model-config-selected-line">
                                        <span>Selected: <strong>{draftConfig.parsingModel || "Not set"}</strong></span>
                                        <span className={extractionAvailabilityClassName}>{extractionAvailabilityLabel}</span>
                                    </p>
                                </div>
                                <ul className="model-config-model-list">
                                    {noLocalModelMessage && (
                                        <li className="model-config-empty-note">{noLocalModelMessage}</li>
                                    )}
                                    {!noLocalModelMessage && filteredLocalModels.map((model) => {
                                        const isSelected = draftConfig.parsingModel === model.name;
                                        return (
                                            <li
                                                key={`extraction-${model.name}`}
                                                className={`model-config-model-item ${isSelected ? "is-selected" : ""}`}
                                            >
                                                <label className="model-config-model-radio">
                                                    <input
                                                        type="radio"
                                                        name="text-extraction-role"
                                                        checked={isSelected}
                                                        onChange={() => handleRoleSelection("text_extraction", model.name)}
                                                        disabled={localSelectionDisabled}
                                                    />
                                                    <span className="model-config-model-copy">
                                                        <span className="model-config-model-name">{model.name}</span>
                                                        <span className="model-config-model-description" title={model.description}>
                                                            {model.description}
                                                        </span>
                                                    </span>
                                                </label>
                                                <div className="model-config-model-item-side">
                                                    <span
                                                        className={`model-config-availability-pill ${model.available_in_ollama ? "is-available" : "is-unavailable"}`}
                                                    >
                                                        {model.available_in_ollama ? "Installed" : "Not installed"}
                                                    </span>
                                                    <div className="model-config-model-action-row">
                                                        {!model.available_in_ollama && (
                                                            <button
                                                                className="btn btn-secondary model-config-inline-btn"
                                                                type="button"
                                                                onClick={() => { void handlePullModel(model.name); }}
                                                                disabled={ollamaControlsDisabled}
                                                            >
                                                                Install
                                                            </button>
                                                        )}
                                                        <button
                                                            className={`btn ${isSelected ? "btn-tertiary" : "btn-secondary"} model-config-inline-btn`}
                                                            type="button"
                                                            onClick={() => handleRoleSelection("text_extraction", model.name)}
                                                            disabled={localSelectionDisabled || isSelected}
                                                        >
                                                            {isSelected ? "Selected" : "Select"}
                                                        </button>
                                                    </div>
                                                </div>
                                            </li>
                                        );
                                    })}
                                </ul>
                            </div>
                        </section>
                    )}

                    {draftConfig.useCloudServices && (
                        <section className="model-config-cloud-row">
                            <h2 className="modal-section-title">Cloud Model Configuration</h2>
                            <div className="model-config-cloud-layout">
                                <div className="model-config-cloud-controls">
                                    <div className="model-config-provider-list">
                                        {cloudProviderOptions.map((provider) => (
                                            <div
                                                key={provider}
                                                className={`model-config-provider-card ${draftProvider === provider ? "is-active" : ""}`}
                                            >
                                                <button
                                                    className="model-config-provider-button"
                                                    type="button"
                                                    onClick={() => handleProviderChange(provider)}
                                                    disabled={isSaving || isLoading}
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

                                    <div className="field model-config-cloud-model-field">
                                        <label className="field-label" htmlFor="cloud-model-select">Cloud Model</label>
                                        <select
                                            id="cloud-model-select"
                                            value={draftCloudModel || ""}
                                            onChange={(e) => handleCloudModelChange(e.target.value)}
                                            disabled={isSaving || isLoading || !activeCloudModels.length}
                                        >
                                            {activeCloudModels.map((modelName) => (
                                                <option key={modelName} value={modelName}>
                                                    {modelName}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                </div>

                                <p className="model-config-cloud-description">
                                    Provider configuration appears only when cloud runtime is selected.
                                </p>
                            </div>
                        </section>
                    )}

                    <section className={`model-config-extra-row ${extraControlsDisabled ? "is-disabled" : ""}`} aria-disabled={extraControlsDisabled}>
                        <h2 className="modal-section-title">Extra Parameters (Search and RAG)</h2>
                        <div className="model-config-extra-stack">
                            <div className="model-config-provider-card model-config-provider-card-compact">
                                <button
                                    className="model-config-provider-button"
                                    type="button"
                                    onClick={() => setOpenProviderModal("tavily")}
                                    disabled={extraControlsDisabled}
                                >
                                    <span>Tavily</span>
                                    <span className="model-config-provider-hint">Research API key</span>
                                </button>
                                <button
                                    className="model-config-provider-key"
                                    type="button"
                                    onClick={() => setOpenProviderModal("tavily")}
                                    disabled={extraControlsDisabled}
                                    aria-label="Manage Tavily access keys"
                                >
                                    <KeyIcon />
                                </button>
                            </div>

                            <label className="field checkbox">
                                <input
                                    type="checkbox"
                                    checked={settings.reasoning}
                                    onChange={(e) => { void handleReasoningChange(e.target.checked); }}
                                    disabled={extraControlsDisabled}
                                />
                                <span className="field-label">Enable SDL/Reasoning</span>
                            </label>
                        </div>
                    </section>
                </div>

                <aside className="model-config-summary-panel">
                    <h2 className="modal-section-title">Current Configuration</h2>
                    <dl className="model-config-summary-list">
                        <div>
                            <dt>Runtime</dt>
                            <dd>{runtimeLabel}</dd>
                        </div>
                        <div>
                            <dt>Clinical Model</dt>
                            <dd>{draftConfig.clinicalModel || "Not set"}</dd>
                        </div>
                        <div>
                            <dt>Text Extraction</dt>
                            <dd>{draftConfig.parsingModel || "Not set"}</dd>
                        </div>
                        {draftConfig.useCloudServices && (
                            <div>
                                <dt>Cloud Model</dt>
                                <dd>{draftCloudModel || "Not set"}</dd>
                            </div>
                        )}
                    </dl>

                    {!draftConfig.useCloudServices && (
                        <div className="model-config-missing">
                            <h3 className="model-config-summary-subtitle">Missing dependencies</h3>
                            {missingRequiredModels.length ? (
                                <ul className="model-config-missing-list">
                                    {missingRequiredModels.map((modelName) => (
                                        <li key={modelName}>{modelName} not installed</li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="model-config-caption">All selected local models are installed.</p>
                            )}
                        </div>
                    )}

                    <div className="model-config-summary-actions">
                        {!draftConfig.useCloudServices && missingRequiredModels.length > 0 && (
                            <button
                                className="btn btn-secondary"
                                type="button"
                                onClick={() => { void handleInstallRequiredModels(); }}
                                disabled={isLoading || isSaving || isPulling}
                            >
                                {isPulling ? "Installing..." : "Install Required Models"}
                            </button>
                        )}
                        <button
                            className="btn btn-primary"
                            type="button"
                            onClick={() => { void handleSaveConfiguration(); }}
                            disabled={saveDisabled}
                        >
                            {isSaving ? "Saving..." : "Save Configuration"}
                        </button>
                    </div>
                </aside>
            </div>

            <StatusMessage message={statusMessage} tone={statusTone} />
            <AccessKeyModal
                isOpen={openProviderModal !== null}
                provider={openProviderModal ?? "openai"}
                providerLabel={PROVIDER_LABELS[openProviderModal ?? "openai"] || (openProviderModal ?? "openai")}
                onClose={() => setOpenProviderModal(null)}
            />
        </main>
    );
}
