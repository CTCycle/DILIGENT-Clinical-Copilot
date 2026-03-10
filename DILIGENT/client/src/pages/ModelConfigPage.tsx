import React, { useEffect, useMemo, useState } from "react";

import { AccessKeyModal } from "../components/AccessKeyModal";
import { ProviderAccessCard } from "../components/ProviderAccessCard";
import { StatusMessage, resolveStatusTone } from "../components/StatusMessage";
import { CLOUD_MODEL_CHOICES } from "../constants";
import { useAppState } from "../context/AppStateContext";
import { useModelPullActions } from "../hooks/useModelPullActions";
import {
    CloudModelChoices,
    buildRuntimeSettingsFromConfig,
    resolveCloudChoices,
    resolveCloudModel,
    resolveProvider,
} from "../modelConfig";
import {
    fetchModelConfigState,
    updateModelConfigState,
} from "../services/api";
import {
    AccessKeyProvider,
    CloudProvider,
    LocalModelCard,
    ModelConfigStateResponse,
    ModelConfigUpdateRequest,
    RuntimeSettings,
} from "../types";

const DEFAULT_CLOUD_PROVIDERS: readonly CloudProvider[] = ["openai", "gemini"];

const PROVIDER_LABELS: Record<AccessKeyProvider, string> = {
    openai: "OpenAI",
    gemini: "Gemini",
    tavily: "Tavily",
};

type ModelFilterKey = "installed" | "reasoning" | "small" | "extraction";

type DraftRuntimeConfig = {
    useCloudServices: boolean;
    provider: CloudProvider;
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

function isCloudProvider(provider: string): provider is CloudProvider {
    return provider === "openai" || provider === "gemini";
}

function isAccessKeyProvider(provider: string): provider is AccessKeyProvider {
    return provider === "openai" || provider === "gemini" || provider === "tavily";
}

function resolveProviderLabel(provider: string): string {
    if (isAccessKeyProvider(provider)) {
        return PROVIDER_LABELS[provider];
    }
    return provider;
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

const ClinicalRoleIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 21s-6.8-4.9-9-8a4.8 4.8 0 0 1 7-6l2 2 2-2a4.8 4.8 0 0 1 7 6c-2.2 3.1-9 8-9 8Z" />
    </svg>
);

const TextExtractionRoleIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
        <path d="M14 3v6h6" />
        <path d="M8 13h8" />
        <path d="M8 17h5" />
    </svg>
);

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
        return [...DEFAULT_CLOUD_PROVIDERS];
    }, [cloudChoices]);

    const cloudProviderOptions = useMemo<CloudProvider[]>(() => {
        const providers = providerOptions.filter(isCloudProvider);
        if (providers.length) {
            return providers;
        }
        return [...DEFAULT_CLOUD_PROVIDERS];
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

    const { pullModelByName, installRequiredModels } = useModelPullActions({
        setPulling: (nextValue) => updateDiliAgent({ isPulling: nextValue }),
        setStatusMessage,
        loadModelConfig,
    });

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

    const handleProviderChange = (provider: CloudProvider) => {
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
        ? `Cloud (${resolveProviderLabel(draftProvider)})`
        : "Local (Ollama)";

    const localInstallSummary = availableLocalModelCount
        ? `${availableLocalModelCount} installed in Ollama.`
        : "No models installed. Install a model to continue.";

    return (
        <main className="page-container model-config-page">
            <header className="page-header">
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
                                    <h3 className="modal-section-title">Model Roles</h3>
                                    <p className="model-config-caption">Use row icons to assign clinical and extraction roles.</p>
                                </div>
                                <div className="model-config-selection-summary" role="group" aria-label="Selected local model roles">
                                    <div className="model-config-selected-card">
                                        <p className="model-config-selected-card-label">Clinical Model</p>
                                        <p className="model-config-selected-line">
                                            <span>Selected: <strong>{draftConfig.clinicalModel || "Not set"}</strong></span>
                                            <span className={clinicalAvailabilityClassName}>{clinicalAvailabilityLabel}</span>
                                        </p>
                                    </div>
                                    <div className="model-config-selected-card">
                                        <p className="model-config-selected-card-label">Text Extraction Model</p>
                                        <p className="model-config-selected-line">
                                            <span>Selected: <strong>{draftConfig.parsingModel || "Not set"}</strong></span>
                                            <span className={extractionAvailabilityClassName}>{extractionAvailabilityLabel}</span>
                                        </p>
                                    </div>
                                </div>
                                <ul className="model-config-model-list">
                                    {noLocalModelMessage && (
                                        <li className="model-config-empty-note">{noLocalModelMessage}</li>
                                    )}
                                    {!noLocalModelMessage && filteredLocalModels.map((model) => {
                                        const isClinicalSelected = draftConfig.clinicalModel === model.name;
                                        const isTextExtractionSelected = draftConfig.parsingModel === model.name;
                                        const isSelected = isClinicalSelected || isTextExtractionSelected;
                                        return (
                                            <li
                                                key={model.name}
                                                className={`model-config-model-item ${isSelected ? "is-selected" : ""}`}
                                            >
                                                <div className="model-config-model-details">
                                                    <div className="model-config-model-name-row">
                                                        <span className="model-config-model-name">{model.name}</span>
                                                        <span
                                                            className={`model-config-availability-pill ${model.available_in_ollama ? "is-available" : "is-unavailable"}`}
                                                        >
                                                            {model.available_in_ollama ? "Installed" : "Not installed"}
                                                        </span>
                                                    </div>
                                                    <span className="model-config-model-description" title={model.description}>
                                                        {model.description}
                                                    </span>
                                                    <div className="model-config-role-pill-row" aria-live="polite">
                                                        {isClinicalSelected && (
                                                            <span className="model-config-role-pill">Clinical</span>
                                                        )}
                                                        {isTextExtractionSelected && (
                                                            <span className="model-config-role-pill">Text extraction</span>
                                                        )}
                                                    </div>
                                                </div>
                                                <div className="model-config-model-item-side">
                                                    <div className="model-config-model-action-row">
                                                        {!model.available_in_ollama && (
                                                            <button
                                                                className="btn btn-secondary model-config-inline-btn"
                                                                type="button"
                                                                onClick={() => { void pullModelByName(model.name); }}
                                                                disabled={ollamaControlsDisabled}
                                                            >
                                                                Install
                                                            </button>
                                                        )}
                                                        <button
                                                            className={`access-key-action model-config-role-action ${isClinicalSelected ? "is-active" : ""}`}
                                                            type="button"
                                                            onClick={() => handleRoleSelection("clinical", model.name)}
                                                            disabled={localSelectionDisabled || isClinicalSelected}
                                                            aria-pressed={isClinicalSelected}
                                                            aria-label={isClinicalSelected ? "Clinical model selected" : `Set ${model.name} as clinical model`}
                                                            title={isClinicalSelected ? "Clinical model selected" : "Set as clinical model"}
                                                        >
                                                            <ClinicalRoleIcon />
                                                        </button>
                                                        <button
                                                            className={`access-key-action model-config-role-action ${isTextExtractionSelected ? "is-active" : ""}`}
                                                            type="button"
                                                            onClick={() => handleRoleSelection("text_extraction", model.name)}
                                                            disabled={localSelectionDisabled || isTextExtractionSelected}
                                                            aria-pressed={isTextExtractionSelected}
                                                            aria-label={isTextExtractionSelected ? "Text extraction model selected" : `Set ${model.name} as text extraction model`}
                                                            title={isTextExtractionSelected ? "Text extraction model selected" : "Set as text extraction model"}
                                                        >
                                                            <TextExtractionRoleIcon />
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
                                            <ProviderAccessCard
                                                key={provider}
                                                variant="selectable"
                                                label={PROVIDER_LABELS[provider]}
                                                isActive={draftProvider === provider}
                                                disabled={isSaving || isLoading}
                                                onSelect={() => handleProviderChange(provider)}
                                                onManageKeys={() => setOpenProviderModal(provider)}
                                                manageKeyAriaLabel={`Manage ${PROVIDER_LABELS[provider]} access keys`}
                                            />
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
                            <ProviderAccessCard
                                variant="compact"
                                label={PROVIDER_LABELS.tavily}
                                hint="Research API key"
                                disabled={extraControlsDisabled}
                                onManageKeys={() => setOpenProviderModal("tavily")}
                                manageKeyAriaLabel="Manage Tavily access keys"
                            />

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
                                onClick={() => { void installRequiredModels(missingRequiredModels); }}
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
                providerLabel={resolveProviderLabel(openProviderModal ?? "openai")}
                onClose={() => setOpenProviderModal(null)}
            />
        </main>
    );
}

