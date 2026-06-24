import { CLOUD_MODEL_CHOICES, DEFAULT_SETTINGS } from "./constants";
import {
  LocalModelCard,
  CloudProvider,
  ModelConfigStateResponse,
  RuntimeSettings,
} from "./models/types";

export type CloudModelChoices = Record<CloudProvider, string[]>;

type IncomingCloudModelChoices = Partial<Record<CloudProvider, string[]>>;

function isCloudProvider(provider: string): provider is CloudProvider {
  return provider === "openai" || provider === "gemini";
}

export function resolveCloudChoices(
  cloudChoices: IncomingCloudModelChoices | null | undefined,
): CloudModelChoices {
  return {
    ...CLOUD_MODEL_CHOICES,
    ...(cloudChoices || {}),
  };
}

export function resolveProvider(
  provider: string | null | undefined,
  cloudChoices: CloudModelChoices,
): CloudProvider {
  const normalized = (provider || "").trim().toLowerCase();
  if (isCloudProvider(normalized) && cloudChoices[normalized]) {
    return normalized;
  }
  if (cloudChoices.openai) {
    return "openai";
  }
  return DEFAULT_SETTINGS.provider;
}

export function resolveCloudModel(
  provider: CloudProvider,
  cloudModel: string | null | undefined,
  cloudChoices: CloudModelChoices,
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

export function buildRuntimeSettingsFromConfig(
  payload: ModelConfigStateResponse,
  previous: RuntimeSettings,
): RuntimeSettings {
  const cloudChoices = resolveCloudChoices(payload.cloud_model_choices);
  const provider = resolveProvider(payload.llm_provider ?? DEFAULT_SETTINGS.provider, cloudChoices);
  const cloudModel = resolveCloudModel(
    provider,
    payload.cloud_model,
    cloudChoices,
  );
  const resolvedClinicalModel = payload.clinical_model ?? "";
  const resolvedtextExtractionModel = payload.text_extraction_model ?? "";
  return {
    ...previous,
    useCloudServices: payload.use_cloud_services,
    provider,
    cloudModel,
    textExtractionModel: resolvedtextExtractionModel,
    clinicalModel: resolvedClinicalModel,
    temperature: payload.cloud_temperature ?? payload.ollama_temperature,
    reasoning: payload.ollama_reasoning,
  };
}

function recommendedLocalModelName(
  localModels: LocalModelCard[],
): string {
  const installed = localModels.filter((model) => model.available_in_ollama);
  const recommended = installed.find(
    (model) => model.recommended_for_local_extraction,
  );
  if (recommended) {
    return recommended.name;
  }
  return installed[0]?.name || "";
}

export function resolveLocalDraftModel(
  candidate: string | null | undefined,
  localModels: LocalModelCard[],
): string {
  const normalized = (candidate || "").trim();
  if (!normalized) {
    return recommendedLocalModelName(localModels);
  }
  const installed = localModels.find(
    (model) => model.available_in_ollama && model.name === normalized,
  );
  if (installed) {
    return installed.name;
  }
  return recommendedLocalModelName(localModels);
}


