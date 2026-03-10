import { CLOUD_MODEL_CHOICES, DEFAULT_SETTINGS } from "./constants";
import {
  CloudProvider,
  ModelConfigStateResponse,
  RuntimeSettings,
} from "./types";

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
  const provider = resolveProvider(payload.llm_provider, cloudChoices);
  const cloudModel = resolveCloudModel(
    provider,
    payload.cloud_model,
    cloudChoices,
  );
  return {
    ...previous,
    useCloudServices: payload.use_cloud_services,
    provider,
    cloudModel,
    parsingModel: payload.text_extraction_model || previous.parsingModel,
    clinicalModel: payload.clinical_model || previous.clinicalModel,
    reasoning: payload.ollama_reasoning,
  };
}

