import { CLOUD_MODEL_CHOICES, DEFAULT_SETTINGS } from "./constants";
import { ModelConfigStateResponse, RuntimeSettings } from "./types";

export type CloudModelChoices = Record<string, string[]>;

export function resolveCloudChoices(
  cloudChoices: CloudModelChoices | null | undefined,
): CloudModelChoices {
  if (!cloudChoices) {
    return CLOUD_MODEL_CHOICES;
  }
  return cloudChoices;
}

export function resolveProvider(
  provider: string | null | undefined,
  cloudChoices: CloudModelChoices,
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

export function resolveCloudModel(
  provider: string,
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
