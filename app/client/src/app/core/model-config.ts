import { DEFAULT_SETTINGS } from "./constants";
import {
  LocalModelCard,
  CloudProvider,
  ModelConfigStateResponse,
  RuntimeSettings,
} from "./models/types";

export type CloudModelChoices = Record<CloudProvider, string[]>;

type IncomingCloudModelChoices = Partial<Record<CloudProvider, string[]>>;

const CLOUD_PROVIDER_IDS: readonly CloudProvider[] = ["openai", "gemini", "deepseek", "anthropic", "opencode_zen", "opencode_go"];
function isCloudProvider(provider: string): provider is CloudProvider {
  return CLOUD_PROVIDER_IDS.includes(provider as CloudProvider);
}

export function resolveCloudChoices(
  cloudChoices: IncomingCloudModelChoices | null | undefined,
): CloudModelChoices {
  return Object.fromEntries(CLOUD_PROVIDER_IDS.map((id) => [id, cloudChoices?.[id] || []])) as CloudModelChoices;
}

export function resolveProvider(
  provider: string | null | undefined,
  cloudChoices: CloudModelChoices,
): CloudProvider {
  const normalized = (provider || "").trim().toLowerCase();
  if (isCloudProvider(normalized)) {
    return normalized;
  }
  return DEFAULT_SETTINGS.provider;
}

export function resolveCloudModel(
  provider: CloudProvider,
  cloudModel: string | null | undefined,
  cloudChoices: CloudModelChoices,
): string | null {
  const options = cloudChoices[provider] || [];
  if (cloudModel && options.includes(cloudModel)) {
    return cloudModel;
  }
  return null;
}

export function buildRuntimeSettingsFromConfig(
  payload: ModelConfigStateResponse,
  previous: RuntimeSettings,
): RuntimeSettings {
  const cloudChoices = resolveCloudChoices(Object.fromEntries(payload.cloud_providers.map((provider) => [provider.id, provider.models.map((model) => model.id)])));
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


