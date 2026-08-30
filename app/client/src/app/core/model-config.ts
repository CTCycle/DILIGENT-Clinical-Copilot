import {
  CloudProvider,
  ModelConfigStateResponse,
  RuntimeSettings,
} from "./models/types";

export type CloudModelChoices = Record<string, string[]>;

type IncomingCloudModelChoices = Record<string, string[]>;

export const EMPTY_RUNTIME_SETTINGS: RuntimeSettings = {
  useCloudServices: false,
  provider: "",
  cloudModel: null,
  textExtractionModel: "",
  clinicalModel: "",
  revisionModel: "",
  timelineModel: "",
  reasoning: "off",
};

export function resolveCloudChoices(
  cloudChoices: IncomingCloudModelChoices | null | undefined,
): CloudModelChoices {
  return Object.fromEntries(
    Object.entries(cloudChoices || {}).map(([provider, models]) => [
      provider,
      Array.isArray(models) ? models.filter((model) => typeof model === "string") : [],
    ]),
  );
}

export function resolveProvider(
  provider: string | null | undefined,
  cloudChoices: CloudModelChoices,
): CloudProvider {
  const normalized = (provider || "").trim().toLowerCase();
  if (!normalized) return "";
  const configuredProvider = Object.keys(cloudChoices).find((candidate) => candidate === normalized);
  return configuredProvider || normalized;
}

export function resolveCloudModel(
  provider: CloudProvider,
  cloudModel: string | null | undefined,
  cloudChoices: CloudModelChoices,
  preserveConfiguredModel = false,
): string | null {
  const options = cloudChoices[provider] || [];
  if (cloudModel && (preserveConfiguredModel || !options.length || options.includes(cloudModel))) {
    return cloudModel;
  }
  return null;
}

export function buildRuntimeSettingsFromConfig(
  payload: ModelConfigStateResponse,
): RuntimeSettings {
  const cloudChoices = resolveCloudChoices(Object.fromEntries(payload.cloud_providers.map((provider) => [provider.id, provider.models.map((model) => model.id)])));
  const provider = resolveProvider(payload.llm_provider, cloudChoices);
  const cloudModel = resolveCloudModel(
    provider,
    payload.cloud_model,
    cloudChoices,
    true,
  );
  return {
    useCloudServices: payload.use_cloud_services,
    provider,
    cloudModel,
    textExtractionModel: payload.text_extraction_model,
    clinicalModel: payload.clinical_model,
    revisionModel: payload.revision_model,
    timelineModel: payload.timeline_model,
    reasoning: payload.reasoning_level,
  };
}
