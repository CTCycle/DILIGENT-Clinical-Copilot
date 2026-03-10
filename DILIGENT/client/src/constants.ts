import { ClinicalFormState, CloudProvider, RuntimeSettings } from "./types";

const apiBaseEnv = (import.meta.env.VITE_API_BASE_URL || "").trim();
const devBase = "/api";
const selectedBase =
  apiBaseEnv.startsWith("/") && apiBaseEnv.length > 1 ? apiBaseEnv : devBase;
export const API_BASE_URL = selectedBase.replace(/\/+$/, "");

export const PARSING_MODEL_CHOICES = [
  "qwen3:1.7b",
  "qwen3:8b",
  "qwen3:14b",
  "llama3.1:8b",
  "mistral-nemo:12b",
  "gemma2:9b",
  "phi3.5:mini",
  "phi3:medium",
];

export const CLINICAL_MODEL_CHOICES = [
  "gpt-oss:20b",
  "llama3.1:8b",
  "llama3.1:70b",
  "phi3.5:mini",
  "phi3.5:moe",
  "deepseek-r1:14b",
  "alibayram/medgemma:4b",
  "alibayram/medgemma:27b",
  "gemma3:9b",
  "gemma3:27b",
];

export const CLOUD_MODEL_CHOICES: Record<CloudProvider, string[]> = {
  openai: [
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
  ],
  gemini: [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
  ],
};

export const CLOUD_PROVIDERS: CloudProvider[] = Object.keys(
  CLOUD_MODEL_CHOICES,
) as CloudProvider[];

export type LLMRuntimeDefaults = {
  parsing_model: string;
  clinical_model: string;
  llm_provider: CloudProvider;
  cloud_model: string;
  use_cloud_services: boolean;
  ollama_temperature: number;
  ollama_reasoning: boolean;
};

export const LLM_RUNTIME_DEFAULTS: Readonly<LLMRuntimeDefaults> = {
  parsing_model: "qwen3:1.7b",
  clinical_model: "gpt-oss:20b",
  llm_provider: "openai",
  cloud_model: "gpt-4o-mini",
  use_cloud_services: false,
  ollama_temperature: 0.7,
  ollama_reasoning: false,
};

function isCloudProvider(provider: string): provider is CloudProvider {
  return provider === "openai" || provider === "gemini";
}

function resolveDefaultProvider(provider: string): CloudProvider {
  const normalized = provider.trim().toLowerCase();
  if (isCloudProvider(normalized) && CLOUD_MODEL_CHOICES[normalized]) {
    return normalized;
  }
  return "openai";
}

function resolveDefaultCloudModel(
  provider: CloudProvider,
  cloudModel: string,
): string | null {
  const models = CLOUD_MODEL_CHOICES[provider] || [];
  if (!models.length) {
    return null;
  }
  if (cloudModel && models.includes(cloudModel)) {
    return cloudModel;
  }
  return models[0];
}

const DEFAULT_PROVIDER = resolveDefaultProvider(LLM_RUNTIME_DEFAULTS.llm_provider);

export const DEFAULT_SETTINGS: RuntimeSettings = {
  useCloudServices: LLM_RUNTIME_DEFAULTS.use_cloud_services,
  provider: DEFAULT_PROVIDER,
  cloudModel: resolveDefaultCloudModel(
    DEFAULT_PROVIDER,
    LLM_RUNTIME_DEFAULTS.cloud_model,
  ),
  parsingModel: LLM_RUNTIME_DEFAULTS.parsing_model,
  clinicalModel: LLM_RUNTIME_DEFAULTS.clinical_model,
  temperature: LLM_RUNTIME_DEFAULTS.ollama_temperature,
  reasoning: LLM_RUNTIME_DEFAULTS.ollama_reasoning,
};

export const DEFAULT_FORM_STATE: ClinicalFormState = {
  patientName: "",
  visitDate: "",
  anamnesis: "",
  drugs: "",
  alt: "",
  altMax: "",
  alp: "",
  alpMax: "",
  useRag: false,
  useWebSearch: false,
};

export const REPORT_EXPORT_FILENAME = "clinical_report.md";

export const HTTP_TIMEOUT_SECONDS = 3600;

