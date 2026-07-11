import { ClinicalFormState, CloudProvider, RuntimeSettings } from "./models/types";

export const API_BASE_URL = "/api";

export const TEXT_EXTRACTION_MODEL_CHOICES = [
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

export type LLMRuntimeDefaults = {
  text_extraction_model: string;
  clinical_model: string;
  llm_provider: CloudProvider;
  cloud_model: string;
  use_cloud_services: boolean;
  ollama_temperature: number;
  cloud_temperature: number;
  ollama_reasoning: boolean;
};

export const LLM_RUNTIME_DEFAULTS: Readonly<LLMRuntimeDefaults> = {
  text_extraction_model: "qwen3:1.7b",
  clinical_model: "gpt-oss:20b",
  llm_provider: "openai",
  cloud_model: "gpt-4o-mini",
  use_cloud_services: false,
  ollama_temperature: 0.7,
  cloud_temperature: 0.7,
  ollama_reasoning: false,
};

export const DEFAULT_SETTINGS: RuntimeSettings = {
  useCloudServices: LLM_RUNTIME_DEFAULTS.use_cloud_services,
  provider: LLM_RUNTIME_DEFAULTS.llm_provider,
  cloudModel: null,
  textExtractionModel: LLM_RUNTIME_DEFAULTS.text_extraction_model,
  clinicalModel: LLM_RUNTIME_DEFAULTS.clinical_model,
  temperature: LLM_RUNTIME_DEFAULTS.cloud_temperature,
  reasoning: LLM_RUNTIME_DEFAULTS.ollama_reasoning,
};

export const DEFAULT_FORM_STATE: ClinicalFormState = {
  patientName: "",
  visitDate: "",
  patientImageDataUrl: null,
  clinicalInput: "",
  useRag: false,
};

export const REPORT_EXPORT_FILENAME = "clinical_report.md";

export const HTTP_TIMEOUT_SECONDS = 3600;
