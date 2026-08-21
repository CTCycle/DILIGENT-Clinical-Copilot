import { ClinicalFormState, CloudProvider, ReasoningLevel, RuntimeSettings } from "./models/types";

export const API_BASE_URL = "/api";

export type LLMRuntimeDefaults = {
  text_extraction_model: string;
  clinical_model: string;
  llm_provider: CloudProvider;
  cloud_model: string;
  use_cloud_services: boolean;
  reasoning_level: ReasoningLevel;
};

export const LLM_RUNTIME_DEFAULTS: Readonly<LLMRuntimeDefaults> = {
  text_extraction_model: "qwen3:1.7b",
  clinical_model: "gpt-oss:20b",
  llm_provider: "openai",
  cloud_model: "gpt-4o-mini",
  use_cloud_services: false,
  reasoning_level: "off",
};

export const DEFAULT_SETTINGS: RuntimeSettings = {
  useCloudServices: LLM_RUNTIME_DEFAULTS.use_cloud_services,
  provider: LLM_RUNTIME_DEFAULTS.llm_provider,
  cloudModel: null,
  textExtractionModel: LLM_RUNTIME_DEFAULTS.text_extraction_model,
  clinicalModel: LLM_RUNTIME_DEFAULTS.clinical_model,
  revisionModel: LLM_RUNTIME_DEFAULTS.clinical_model,
  timelineModel: LLM_RUNTIME_DEFAULTS.text_extraction_model,
  reasoning: LLM_RUNTIME_DEFAULTS.reasoning_level,
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
