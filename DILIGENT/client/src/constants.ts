import { ClinicalFormState, RuntimeSettings } from "./types";

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

export const CLOUD_MODEL_CHOICES: Record<string, string[]> = {
  openai: ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"],
  gemini: [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.0-pro",
    "gemini-1.0-pro-vision",
  ],
};

export const CLOUD_PROVIDERS = Object.keys(CLOUD_MODEL_CHOICES);

export const DEFAULT_SETTINGS: RuntimeSettings = {
  useCloudServices: false,
  provider: "openai",
  cloudModel: "gpt-4o-mini",
  parsingModel: "qwen3:1.7b",
  clinicalModel: "gpt-oss:20b",
  temperature: 0.7,
  reasoning: false,
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
};

export const REPORT_EXPORT_FILENAME = "clinical_report.md";

export const HTTP_TIMEOUT_SECONDS = 1800;
