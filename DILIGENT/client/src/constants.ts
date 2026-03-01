import { ClinicalFormState, RuntimeSettings } from "./types";

const apiBaseEnv = (import.meta.env.VITE_API_BASE_URL || "").trim();
const absoluteBase = /^https?:\/\//i.test(apiBaseEnv);
const devBase = "/api";
const selectedBase =
  import.meta.env.DEV && absoluteBase ? devBase : apiBaseEnv || devBase;
export const API_BASE_URL = selectedBase.replace(/\/+$/, "");

const ollamaUrlEnv = (import.meta.env.VITE_OLLAMA_URL || "").trim();
const ollamaHostEnv = (import.meta.env.VITE_OLLAMA_HOST || "localhost").trim() || "localhost";
const parsedOllamaPort = Number.parseInt(import.meta.env.VITE_OLLAMA_PORT || "11434", 10);
export const OLLAMA_PORT = Number.isFinite(parsedOllamaPort) && parsedOllamaPort > 0
  ? parsedOllamaPort
  : 11434;
export const OLLAMA_HOST = ollamaHostEnv;
export const OLLAMA_BASE_URL = (
  ollamaUrlEnv || `http://${OLLAMA_HOST}:${OLLAMA_PORT}`
).replace(/\/+$/, "");

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

export const CLOUD_PROVIDERS = Object.keys(CLOUD_MODEL_CHOICES);

export const DEFAULT_SETTINGS: RuntimeSettings = {
  useCloudServices: false,
  provider: "openai",
  cloudModel: "gpt-5.2",
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

export const HTTP_TIMEOUT_SECONDS = 3600;
