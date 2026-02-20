export type RuntimeSettings = {
  useCloudServices: boolean;
  provider: string;
  cloudModel: string | null;
  parsingModel: string;
  clinicalModel: string;
  temperature: number;
  reasoning: boolean;
};

export type LocalModelCard = {
  name: string;
  description: string;
};

export type ModelConfigStateResponse = {
  status: "success";
  local_models: LocalModelCard[];
  cloud_model_choices: Record<string, string[]>;
  use_cloud_services: boolean;
  llm_provider: string;
  cloud_model: string | null;
  clinical_model: string | null;
  text_extraction_model: string | null;
  ollama_reasoning: boolean;
  updated_at: string | null;
};

export type ModelConfigUpdateRequest = {
  use_cloud_services?: boolean;
  llm_provider?: string;
  cloud_model?: string | null;
  clinical_model?: string | null;
  text_extraction_model?: string | null;
  ollama_reasoning?: boolean;
};

export type ClinicalFormState = {
  patientName: string;
  visitDate: string;
  anamnesis: string;
  drugs: string;
  alt: string;
  altMax: string;
  alp: string;
  alpMax: string;
  useRag: boolean;
};

export type ClinicalRequestPayload = {
  name: string | null;
  visit_date: { day: number; month: number; year: number } | null;
  anamnesis: string | null;
  drugs: string | null;
  alt: string | null;
  alt_max: string | null;
  alp: string | null;
  alp_max: string | null;
  allow_missing_labs?: boolean | null;
  use_rag: boolean;
  use_cloud_services: boolean;
  llm_provider: string | null;
  cloud_model: string | null;
  parsing_model: string;
  clinical_model: string;
  ollama_temperature: number;
  ollama_reasoning: boolean;
};

export type ApiResult = {
  message: string;
  json: unknown | null;
};

export type JobStartResponse = {
  job_id: string;
  job_type: string;
  status: string;
  message: string;
  poll_interval: number;
};

export type JobStatusResponse = {
  job_id: string;
  job_type: string;
  status: string;
  progress: number;
  result: Record<string, unknown> | null;
  error: string | null;
};

export type JobCancelResponse = {
  job_id: string;
  success: boolean;
  message: string;
};
