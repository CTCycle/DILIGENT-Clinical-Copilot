export type RuntimeSettings = {
  useCloudServices: boolean;
  provider: string;
  cloudModel: string | null;
  parsingModel: string;
  clinicalModel: string;
  temperature: number;
  reasoning: boolean;
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

export type CloudSelection = {
  provider: string;
  models: string[];
  model: string | null;
};
