export type CloudProvider = "openai" | "gemini";

export type RuntimeSettings = {
  useCloudServices: boolean;
  provider: CloudProvider;
  cloudModel: string | null;
  parsingModel: string;
  clinicalModel: string;
  temperature: number;
  reasoning: boolean;
};

export type LocalModelCard = {
  name: string;
  description: string;
  family: string;
  available_in_ollama: boolean;
};

export type ModelConfigStateResponse = {
  status: "success";
  local_models: LocalModelCard[];
  cloud_model_choices: Partial<Record<CloudProvider, string[]>>;
  use_cloud_services: boolean;
  llm_provider: CloudProvider;
  cloud_model: string | null;
  clinical_model: string | null;
  text_extraction_model: string | null;
  ollama_reasoning: boolean;
  updated_at: string | null;
};

export type ModelConfigUpdateRequest = {
  use_cloud_services?: boolean;
  llm_provider?: CloudProvider;
  cloud_model?: string | null;
  clinical_model?: string | null;
  text_extraction_model?: string | null;
  ollama_reasoning?: boolean;
};

export type AccessKeyProvider = "openai" | "gemini" | "tavily";

export type AccessKeyRecord = {
  id: number;
  provider: AccessKeyProvider;
  is_active: boolean;
  fingerprint: string;
  created_at: string | null;
  updated_at: string | null;
  last_used_at: string | null;
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
  useWebSearch: boolean;
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
  use_web_search: boolean;
  use_cloud_services: boolean;
  llm_provider: CloudProvider | null;
  cloud_model: string | null;
  parsing_model: string;
  clinical_model: string;
  ollama_temperature: number;
  ollama_reasoning: boolean;
};

export type ApiResult = {
  message: string;
  json: unknown;
};

export type JobType =
  | "clinical"
  | "ollama_pull"
  | "rxnav_update"
  | "livertox_update";

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type ClinicalJobResult = {
  report?: string;
  progress_stage?: string;
  progress_message?: string;
  [key: string]: unknown;
};

export type OllamaPullJobResult = {
  model?: string;
  pulled?: boolean;
  progress_status?: string;
  progress_message?: string;
  total_bytes?: number;
  completed_bytes?: number;
  [key: string]: unknown;
};

export type JobStartResponse = {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
  message: string;
  poll_interval: number;
};

export type JobStatusResponse<TJobResult extends Record<string, unknown> = ClinicalJobResult> = {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
  progress: number;
  result: TJobResult | null;
  error: string | null;
};

export type JobCancelResponse = {
  job_id: string;
  success: boolean;
  message: string;
};

export type InspectionSessionStatus = "successful" | "failed";
export type InspectionDateFilterMode = "before" | "after" | "exact";

export type InspectionSessionItem = {
  session_id: number;
  patient_name: string | null;
  session_timestamp: string | null;
  status: InspectionSessionStatus;
  total_duration: number | null;
};

export type InspectionSessionCatalogResponse = {
  items: InspectionSessionItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionSessionReportResponse = {
  session_id: number;
  report: string;
};

export type InspectionCatalogQuery = {
  search?: string;
  offset?: number;
  limit?: number;
};

export type InspectionSessionQuery = InspectionCatalogQuery & {
  status?: InspectionSessionStatus;
  date_mode?: InspectionDateFilterMode;
  date?: string;
};

export type InspectionRxNavItem = {
  drug_id: number;
  drug_name: string;
  last_update: string | null;
};

export type InspectionRxNavCatalogResponse = {
  items: InspectionRxNavItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionAliasEntry = {
  alias: string;
  alias_kind: string;
};

export type InspectionAliasGroup = {
  source: string;
  aliases: InspectionAliasEntry[];
};

export type InspectionDrugAliasesResponse = {
  drug_id: number;
  drug_name: string;
  groups: InspectionAliasGroup[];
};

export type InspectionLiverToxItem = {
  drug_id: number;
  drug_name: string;
  last_update: string | null;
};

export type InspectionLiverToxCatalogResponse = {
  items: InspectionLiverToxItem[];
  total: number;
  offset: number;
  limit: number;
};

export type InspectionLiverToxExcerptResponse = {
  drug_id: number;
  drug_name: string;
  excerpt: string;
  last_update: string | null;
};

export type InspectionDeleteResponse = {
  deleted: boolean;
};

