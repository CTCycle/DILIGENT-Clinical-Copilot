import { CloudProvider, JobStatus } from '../../core/models/types';

export type ModelFilterKey = 'installed' | 'missing' | 'small' | 'large' | 'quantized';

export type ModelRole = 'clinical' | 'text_extraction';

export type DraftRuntimeConfig = {
  useCloudServices: boolean;
  provider: CloudProvider;
  cloudModel: string | null;
  clinicalModel: string;
  textExtractionModel: string;
  temperature: number;
};

export type ModelPullProgressState = {
  progress: number;
  status: JobStatus;
  message: string;
};
