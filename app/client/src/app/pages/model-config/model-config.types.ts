// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { CloudProvider, RagSettings } from '../../core/models/types';

export type ModelFilterKey = 'installed' | 'missing' | 'small' | 'large' | 'quantized';

export type ModelRole = 'clinical' | 'text_extraction' | 'revision' | 'timeline';

export type DraftRuntimeConfig = {
  useCloudServices: boolean;
  provider: CloudProvider;
  cloudModel: string | null;
  clinicalModel: string;
  textExtractionModel: string;
  revisionModel: string;
  timelineModel: string;
};

export type DraftRagSettings = RagSettings;

export type RagSettingsSectionKey =
  | 'general'
  | 'chunking'
  | 'embeddings'
  | 'models'
  | 'storage';
