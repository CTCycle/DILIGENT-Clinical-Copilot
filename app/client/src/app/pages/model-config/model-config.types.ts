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

export type RagNumberKey = keyof Pick<
  DraftRagSettings,
  | 'chunk_size'
  | 'chunk_overlap'
  | 'embedding_batch_size'
  | 'retrieval_candidate_count'
  | 'retrieval_selected_count'
  | 'vector_stream_batch_size'
>;

export type RagFloatKey = keyof Pick<DraftRagSettings, 'hybrid_vector_weight' | 'hybrid_text_weight'>;

export type RagTextKey = keyof Pick<
  DraftRagSettings,
  'reranker_model' | 'vector_collection_name' | 'vector_index_metric' | 'vector_index_type'
>;

export type RagBooleanKey = keyof Pick<
  DraftRagSettings,
  'allow_local_filesystem_access' | 'use_hybrid_search' | 'use_reranking' | 'embedding_offline_mode'
>;
