// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { RagSettings } from '../../core/models/types';

export type ModelFilterKey = 'installed' | 'missing' | 'small' | 'large' | 'quantized';

export type DraftRagSettings = RagSettings;

export type RagSettingsSectionKey = 'chunking' | 'embeddings' | 'storage';

export type RagNumberKey = keyof Pick<
  DraftRagSettings,
  'chunk_size' | 'chunk_overlap' | 'embedding_batch_size' | 'vector_stream_batch_size'
>;

export type RagTextKey = keyof Pick<
  DraftRagSettings,
  'vector_collection_name' | 'vector_index_metric' | 'vector_index_type'
>;

export type RagBooleanKey = keyof Pick<
  DraftRagSettings,
  'allow_local_filesystem_access' | 'embedding_offline_mode'
>;
