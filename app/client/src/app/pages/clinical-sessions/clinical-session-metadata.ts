// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

export const DEFAULT_CLINICAL_SESSION_METADATA_TEXT = '{\n  "documents": [],\n  "images": []\n}';

export function normalizeClinicalSessionMetadata(
  metadata: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...metadata,
    documents: Array.isArray(metadata['documents']) ? metadata['documents'] : [],
    images: Array.isArray(metadata['images']) ? metadata['images'] : [],
    manual_metadata:
      metadata['manual_metadata']
      && typeof metadata['manual_metadata'] === 'object'
      && !Array.isArray(metadata['manual_metadata'])
        ? metadata['manual_metadata']
        : {},
  };
}
