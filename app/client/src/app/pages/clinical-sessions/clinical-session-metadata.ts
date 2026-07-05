export type ClinicalSessionMetadataKey = 'documents' | 'images';

export const DEFAULT_CLINICAL_SESSION_METADATA_TEXT = '{\n  "documents": [],\n  "images": []\n}';

export function normalizeClinicalSessionMetadata(
  metadata: Record<string, unknown>,
): Record<string, unknown> {
  return {
    documents: Array.isArray(metadata['documents']) ? metadata['documents'] : [],
    images: Array.isArray(metadata['images']) ? metadata['images'] : [],
    manual_metadata:
      metadata['manual_metadata'] && typeof metadata['manual_metadata'] === 'object'
        ? metadata['manual_metadata']
        : {},
    ...metadata,
  };
}

export function readMetadataEntries(
  metadataText: string,
  key: ClinicalSessionMetadataKey,
): string[] {
  try {
    const parsed = JSON.parse(metadataText) as Record<string, unknown>;
    const values = parsed[key];
    if (!Array.isArray(values)) return [];
    return values
      .map((item) => {
        if (typeof item === 'string') return item.trim();
        if (!item || typeof item !== 'object') return '';
        const record = item as Record<string, unknown>;
        const label =
          record['title'] ||
          record['file_name'] ||
          record['name'] ||
          record['path'] ||
          record['source'];
        return typeof label === 'string' ? label.trim() : JSON.stringify(record);
      })
      .filter((item) => item.length > 0);
  } catch {
    return [];
  }
}
