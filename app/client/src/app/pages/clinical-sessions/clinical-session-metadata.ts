import { isRecord } from '../../core/utils';

export type ClinicalSessionMetadataKey = 'documents' | 'images';

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

export function readMetadataEntries(
  metadataText: string,
  key: ClinicalSessionMetadataKey,
): string[] {
  try {
    const parsed: unknown = JSON.parse(metadataText);
    if (!isRecord(parsed)) return [];
    const values = parsed[key];
    if (!Array.isArray(values)) return [];
    return values
      .map((item) => {
        if (typeof item === 'string') return item.trim();
        if (!isRecord(item)) return '';
        const record = item;
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
