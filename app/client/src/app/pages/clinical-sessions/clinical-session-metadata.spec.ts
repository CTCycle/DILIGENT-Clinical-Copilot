import { describe, expect, it } from 'vitest';

import {
  normalizeClinicalSessionMetadata,
  readMetadataEntries,
} from './clinical-session-metadata';

describe('normalizeClinicalSessionMetadata', () => {
  it('preserves unrelated metadata while enforcing collection and object contracts', () => {
    expect(normalizeClinicalSessionMetadata({
      documents: 'invalid',
      images: null,
      manual_metadata: [],
      source: 'manual',
    })).toEqual({
      documents: [],
      images: [],
      manual_metadata: {},
      source: 'manual',
    });
  });

  it('preserves valid metadata collections and manual fields', () => {
    const documents = [{ file_name: 'report.pdf' }];
    const images = ['scan.png'];
    const manualMetadata = { reviewer: 'Clinician' };

    expect(normalizeClinicalSessionMetadata({
      documents,
      images,
      manual_metadata: manualMetadata,
    })).toEqual({
      documents,
      images,
      manual_metadata: manualMetadata,
    });
  });

  it('rejects non-object JSON before reading metadata entries', () => {
    expect(readMetadataEntries('null', 'documents')).toEqual([]);
    expect(readMetadataEntries('["report.pdf"]', 'documents')).toEqual([]);
  });
});
