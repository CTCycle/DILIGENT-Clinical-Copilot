// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { describe, expect, it } from 'vitest';

import { buildSectionPreviews } from './section-preview';

const SOURCE = 'Anamnesis\nPatient history.\nTherapy\nDrug A 10 mg\nLaboratory analysis\nALT 120';

function diagnostics(sections: Record<string, unknown>) {
  return { parser: { sections } };
}

function meta(span: [number, number], strategy = 'exact', requiresReview = false) {
  return {
    match_strategy: strategy,
    confidence_score: 0.9,
    body_line_span: [2, 3],
    char_span: span,
    requires_review: requiresReview,
  };
}

describe('buildSectionPreviews', () => {
  it('maps explicit and inferred sections to excerpts and textarea offsets', () => {
    const start = SOURCE.indexOf('Drug A');
    const previews = buildSectionPreviews(
      diagnostics({
        anamnesis: meta([10, 27]),
        drugs: meta([start, start + 13], 'content_inference', true),
      }),
      SOURCE,
    );
    expect(previews.map((item) => item.basis)).toEqual(['Explicit heading', 'Inferred from content', 'Not found']);
    expect(previews[1].excerpt).toBe('Drug A 10 mg');
    expect(previews[1].requiresReview).toBe(true);
    expect(previews[2].found).toBe(false);
  });

  it('offsets spans past leading whitespace and normalizes CRLF', () => {
    const raw = `\r\n  ${SOURCE.replace(/\n/g, '\r\n')}  `;
    const trimmedStart = SOURCE.indexOf('Drug A');
    const [drugs] = buildSectionPreviews(
      diagnostics({ drugs: meta([trimmedStart, trimmedStart + 12]) }),
      raw,
    ).filter((item) => item.key === 'drugs');
    const textarea = raw.replace(/\r\n/g, '\n');
    expect(textarea.slice(drugs.selection!.start, drugs.selection!.end)).toBe('Drug A 10 mg');
  });

  it('drops stale spans instead of pointing at the wrong passage', () => {
    const [anamnesis] = buildSectionPreviews(diagnostics({ anamnesis: meta([0, 9999]) }), SOURCE);
    expect(anamnesis.found).toBe(true);
    expect(anamnesis.excerpt).toBeNull();
    expect(anamnesis.selection).toBeNull();
  });

  it('returns nothing without parser diagnostics', () => {
    expect(buildSectionPreviews({}, SOURCE)).toEqual([]);
  });
});
