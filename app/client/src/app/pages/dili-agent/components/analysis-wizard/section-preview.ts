// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

export type SourceSelection = { start: number; end: number };

export type SectionPreview = {
  key: string;
  label: string;
  /** False when the parser did not locate the section. */
  found: boolean;
  basis: string;
  lines: string | null;
  confidence: number | null;
  requiresReview: boolean;
  excerpt: string | null;
  /** Range in the clinical input textarea; null when offsets cannot be trusted. */
  selection: SourceSelection | null;
};

const SECTIONS: ReadonlyArray<{ key: string; label: string }> = [
  { key: 'anamnesis', label: 'Anamnesis' },
  { key: 'drugs', label: 'Current therapy' },
  { key: 'laboratory_analysis', label: 'Laboratory analysis' },
];

const HEADING_STRATEGIES = new Set(['exact', 'phrase']);
const EXCERPT_LIMIT = 180;

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asSpan(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const [first, second] = value;
  return Number.isInteger(first) && Number.isInteger(second) ? [first, second] : null;
}

function describeBasis(strategy: unknown): string {
  if (typeof strategy !== 'string') return 'Unknown basis';
  if (HEADING_STRATEGIES.has(strategy)) return 'Explicit heading';
  if (strategy === 'content_inference' || strategy === 'fallback_assignment') return 'Inferred from content';
  return 'Approximate heading';
}

/**
 * Builds the pre-analysis section preview from the parser diagnostics returned
 * by preflight. Parser offsets refer to the trimmed, newline-normalized input,
 * so they are translated back to textarea offsets and discarded when they do not
 * fit the current text.
 */
export function buildSectionPreviews(
  diagnostics: Record<string, unknown> | undefined,
  clinicalInput: string,
): SectionPreview[] {
  const sections = asRecord(asRecord(diagnostics?.['parser'])?.['sections']);
  if (!sections) return [];
  const normalized = clinicalInput.replace(/\r\n?/g, '\n');
  const leading = normalized.length - normalized.trimStart().length;
  const source = normalized.trim();

  return SECTIONS.map(({ key, label }) => {
    const meta = asRecord(sections[key]);
    if (!meta) {
      return {
        key, label, found: false, basis: 'Not found', lines: null,
        confidence: null, requiresReview: true, excerpt: null, selection: null,
      };
    }
    const span = asSpan(meta['char_span']);
    const valid = span !== null && span[0] >= 0 && span[0] < span[1] && span[1] <= source.length;
    const lineSpan = asSpan(meta['body_line_span']);
    const confidence = meta['confidence_score'];
    return {
      key,
      label,
      found: true,
      basis: describeBasis(meta['match_strategy']),
      lines: lineSpan ? `Lines ${lineSpan[0]}–${lineSpan[1]}` : null,
      confidence: typeof confidence === 'number' ? confidence : null,
      requiresReview: meta['requires_review'] === true,
      excerpt: valid ? truncate(source.slice(span[0], span[1])) : null,
      selection: valid ? { start: span[0] + leading, end: span[1] + leading } : null,
    };
  });
}

function truncate(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > EXCERPT_LIMIT ? `${flat.slice(0, EXCERPT_LIMIT)}…` : flat;
}
