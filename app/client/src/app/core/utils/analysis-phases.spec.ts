// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { describe, expect, it } from 'vitest';

import { resolvePhaseIndex } from './analysis-phases';

describe('resolvePhaseIndex', () => {
  it.each([
    ['preflight.validated', 0],
    ['session_initialization', 0],
    ['drugs.extracting', 1],
    ['therapy_extraction', 1],
    ['drugs.resolving', 2],
    ['labs.extracting', 3],
    ['anamnesis_disease_extraction', 3],
    ['rucam.initial', 4],
    ['hepatotoxicity_pattern', 4],
    ['retrieval.evidence', 5],
    ['evidence_lookup.llm_only', 5],
    ['rucam.refined', 5],
    ['report.generating', 6],
    ['session.saving', 7],
    ['completed', 7],
  ])('maps %s to phase %i', (stage, index) => {
    expect(resolvePhaseIndex(stage)).toBe(index);
  });

  it('returns null for unknown or missing keys', () => {
    expect(resolvePhaseIndex('something.new')).toBeNull();
    expect(resolvePhaseIndex(null)).toBeNull();
  });
});
