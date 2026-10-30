// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { JobStatus } from '../../core/models/types';

export const ANALYSIS_PHASES = [
  'Validating clinical information',
  'Extracting medications',
  'Normalizing drug names',
  'Extracting conditions & laboratory results',
  'Calculating clinical indicators',
  'Retrieving evidence',
  'Evaluating DILI & generating the assessment',
  'Finalizing report',
] as const;

const PHASE_KEYS: ReadonlyArray<readonly [string, number]> = [
  ['preflight.', 0],
  ['sections.', 0],
  ['assessment.', 0],
  ['session_initialization', 0],
  ['drugs.extracting', 1],
  ['therapy_extraction', 1],
  ['anamnesis_extraction', 1],
  ['drugs.resolving', 2],
  ['diseases.', 3],
  ['anamnesis_disease_extraction', 3],
  ['labs.', 3],
  ['anamnesis_lab_extraction', 3],
  ['pattern.', 4],
  ['hepatotoxicity_pattern', 4],
  ['candidates.', 4],
  ['rucam.initial', 4],
  ['rucam_estimation', 4],
  ['retrieval.', 5],
  ['rag_query_building', 5],
  ['evidence_lookup.', 5],
  ['rucam.refined', 5],
  ['report.', 6],
  ['report_composition', 6],
  ['session.', 7],
  ['finalization', 7],
  ['completed', 7],
];

/** Maps a backend `progress_stage` key to a zero-based phase index, or null when unknown. */
export function resolvePhaseIndex(stage: string | null): number | null {
  if (!stage) {
    return null;
  }
  const match = PHASE_KEYS.find(([key]) => stage.startsWith(key));
  return match ? match[1] : null;
}

export function formatElapsed(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

export type AnalysisProgressTone = 'running' | 'completed' | 'failed' | 'cancelled';

export function resolveProgressTone(status: JobStatus | null): AnalysisProgressTone {
  return status === 'completed' || status === 'failed' || status === 'cancelled' ? status : 'running';
}
