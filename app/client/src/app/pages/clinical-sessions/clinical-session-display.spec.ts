// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { ClinicalSessionDetail } from '../../core/models/inspection-types';
import { buildPersistedDrugEvidence, previewDetectedDiseases, previewHepatotoxicityPattern, previewLabTimeline, previewReport, terminologyResolvedLabel } from './clinical-session-preview';
import { terminologyStatusLabel, terminologyStatusTone } from '../../core/utils/terminology-status';
import { describe, expect, it } from 'vitest';
import { type RevisionArtifact, type RevisionPipelineStep } from '../../core/models/revision-types';
import { formatRevisionArtifactLabel, formatRevisionArtifactMeta, formatRevisionArtifactPurpose, formatRevisionStatusLabel, formatRevisionStepLabel, formatRevisionStepMeta, formatRevisionStepSummary, revisionStatusTone } from './clinical-session-revision-display';

describe('clinical-session-preview', () => {
  const detail = (overrides: Partial<ClinicalSessionDetail> = {}): ClinicalSessionDetail => ({
    session_id: 1,
    patient_name: null,
    visit_date: null,
    session_timestamp: null,
    version: 1,
    original_session_id: null,
    status: 'completed',
    text_extraction_model: null,
    clinical_model: null,
    metadata: {},
    sections: {},
    session_text: '',
    source_clinical_text: '',
    result_payload: {},
    official_report_text: null,
    manual_edit_history: [],
    ...overrides,
  });

  it('previews only the official report text', () => {
    expect(previewReport(detail({ official_report_text: ' official report ' }))).toBe('official report');
    expect(previewReport(detail({ result_payload: { report: 'payload report' } })))
      .toBe('No AI report preview is available for this session.');
  });

  it('exposes the persisted terminology identity of detected drugs', () => {
    const rows = buildPersistedDrugEvidence(detail({
      result_payload: {
        matched_drugs: [
          {
            raw_drug_name: 'Augmentin',
            match_status: 'accepted_verified',
            terminology: {
              status: 'verified',
              accepted: true,
              method: 'exact_alias',
              drug_id: 7,
              preferred_name: 'amoxicillin / clavulanate',
              ingredients: [
                { drug_id: 1, name: 'amoxicillin', concept_type: 'ingredient' },
                { drug_id: 2, name: 'clavulanate', concept_type: 'ingredient' },
              ],
              candidates: [],
              review_reasons: [],
            },
          },
          {
            raw_drug_name: 'Hydroxazine',
            match_status: 'ambiguous_requires_review',
            terminology: {
              status: 'ambiguous',
              accepted: false,
              method: 'fuzzy',
              candidates: [
                { drug_id: 3, name: 'hydroxyzine', concept_type: 'ingredient' },
                { drug_id: 4, name: 'hydralazine', concept_type: 'ingredient' },
                { name: 'malformed candidate' },
              ],
            },
          },
          { raw_drug_name: 'Legacy drug', match_status: 'matched', terminology: { status: 'legacy' } },
        ],
      },
    }));
    const byName = new Map(rows.map((row) => [row.name, row]));

    const augmentin = byName.get('Augmentin')!.terminology!;
    expect(terminologyResolvedLabel(augmentin)).toBe('amoxicillin / clavulanate (amoxicillin + clavulanate)');
    expect(byName.get('Augmentin')!.liverTox).toBe(true);
    const ambiguous = byName.get('Hydroxazine')!.terminology!;
    expect(terminologyStatusLabel(ambiguous.status)).toBe('Ambiguous');
    expect(terminologyStatusTone(ambiguous.status)).toBe('is-warning');
    expect(terminologyResolvedLabel(ambiguous)).toBe('');
    expect(ambiguous.candidates.map((item) => item.name)).toEqual(['hydroxyzine', 'hydralazine']);
    expect(byName.get('Legacy drug')!.terminology).toBeNull();
    expect(terminologyStatusLabel(undefined)).toBe('Not resolved');
  });

  it('marks drug origins and timing from the structured case and the drugs section', () => {
    const rows = buildPersistedDrugEvidence(detail({
      sections: { drugs: 'Ramipril 5 mg daily' },
      result_payload: {
        matched_drugs: [{ raw_drug_name: 'Ramipril', rxnorm_rxcui: '35296' }],
        structured_case: {
          therapy_drugs: [{ name: 'Ramipril', therapy_start_date: '2026-08-01' }],
          anamnesis_drugs: [{ name: 'Paracetamol' }],
        },
      },
    }));

    expect(rows.map((row) => [row.name, row.inTherapy, row.inAnamnesis, row.temporalReference, row.bibliographyLabel])).toEqual([
      ['Ramipril', true, false, '2026-08-01', 'RxNav'],
      ['Paracetamol', false, true, 'time not specified', 'No backend match'],
    ]);
  });

  it('reads diseases only from the structured case', () => {
    expect(previewDetectedDiseases(detail({
      result_payload: { anamnesis_diseases: ['Unstructured diagnosis'], structured_case: ['unexpected'] },
      sections: { anamnesis: 'Past history includes diabetes.' },
      official_report_text: 'Detected diseases: hepatitis',
    }))).toEqual([]);
  });

  it('does not present negated or ruled-out diseases as detected conditions', () => {
    expect(previewDetectedDiseases(detail({
      result_payload: {
        structured_case: {
          anamnesis_diseases: [
            {
              name: 'chronic liver disease',
              diagnosis_status: 'ruled-out',
              attribution: 'negated',
            },
            { name: 'gastroesophageal reflux disease', diagnosis_status: 'confirmed' },
          ],
        },
      },
    }))).toEqual(['gastroesophageal reflux disease']);
  });

  it('reads the lab timeline and the resolved hepatic pattern from the structured payload', () => {
    const session = detail({
      result_payload: {
        extraction_metadata: { hepatic_pattern: { final_value: 'cholestatic' } },
        lab_timeline: [{ marker_name: 'ALT', value: 123, relative_time: 'day 1' }],
      },
      official_report_text: 'Hepatotoxicity pattern: mixed',
    });

    expect(previewHepatotoxicityPattern(session)).toBe('cholestatic');
    expect(previewHepatotoxicityPattern(detail({ official_report_text: 'Classification: mixed' }))).toBe('N/A');
    expect(previewLabTimeline(session)).toEqual([expect.objectContaining({
      marker: 'ALT',
      value: '123',
      timing: 'day 1',
    })]);
  });
});

describe('clinical-session-revision-display', () => {
  it('turns issue-scan and task records into readable workflow metadata', () => {
    const step: RevisionPipelineStep = {
      step_name: 'revision_agent_task_4',
      step_index: 2,
      step_count: 4,
      attempt_number: 2,
      status: 'completed',
      retry_count: 1,
      output_summary: { tool_call_count: 3 },
      pipeline_run_id: 'run-1',
      input_hash: null,
      output_hash: null,
      input_summary: null,
      output_payload: null,
      schema_name: null,
      schema_version: null,
      prompt_version: null,
      parser_version: null,
      model_provider: null,
      model_name: null,
      token_usage: null,
      latency_ms: null,
      error: null,
      started_at: null,
      completed_at: null,
      superseded_at: null,
      created_at: '2026-09-01T08:00:00Z',
      updated_at: '2026-09-01T08:00:00Z',
    };

    expect(formatRevisionStepLabel(step)).toBe('Revision task 4');
    expect(formatRevisionStepMeta(step)).toBe('Step 2 of 4 · Attempt 2 · 1 retry');
    expect(formatRevisionStepSummary(step)).toBe('3 tool calls');
    expect(formatRevisionStatusLabel(step.status)).toBe('Completed');
    expect(revisionStatusTone(step.status)).toBe('success');
  });

  it('gives persisted artifacts a useful label, purpose, and compact metadata', () => {
    const artifact: RevisionArtifact = {
      artifact_key: 'revision_agent_draft_report',
      artifact_kind: 'pipeline_artifact',
      status: 'requires_human_review',
      schema_version: '1',
      entity_type: null,
      entity_name: null,
      payload: null,
      revision_version_id: 1,
      pipeline_run_id: 'run-1',
      created_at: '2026-09-01T08:00:00Z',
      updated_at: '2026-09-01T08:00:00Z',
    };

    expect(formatRevisionArtifactLabel(artifact)).toBe('Generated draft');
    expect(formatRevisionArtifactPurpose(artifact)).toBe('Proposed report retained for human review.');
    expect(formatRevisionArtifactMeta(artifact)).toBe('Pipeline Artifact · Schema 1');
    expect(formatRevisionStatusLabel(artifact.status)).toBe('Human review required');
    expect(revisionStatusTone(artifact.status)).toBe('warning');
  });

  it('labels bounded repair stages and artifacts explicitly', () => {
    const editorStep: RevisionPipelineStep = {
      step_name: 'revision_agent_editor',
      step_index: 4,
      step_count: 6,
      attempt_number: 2,
      status: 'completed',
      retry_count: 0,
      output_summary: null,
      pipeline_run_id: 'run-1',
      input_hash: null,
      output_hash: null,
      input_summary: null,
      output_payload: null,
      schema_name: null,
      schema_version: null,
      prompt_version: null,
      parser_version: null,
      model_provider: null,
      model_name: null,
      token_usage: null,
      latency_ms: null,
      error: null,
      started_at: null,
      completed_at: null,
      superseded_at: null,
      created_at: '2026-09-01T08:00:00Z',
      updated_at: '2026-09-01T08:00:00Z',
    };
    const repairArtifact: RevisionArtifact = {
      artifact_key: 'revision_agent_draft_report_repair',
      artifact_kind: 'pipeline_artifact',
      status: 'pending_qa',
      entity_type: null,
      entity_name: null,
      schema_version: null,
      payload: null,
      revision_version_id: 1,
      pipeline_run_id: 'run-1',
      created_at: '2026-09-01T08:00:00Z',
      updated_at: '2026-09-01T08:00:00Z',
    };

    expect(formatRevisionStepLabel(editorStep)).toBe('Report editor');
    expect(formatRevisionArtifactLabel(repairArtifact)).toBe('Repair draft');
    expect(formatRevisionArtifactPurpose(repairArtifact)).toContain('repair candidate');
    expect(formatRevisionStatusLabel(repairArtifact.status)).toBe('Pending QA');
    expect(revisionStatusTone(repairArtifact.status)).toBe('warning');
  });

  it('preserves safe fallbacks when records have unknown or missing labels', () => {
    const step: RevisionPipelineStep = {
      step_name: 'custom_revision_checkpoint',
      step_index: 1,
      step_count: 1,
      attempt_number: 1,
      status: 'recorded',
      retry_count: 0,
      output_summary: null,
      pipeline_run_id: 'run-1',
      input_hash: null,
      output_hash: null,
      input_summary: null,
      output_payload: null,
      schema_name: null,
      schema_version: null,
      prompt_version: null,
      parser_version: null,
      model_provider: null,
      model_name: null,
      token_usage: null,
      latency_ms: null,
      error: null,
      started_at: null,
      completed_at: null,
      superseded_at: null,
      created_at: '2026-09-01T08:00:00Z',
      updated_at: '2026-09-01T08:00:00Z',
    };
    const artifact: RevisionArtifact = {
      artifact_key: null,
      artifact_kind: null,
      status: null,
      entity_type: null,
      entity_name: null,
      schema_version: null,
      payload: null,
      revision_version_id: 1,
      pipeline_run_id: 'run-1',
      created_at: '2026-09-01T08:00:00Z',
      updated_at: '2026-09-01T08:00:00Z',
    };

    expect(formatRevisionStepLabel(step)).toBe('Custom Revision Checkpoint');
    expect(formatRevisionStepSummary(step)).toBeNull();
    expect(formatRevisionArtifactLabel(artifact)).toBe('Artifact');
    expect(formatRevisionArtifactPurpose(artifact)).toBe('Persisted output retained from the revision run.');
    expect(formatRevisionStatusLabel(artifact.status)).toBe('Recorded');
    expect(revisionStatusTone(artifact.status)).toBe('neutral');
  });
});
