import { describe, expect, it } from 'vitest';

import type { RevisionArtifact, RevisionPipelineStep } from '../../core/models/revision-types';
import {
  formatRevisionArtifactLabel,
  formatRevisionArtifactMeta,
  formatRevisionArtifactPurpose,
  formatRevisionStatusLabel,
  formatRevisionStepLabel,
  formatRevisionStepMeta,
  formatRevisionStepSummary,
  revisionStatusTone,
} from './clinical-session-revision-display';

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
    };
    const repairArtifact: RevisionArtifact = {
      artifact_key: 'revision_agent_draft_report_repair',
      artifact_kind: 'pipeline_artifact',
      status: 'pending_qa',
      payload: null,
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
      status: 'recorded',
    };
    const artifact: RevisionArtifact = {
      artifact_key: null,
      artifact_kind: null,
      status: null,
      payload: null,
    };

    expect(formatRevisionStepLabel(step)).toBe('Custom Revision Checkpoint');
    expect(formatRevisionStepSummary(step)).toBeNull();
    expect(formatRevisionArtifactLabel(artifact)).toBe('Artifact');
    expect(formatRevisionArtifactPurpose(artifact)).toBe('Persisted output retained from the revision run.');
    expect(formatRevisionStatusLabel(artifact.status)).toBe('Recorded');
    expect(revisionStatusTone(artifact.status)).toBe('neutral');
  });
});
