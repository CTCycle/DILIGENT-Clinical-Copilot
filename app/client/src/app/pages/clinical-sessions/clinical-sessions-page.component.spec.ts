import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('ClinicalSessionsPage revision template', () => {
  it('renders separate official version history and pipeline retry affordances', () => {
    const templatePath = resolve(
      __dirname,
      './clinical-sessions-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');

    expect(template).toContain('Official Version History');
    expect(template).toContain('Revision Pipeline Details');
    expect(template).toContain('Human Clinical Review');
    expect(template).toContain('Approve Revision');
    expect(template).toContain('Version Comparison');
    expect(template).toContain('Compare against');
    expect(template).toContain('app-revision-pipeline-status');
    expect(template).toContain('app-revision-qa-badge');
    expect(template).toContain('Save manual report edit');
    expect(template).toContain(
      'Manual report edits stay separate and in-place; this workflow creates versioned artifacts, persisted run checkpoints, and a new reviewable draft shell.',
    );
    expect(template).toContain(
      'LLM-assisted version lineage stays separate from manual in-place report edits.',
    );
    expect(template).toContain('Revision QA And Artifacts');
    expect(template).toContain('Revision Consultation Provenance');
    expect(template).toContain('Revision Finalization Provenance');
    expect(template).toContain('Structured Case Entities');
    expect(template).toContain('LiverTox Match Decisions');
    expect(template).toContain('Revised DILI Assessments');
    expect(template).toContain('Manual Edit History');
  });

  it('labels frontend-derived clinical evidence as fallback', () => {
    const templatePath = resolve(
      __dirname,
      './clinical-sessions-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');

    expect(template).toContain('Display fallback');
    expect(template).toContain('Not backend-confirmed');
    expect(template).toContain('drug.bibliographyLabel');
    expect(template).toContain('drug.bibliographyFallback');
  });
});
