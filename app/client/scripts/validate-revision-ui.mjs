import { readFileSync } from 'node:fs';
import path from 'node:path';

const clientRoot = process.cwd();

function load(relativePath) {
  return readFileSync(path.resolve(clientRoot, relativePath), 'utf8');
}

function assertContains(source, snippet, label) {
  if (!source.includes(snippet)) {
    throw new Error(`Missing ${label}: ${snippet}`);
  }
}

function assertAll(source, checks) {
  for (const [snippet, label] of checks) {
    assertContains(source, snippet, label);
  }
}

const clinicalSessionsTemplate = load(
  'src/app/pages/clinical-sessions/clinical-sessions-page.component.html',
);
const clinicalSessionsComponent = load(
  'src/app/pages/clinical-sessions/clinical-sessions-page.component.ts',
);
const revisionQaBadgeComponent = load(
  'src/app/components/revision-qa-badge/revision-qa-badge.component.ts',
);
const revisionPipelineStatusComponent = load(
  'src/app/components/revision-pipeline-status/revision-pipeline-status.component.ts',
);
const typesFile = load('src/app/core/models/types.ts');

assertAll(clinicalSessionsTemplate, [
  ['Save manual report edit', 'manual-edit action'],
  ['Manual Edit History', 'manual-edit history section'],
  ['Official Version History', 'official version history section'],
  [
    'Manual report edits stay separate and in-place; this workflow creates versioned artifacts, persisted run checkpoints, and a new reviewable draft shell.',
    'manual-edit versus revision separation copy',
  ],
  [
    'LLM-assisted version lineage stays separate from manual in-place report edits.',
    'version-lineage separation copy',
  ],
  ['Revision Pipeline Details', 'revision pipeline panel'],
  ['Revision QA And Artifacts', 'revision QA panel'],
  ['Human Clinical Review', 'human clinical review panel'],
  ['Version Comparison', 'version comparison panel'],
  ['app-revision-pipeline-status', 'pipeline status component usage'],
  ['app-revision-qa-badge', 'QA badge component usage'],
  ["updateRevisionClinicalReview('approved_by_human')", 'approve review action'],
  ["updateRevisionClinicalReview('rejected_by_human')", 'reject review action'],
]);

assertAll(clinicalSessionsComponent, [
  ['revisionEntitySummary(entity: RevisionEntity): string {', 'revision entity summary helper'],
  ["parts.push('Requires human review');", 'human-review indicator rendering'],
  ['revisionComparisonEntitySummary(items: RevisionEntityDiff[]): string {', 'comparison summary helper'],
  ['revisionArtifactPayloadFlag(', 'artifact flag formatter'],
]);

assertAll(revisionQaBadgeComponent, [
  ["case 'approved_by_human':", 'approved human-review status'],
  ["case 'requires_human_review':", 'requires-human-review status'],
  ["case 'qa_failed':", 'QA failure status'],
  ["case 'manual_edit':", 'manual-edit revision kind status'],
  ["return this.label.trim() || this.statusLabel(this.status);", 'custom-label preference'],
]);

assertAll(revisionPipelineStatusComponent, [
  ["return this.steps.filter((step) => step.status === 'completed').length;", 'completed-step counter'],
  ['const outputSummary = step.output_summary;', 'output summary prioritization'],
  ['const inputSummary = step.input_summary;', 'input summary fallback'],
  ["return 'No structured summary saved.';", 'empty summary fallback'],
]);

assertAll(typesFile, [
  ['export type ManualReportEditAudit = {', 'manual-edit audit contract'],
  ['export type SessionVersionSummary = {', 'version summary contract'],
  ['export type RevisionPipelineRun = {', 'revision pipeline run contract'],
  ['export type RevisionPipelineStep = {', 'revision pipeline step contract'],
  ['export type RevisionArtifact = {', 'revision artifact contract'],
  ['export type RevisionEntity = {', 'revision entity contract'],
]);

console.log('Revision UI validation passed.');
