import type { RevisionArtifact, RevisionPipelineStep } from '../../core/models/revision-types';

export type RevisionAuditStatusTone = 'success' | 'warning' | 'error' | 'neutral';

const ARTIFACT_LABELS: Readonly<Record<string, string>> = {
  revision_agent_context: 'Revision context',
  revision_agent_issue_scan: 'Issue scan',
  revision_agent_plan: 'Revision plan',
  revision_agent_tool_trace: 'Tool trace',
  revision_agent_draft_report: 'Generated draft',
  revision_agent_draft_report_repair: 'Repair draft',
  revision_agent_qa: 'Quality review',
  revision_agent_qa_repair: 'Repair quality review',
};

const ARTIFACT_PURPOSES: Readonly<Record<string, string>> = {
  revision_agent_context: 'Context assembled for the revision run.',
  revision_agent_issue_scan: 'Initial issues and tool intents identified for review.',
  revision_agent_plan: 'Plan used to guide the revision tasks.',
  revision_agent_tool_trace: 'Recorded tool interactions used during revision.',
  revision_agent_draft_report: 'Proposed report retained for human review.',
  revision_agent_draft_report_repair: 'Evidence-focused repair candidate retained before its follow-up quality review.',
  revision_agent_qa: 'Quality checks applied to the generated draft.',
  revision_agent_qa_repair: 'Follow-up quality checks applied after the bounded repair attempt.',
};

function normalizedStatus(status: string | null | undefined): string {
  return String(status ?? '').trim().toLowerCase();
}

function humanize(value: string | null | undefined, fallback: string): string {
  const normalized = String(value ?? '')
    .trim()
    .replace(/^[._-]+|[._-]+$/g, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ');
  if (!normalized) return fallback;
  return normalized.replace(/\b\w/g, (character) => character.toUpperCase());
}

function countLabel(value: unknown, singular: string, plural: string): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const count = Math.max(0, Math.round(value));
  return `${count} ${count === 1 ? singular : plural}`;
}

export function formatRevisionStepLabel(step: RevisionPipelineStep): string {
  const stepName = step.step_name.trim();
  if (stepName === 'revision_agent_issue_scan') return 'Issue scan';
  if (stepName === 'revision_agent_planner') return 'Revision planning';
  if (stepName === 'revision_agent_editor') return 'Report editor';
  if (stepName === 'revision_agent_qa') return 'Quality review';
  const taskMatch = /^revision_agent_task_(\d+)$/.exec(stepName);
  if (taskMatch) return `Revision task ${taskMatch[1]}`;
  return humanize(stepName, 'Revision step');
}

export function formatRevisionStepMeta(step: RevisionPipelineStep): string {
  const parts: string[] = [];
  const stepIndex = Number(step.step_index);
  const stepCount = Number(step.step_count);
  if (Number.isFinite(stepIndex) && Number.isFinite(stepCount) && stepCount > 0) {
    parts.push(`Step ${Math.max(1, Math.round(stepIndex))} of ${Math.round(stepCount)}`);
  } else if (Number.isFinite(stepIndex)) {
    parts.push(`Step ${Math.max(1, Math.round(stepIndex))}`);
  }

  const attempt = Number(step.attempt_number);
  if (Number.isFinite(attempt) && attempt > 1) parts.push(`Attempt ${Math.round(attempt)}`);
  const retryCount = Number(step.retry_count);
  if (Number.isFinite(retryCount) && retryCount > 0) {
    parts.push(`${Math.round(retryCount)} ${retryCount === 1 ? 'retry' : 'retries'}`);
  }
  return parts.join(' · ');
}

export function formatRevisionStepSummary(step: RevisionPipelineStep): string | null {
  const summary = step.output_summary;
  if (!summary) return null;
  const parts = [
    countLabel(summary['issue_count'], 'issue', 'issues'),
    countLabel(summary['tool_intent_count'], 'tool intent', 'tool intents'),
    countLabel(summary['tool_call_count'], 'tool call', 'tool calls'),
  ].filter((value): value is string => Boolean(value));
  return parts.length ? parts.join(' · ') : null;
}

export function formatRevisionArtifactLabel(artifact: RevisionArtifact): string {
  const key = artifact.artifact_key?.trim() || '';
  return ARTIFACT_LABELS[key]
    || humanize(key || artifact.entity_name || artifact.artifact_kind, 'Artifact');
}

export function formatRevisionArtifactPurpose(artifact: RevisionArtifact): string {
  const key = artifact.artifact_key?.trim() || '';
  if (ARTIFACT_PURPOSES[key]) return ARTIFACT_PURPOSES[key];
  if (artifact.entity_name?.trim()) {
    return `${humanize(artifact.artifact_kind, 'Persisted output')} for ${artifact.entity_name.trim()}.`;
  }
  return 'Persisted output retained from the revision run.';
}

export function formatRevisionArtifactMeta(artifact: RevisionArtifact): string {
  const parts: string[] = [];
  if (artifact.artifact_kind?.trim()) parts.push(humanize(artifact.artifact_kind, 'Artifact'));
  if (artifact.entity_type?.trim()) parts.push(humanize(artifact.entity_type, 'Entity'));
  if (artifact.schema_version?.trim()) parts.push(`Schema ${artifact.schema_version.trim()}`);
  return parts.join(' · ');
}

export function formatRevisionStatusLabel(status: string | null | undefined): string {
  switch (normalizedStatus(status)) {
    case 'completed':
    case 'complete':
      return 'Completed';
    case 'running':
    case 'started':
    case 'in_progress':
      return 'In progress';
    case 'pending':
    case 'queued':
      return 'Queued';
    case 'failed':
    case 'error':
      return 'Failed';
    case 'cancelled':
    case 'canceled':
      return 'Cancelled';
    case 'qa_failed':
      return 'QA failed';
    case 'passed':
    case 'qa_passed':
    case 'llm_qa_passed':
      return 'Passed';
    case 'requires_human_review':
      return 'Human review required';
    case 'under_review':
      return 'Under review';
    case 'derived':
      return 'Derived';
    case 'pending_qa':
      return 'Pending QA';
    case 'recorded':
    case '':
      return 'Recorded';
    default:
      return humanize(status, 'Recorded');
  }
}

export function revisionStatusTone(status: string | null | undefined): RevisionAuditStatusTone {
  switch (normalizedStatus(status)) {
    case 'completed':
    case 'complete':
    case 'passed':
    case 'qa_passed':
    case 'llm_qa_passed':
      return 'success';
    case 'failed':
    case 'error':
    case 'qa_failed':
      return 'error';
    case 'running':
    case 'started':
    case 'in_progress':
    case 'pending':
    case 'queued':
    case 'cancelled':
    case 'canceled':
    case 'requires_human_review':
    case 'under_review':
    case 'pending_qa':
      return 'warning';
    default:
      return 'neutral';
  }
}
