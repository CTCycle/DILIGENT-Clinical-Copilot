import { Component, input } from '@angular/core';

import type { RevisionArtifact, RevisionPipelineStep } from '../../../core/models/revision-types';
import {
  formatRevisionArtifactLabel,
  formatRevisionArtifactMeta,
  formatRevisionArtifactPurpose,
  formatRevisionStatusLabel,
  formatRevisionStepLabel,
  formatRevisionStepMeta,
  formatRevisionStepSummary,
  revisionStatusTone,
} from '../clinical-session-revision-display';

@Component({
  selector: 'app-clinical-session-revision-audit',
  standalone: true,
  template: `
    <section class="clinical-session-revision-audit" aria-label="Revision audit">
      <div class="clinical-session-revision-audit-header">
        <div>
          <p class="clinical-session-revision-eyebrow">Revision audit</p>
          <h4>Process and saved outputs</h4>
          <p>The trace records the model workflow; artifacts are the outputs retained for review.</p>
        </div>
        <div class="clinical-session-revision-audit-counts" aria-label="Revision audit counts">
          <span>{{ steps().length }} {{ steps().length === 1 ? 'step' : 'steps' }}</span>
          <span>{{ artifacts().length }} {{ artifacts().length === 1 ? 'artifact' : 'artifacts' }}</span>
        </div>
      </div>

      <div class="clinical-session-revision-audit-grid">
        <section class="clinical-session-revision-audit-group" aria-labelledby="revision-agent-trace-heading">
          <div class="clinical-session-revision-audit-group-header">
            <div>
              <p class="clinical-session-revision-eyebrow">Ordered workflow</p>
              <h5 id="revision-agent-trace-heading">Agent trace</h5>
            </div>
            <span class="clinical-session-revision-audit-count">{{ steps().length }}</span>
          </div>
          @if (steps().length) {
            <ol class="clinical-session-revision-step-list">
              @for (step of steps(); track $index) {
                <li class="clinical-session-revision-step">
                  <span class="clinical-session-revision-step-marker" aria-hidden="true">{{ $index + 1 }}</span>
                  <div class="clinical-session-revision-audit-row-content">
                    <div class="clinical-session-revision-audit-row-heading">
                      <strong>{{ revisionStepLabel(step) }}</strong>
                      <span
                        class="clinical-session-revision-status"
                        [class.is-success]="revisionStatusTone(step.status) === 'success'"
                        [class.is-warning]="revisionStatusTone(step.status) === 'warning'"
                        [class.is-error]="revisionStatusTone(step.status) === 'error'"
                      >{{ revisionStatusLabel(step.status) }}</span>
                    </div>
                    @if (revisionStepMeta(step); as stepMeta) {
                      <small class="clinical-session-revision-audit-meta">{{ stepMeta }}</small>
                    }
                    @if (revisionStepSummary(step); as stepSummary) {
                      <small class="clinical-session-revision-audit-summary">{{ stepSummary }}</small>
                    }
                    @if (step.error?.['message']; as stepError) {
                      <small class="clinical-session-revision-audit-error">{{ stepError }}</small>
                    }
                  </div>
                </li>
              }
            </ol>
          } @else {
            <p class="clinical-session-revision-audit-empty">No process steps recorded yet.</p>
          }
        </section>

        <section class="clinical-session-revision-audit-group" aria-labelledby="revision-draft-artifacts-heading">
          <div class="clinical-session-revision-audit-group-header">
            <div>
              <p class="clinical-session-revision-eyebrow">Persisted outputs</p>
              <h5 id="revision-draft-artifacts-heading">Draft artifacts</h5>
            </div>
            <span class="clinical-session-revision-audit-count">{{ artifacts().length }}</span>
          </div>
          @if (artifacts().length) {
            <ul class="clinical-session-revision-artifact-list">
              @for (artifact of artifacts(); track $index) {
                <li class="clinical-session-revision-artifact">
                  <div class="clinical-session-revision-audit-row-content">
                    <div class="clinical-session-revision-audit-row-heading">
                      <strong>{{ revisionArtifactLabel(artifact) }}</strong>
                      <span
                        class="clinical-session-revision-status"
                        [class.is-success]="revisionStatusTone(artifact.status) === 'success'"
                        [class.is-warning]="revisionStatusTone(artifact.status) === 'warning'"
                        [class.is-error]="revisionStatusTone(artifact.status) === 'error'"
                      >{{ revisionStatusLabel(artifact.status) }}</span>
                    </div>
                    <small class="clinical-session-revision-audit-summary">{{ revisionArtifactPurpose(artifact) }}</small>
                    @if (revisionArtifactMeta(artifact); as artifactMeta) {
                      <small class="clinical-session-revision-audit-meta">{{ artifactMeta }}</small>
                    }
                  </div>
                </li>
              }
            </ul>
          } @else {
            <p class="clinical-session-revision-audit-empty">No draft artifacts recorded yet.</p>
          }
        </section>
      </div>
    </section>
  `,
  styleUrl: './clinical-session-revision-audit.component.scss',
})
export class ClinicalSessionRevisionAuditComponent {
  readonly steps = input.required<RevisionPipelineStep[]>();
  readonly artifacts = input.required<RevisionArtifact[]>();

  readonly revisionStepLabel = formatRevisionStepLabel;
  readonly revisionStepMeta = formatRevisionStepMeta;
  readonly revisionStepSummary = formatRevisionStepSummary;
  readonly revisionArtifactLabel = formatRevisionArtifactLabel;
  readonly revisionArtifactPurpose = formatRevisionArtifactPurpose;
  readonly revisionArtifactMeta = formatRevisionArtifactMeta;
  readonly revisionStatusLabel = formatRevisionStatusLabel;
  readonly revisionStatusTone = revisionStatusTone;
}
