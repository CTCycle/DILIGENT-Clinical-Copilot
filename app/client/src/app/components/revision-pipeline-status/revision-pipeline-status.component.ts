import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

import { RevisionPipelineRun, RevisionPipelineStep } from '../../core/models/types';
import { humanizeStatusLabel } from '../../core/utils/status-formatting';
import { RevisionQaBadgeComponent } from '../revision-qa-badge/revision-qa-badge.component';

@Component({
  selector: 'app-revision-pipeline-status',
  standalone: true,
  imports: [CommonModule, RevisionQaBadgeComponent],
  templateUrl: './revision-pipeline-status.component.html',
  styleUrl: './revision-pipeline-status.component.scss',
})
export class RevisionPipelineStatusComponent {
  @Input() run: RevisionPipelineRun | null = null;
  @Input() steps: RevisionPipelineStep[] = [];
  @Input() canRetry = false;
  @Input() retryLoading = false;

  @Output() retry = new EventEmitter<void>();

  get completedStepCount(): number {
    return this.steps.filter((step) => step.status === 'completed').length;
  }

  statusLabel(value: string | null | undefined): string {
    return humanizeStatusLabel(value);
  }

  stepSummary(step: RevisionPipelineStep): string {
    const outputSummary = step.output_summary;
    if (outputSummary && Object.keys(outputSummary).length > 0) {
      return Object.entries(outputSummary)
        .slice(0, 3)
        .map(([key, value]) => `${this.statusLabel(key)}: ${String(value)}`)
        .join(' · ');
    }
    const inputSummary = step.input_summary;
    if (inputSummary && Object.keys(inputSummary).length > 0) {
      return Object.entries(inputSummary)
        .slice(0, 3)
        .map(([key, value]) => `${this.statusLabel(key)}: ${String(value)}`)
        .join(' · ');
    }
    return 'No structured summary saved.';
  }
}
