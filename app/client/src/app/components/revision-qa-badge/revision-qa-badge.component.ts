import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

import {
  JobStatus,
  SessionClinicalReviewStatus,
  SessionLlmQaStatus,
  SessionRevisionKind,
  SessionVersionStatus,
} from '../../core/models/types';
import { humanizeStatusLabel } from '../../core/utils/status-formatting';

type RevisionQaBadgeStatus =
  | JobStatus
  | SessionClinicalReviewStatus
  | SessionLlmQaStatus
  | SessionRevisionKind
  | SessionVersionStatus
  | string;

@Component({
  selector: 'app-revision-qa-badge',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './revision-qa-badge.component.html',
  styleUrl: './revision-qa-badge.component.scss',
})
export class RevisionQaBadgeComponent {
  @Input() status: RevisionQaBadgeStatus = '';
  @Input() label = '';

  get resolvedLabel(): string {
    return this.label.trim() || this.statusLabel(this.status);
  }

  get toneClass(): string {
    switch (this.status) {
      case 'current':
      case 'completed':
      case 'llm_qa_passed':
      case 'passed':
      case 'approved_by_human':
      case 'human_approved':
        return 'is-good';
      case 'requires_human_review':
      case 'pending':
      case 'pending_qa':
      case 'under_review':
      case 'draft_revision':
      case 'running':
      case 'passed_with_warnings':
        return 'is-warn';
      case 'failed':
      case 'qa_failed':
      case 'human_rejected':
      case 'rejected_by_human':
      case 'cancelled':
        return 'is-bad';
      case 'llm_assisted_revision':
      case 'manual_edit':
      case 'original':
      case 'not_run':
      case 'not_reviewed':
      case 'superseded':
        return 'is-neutral';
      default:
        return 'is-info';
    }
  }

  private statusLabel(value: string | null | undefined): string {
    return humanizeStatusLabel(value);
  }
}
