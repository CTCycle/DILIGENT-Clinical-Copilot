import { CommonModule } from '@angular/common';
import { Component, Input, OnChanges, inject } from '@angular/core';

import { NotificationService, NotificationTone } from '../../core/services/notification.service';

export type StatusTone = 'is-error' | 'is-info' | 'is-success';
export type StatusPresentation = 'notification' | 'inline';

export function resolveStatusTone(message: string): StatusTone {
  const normalized = message.trim().toUpperCase();
  if (!normalized) {
    return 'is-info';
  }
  if (normalized.startsWith('[ERROR]')) {
    return 'is-error';
  }
  if (normalized.startsWith('[INFO]')) {
    return 'is-info';
  }
  return 'is-success';
}

function notificationToneFromStatus(tone: StatusTone): NotificationTone {
  if (tone === 'is-error') {
    return 'error';
  }
  if (tone === 'is-success') {
    return 'success';
  }
  return 'info';
}

@Component({
  selector: 'app-status-message',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (presentation === 'inline' && normalizedMessage) {
      <p
        [class]="className + ' ' + resolvedTone"
        [attr.role]="resolvedTone === 'is-error' ? 'alert' : 'status'"
        [attr.aria-live]="resolvedTone === 'is-error' ? 'assertive' : 'polite'"
      >
        {{ normalizedMessage }}
      </p>
    }
  `,
})
export class StatusMessageComponent implements OnChanges {
  @Input() message = '';
  @Input() tone?: StatusTone;
  @Input() className = 'model-config-status-message';
  @Input() presentation: StatusPresentation = 'notification';

  private readonly notifications = inject(NotificationService);
  private lastPublishedSignature = '';

  ngOnChanges(): void {
    if (this.presentation !== 'notification') {
      return;
    }

    const normalizedMessage = this.normalizedMessage;
    if (!normalizedMessage) {
      this.lastPublishedSignature = '';
      return;
    }

    const signature = `${this.resolvedTone}:${normalizedMessage}`;
    if (signature === this.lastPublishedSignature) {
      return;
    }
    this.lastPublishedSignature = signature;
    this.notifications.notify(normalizedMessage, notificationToneFromStatus(this.resolvedTone));
  }

  get normalizedMessage(): string {
    return this.message.trim();
  }

  get resolvedTone(): StatusTone {
    return this.tone ?? resolveStatusTone(this.normalizedMessage);
  }
}
