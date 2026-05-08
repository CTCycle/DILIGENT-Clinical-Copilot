import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

export type StatusTone = 'is-error' | 'is-info' | 'is-success';

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

@Component({
  selector: 'app-status-message',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (normalizedMessage) {
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
export class StatusMessageComponent {
  @Input() message = '';
  @Input() tone?: StatusTone;
  @Input() className = 'model-config-status-message';

  get normalizedMessage(): string {
    return this.message.trim();
  }

  get resolvedTone(): StatusTone {
    return this.tone ?? resolveStatusTone(this.normalizedMessage);
  }
}
