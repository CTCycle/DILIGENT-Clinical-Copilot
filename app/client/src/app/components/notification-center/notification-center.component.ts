import { Component, HostListener, inject, signal } from '@angular/core';

import { AppNotification, NotificationService } from '../../core/services/notification.service';

@Component({
  selector: 'app-notification-center',
  standalone: true,
  template: `
    <div class="notification-center">
      <button
        type="button"
        class="app-header-icon-btn notification-center-button"
        (click)="toggleHistory()"
        [attr.aria-expanded]="historyOpen()"
        aria-haspopup="dialog"
        aria-label="Notifications"
        title="Notifications"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
          <path d="M10 21h4" />
        </svg>
        @if (notifications.unreadCount() > 0) {
          <span class="notification-center-badge" aria-hidden="true">{{ unreadBadge() }}</span>
        }
      </button>

      @if (historyOpen()) {
        <section class="notification-history" role="dialog" aria-label="Notification history">
          <header class="notification-history-header">
            <div>
              <strong>Notifications</strong>
              <span>{{ notifications.history().length }} recent</span>
            </div>
            @if (notifications.history().length > 0) {
              <button type="button" class="notification-history-clear" (click)="clearHistory()">Clear</button>
            }
          </header>

          <div class="notification-history-list">
            @for (item of notifications.history(); track item.id) {
              <article class="notification-history-item" [class]="'notification-history-item is-' + item.tone">
                <span class="notification-history-indicator" aria-hidden="true"></span>
                <div>
                  <p>{{ item.message }}</p>
                  <time [attr.datetime]="isoDate(item)">{{ formatTime(item) }}</time>
                </div>
              </article>
            } @empty {
              <p class="notification-history-empty">No notifications yet.</p>
            }
          </div>
        </section>
      }
    </div>

    @if (notifications.active(); as item) {
      <aside
        class="notification-toast"
        [class]="'notification-toast is-' + item.tone"
        [attr.role]="item.tone === 'error' ? 'alert' : 'status'"
        [attr.aria-live]="item.tone === 'error' ? 'assertive' : 'polite'"
      >
        <span class="notification-toast-indicator" aria-hidden="true"></span>
        <p>{{ item.message }}</p>
        <button type="button" (click)="notifications.dismiss(item.id)" aria-label="Dismiss notification">Close</button>
      </aside>
    }
  `,
  styles: `
    :host {
      display: inline-flex;
    }

    .notification-center {
      position: relative;
      display: inline-flex;
    }

    .notification-center-button {
      position: relative;
    }

    .notification-center-button svg {
      width: 18px;
      height: 18px;
      fill: none;
      stroke: currentColor;
      stroke-width: 1.8;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .notification-center-badge {
      position: absolute;
      top: 2px;
      right: 1px;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      border: 2px solid var(--color-surface);
      border-radius: 999px;
      background: var(--color-status-error-text);
      color: #fff;
      font-size: 9px;
      font-weight: 800;
      line-height: 12px;
      text-align: center;
    }

    .notification-history {
      position: absolute;
      top: calc(100% + 10px);
      right: 0;
      z-index: 1300;
      width: min(360px, calc(100vw - 32px));
      overflow: hidden;
      border: 1px solid var(--color-border-subtle);
      border-radius: var(--radius-lg);
      background: var(--color-surface);
      box-shadow: var(--shadow-lg);
    }

    .notification-history-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-md);
      padding: var(--space-md) var(--space-lg);
      border-bottom: 1px solid var(--color-divider);
    }

    .notification-history-header div {
      display: grid;
      gap: 1px;
    }

    .notification-history-header strong {
      font-size: var(--font-base);
    }

    .notification-history-header span,
    .notification-history-item time {
      color: var(--color-text-muted);
      font-size: var(--font-xs);
    }

    .notification-history-clear {
      border: 0;
      background: transparent;
      color: var(--color-brand);
      font: inherit;
      font-size: var(--font-sm);
      font-weight: 700;
      cursor: pointer;
    }

    .notification-history-list {
      max-height: min(420px, 62vh);
      overflow-y: auto;
    }

    .notification-history-item {
      display: grid;
      grid-template-columns: 8px minmax(0, 1fr);
      gap: var(--space-md);
      padding: var(--space-md) var(--space-lg);
      border-bottom: 1px solid var(--color-border-subtle);
    }

    .notification-history-item:last-child {
      border-bottom: 0;
    }

    .notification-history-item p,
    .notification-toast p {
      margin: 0;
    }

    .notification-history-item p {
      color: var(--color-text-primary);
      font-size: var(--font-sm);
      line-height: 1.45;
    }

    .notification-history-indicator,
    .notification-toast-indicator {
      width: 8px;
      height: 8px;
      margin-top: 5px;
      border-radius: 50%;
      background: var(--color-status-info-text);
    }

    .notification-history-item.is-success .notification-history-indicator,
    .notification-toast.is-success .notification-toast-indicator {
      background: var(--color-status-success-text);
    }

    .notification-history-item.is-error .notification-history-indicator,
    .notification-toast.is-error .notification-toast-indicator {
      background: var(--color-status-error-text);
    }

    .notification-history-empty {
      margin: 0;
      padding: var(--space-xl);
      color: var(--color-text-muted);
      font-size: var(--font-sm);
      text-align: center;
    }

    .notification-toast {
      position: fixed;
      left: 50%;
      bottom: 24px;
      z-index: 1400;
      display: grid;
      grid-template-columns: 8px minmax(0, 1fr) auto;
      align-items: start;
      gap: var(--space-md);
      width: min(460px, calc(100vw - 32px));
      padding: var(--space-md) var(--space-lg);
      border: 1px solid var(--color-status-info-border);
      border-radius: var(--radius-lg);
      background: color-mix(in srgb, var(--color-surface) 94%, var(--color-status-info-bg));
      color: var(--color-text-primary);
      box-shadow: var(--shadow-lg);
      transform: translateX(-50%);
    }

    .notification-toast.is-success {
      border-color: var(--color-status-success-border);
      background: color-mix(in srgb, var(--color-surface) 94%, var(--color-status-success-bg));
    }

    .notification-toast.is-error {
      border-color: var(--color-status-error-border);
      background: color-mix(in srgb, var(--color-surface) 94%, var(--color-status-error-bg));
    }

    .notification-toast p {
      font-size: var(--font-sm);
      font-weight: 650;
      line-height: 1.45;
    }

    .notification-toast button {
      border: 0;
      background: transparent;
      color: var(--color-text-muted);
      font: inherit;
      font-size: var(--font-xs);
      font-weight: 700;
      cursor: pointer;
    }
  `,
})
export class NotificationCenterComponent {
  readonly notifications = inject(NotificationService);
  readonly historyOpen = signal(false);

  toggleHistory(): void {
    const nextOpen = !this.historyOpen();
    this.historyOpen.set(nextOpen);
    if (nextOpen) {
      this.notifications.markAllRead();
    }
  }

  clearHistory(): void {
    this.notifications.clearHistory();
  }

  unreadBadge(): string {
    const count = this.notifications.unreadCount();
    return count > 99 ? '99+' : String(count);
  }

  formatTime(item: AppNotification): string {
    return new Intl.DateTimeFormat('en-GB', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: 'short',
    }).format(item.createdAt);
  }

  isoDate(item: AppNotification): string {
    return new Date(item.createdAt).toISOString();
  }

  @HostListener('document:keydown.escape')
  closeHistory(): void {
    this.historyOpen.set(false);
  }
}
