import { Injectable, computed, signal } from '@angular/core';

export type NotificationTone = 'error' | 'info' | 'success';

export type AppNotification = {
  id: number;
  message: string;
  tone: NotificationTone;
  createdAt: number;
  read: boolean;
};

const MAX_NOTIFICATION_HISTORY = 80;
const DEFAULT_NOTIFICATION_DURATION_MS = 4500;

function stripStatusPrefix(message: string): string {
  return message.trim().replace(/^\[(ERROR|INFO|SUCCESS)\]\s*/i, '');
}

@Injectable({ providedIn: 'root' })
export class NotificationService {
  readonly history = signal<AppNotification[]>([]);
  readonly active = signal<AppNotification | null>(null);
  readonly unreadCount = computed(() => this.history().filter((item) => !item.read).length);

  private nextId = 1;
  private dismissTimer: ReturnType<typeof setTimeout> | null = null;

  notify(
    message: string,
    tone: NotificationTone = 'info',
    durationMs = DEFAULT_NOTIFICATION_DURATION_MS,
  ): void {
    const normalizedMessage = stripStatusPrefix(message);
    if (!normalizedMessage) {
      return;
    }

    const notification: AppNotification = {
      id: this.nextId++,
      message: normalizedMessage,
      tone,
      createdAt: Date.now(),
      read: false,
    };

    this.history.update((items) => [notification, ...items].slice(0, MAX_NOTIFICATION_HISTORY));
    this.active.set(notification);
    this.scheduleDismiss(notification.id, durationMs);
  }

  dismiss(notificationId?: number): void {
    const current = this.active();
    if (!current || (notificationId !== undefined && current.id !== notificationId)) {
      return;
    }
    this.clearDismissTimer();
    this.active.set(null);
  }

  markAllRead(): void {
    this.history.update((items) => items.map((item) => (item.read ? item : { ...item, read: true })));
  }

  clearHistory(): void {
    this.history.set([]);
  }

  private scheduleDismiss(notificationId: number, durationMs: number): void {
    this.clearDismissTimer();
    if (durationMs <= 0) {
      return;
    }
    this.dismissTimer = setTimeout(() => this.dismiss(notificationId), durationMs);
  }

  private clearDismissTimer(): void {
    if (this.dismissTimer !== null) {
      clearTimeout(this.dismissTimer);
      this.dismissTimer = null;
    }
  }
}
