import { Component, ElementRef, HostListener, inject, signal } from '@angular/core';

import { AppNotification, NotificationService } from '../../core/services/notification.service';

@Component({
  selector: 'app-notification-center',
  standalone: true,
  templateUrl: './notification-center.component.html',
  styleUrl: './notification-center.component.scss',
})
export class NotificationCenterComponent {
  readonly notifications = inject(NotificationService);
  readonly historyOpen = signal(false);
  private readonly host = inject(ElementRef<HTMLElement>);

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

  @HostListener('document:click', ['$event'])
  closeHistoryOnOutsideClick(event: MouseEvent): void {
    const target = event.target;
    if (this.historyOpen() && target instanceof Node && !this.host.nativeElement.contains(target)) {
      this.historyOpen.set(false);
    }
  }

  @HostListener('document:keydown.escape')
  closeHistory(): void {
    this.historyOpen.set(false);
  }
}
