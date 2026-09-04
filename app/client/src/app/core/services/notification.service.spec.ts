import { describe, expect, it } from 'vitest';

import { NotificationService } from './notification.service';

describe('NotificationService', () => {
  it('publishes a transient notification and stores a clean history entry', () => {
    const service = new NotificationService();

    service.notify('[ERROR] Provider connection failed.', 'error', 0);

    expect(service.active()?.message).toBe('Provider connection failed.');
    expect(service.active()?.tone).toBe('error');
    expect(service.history()).toHaveLength(1);
    expect(service.unreadCount()).toBe(1);
  });

  it('marks history as read without removing entries', () => {
    const service = new NotificationService();
    service.notify('Configuration saved.', 'success', 0);
    service.notify('Catalog refreshed.', 'info', 0);

    service.markAllRead();

    expect(service.history()).toHaveLength(2);
    expect(service.unreadCount()).toBe(0);
  });

  it('clears history independently from the active popup', () => {
    const service = new NotificationService();
    service.notify('Configuration saved.', 'success', 0);

    service.clearHistory();

    expect(service.history()).toEqual([]);
    expect(service.active()?.message).toBe('Configuration saved.');
  });
});
