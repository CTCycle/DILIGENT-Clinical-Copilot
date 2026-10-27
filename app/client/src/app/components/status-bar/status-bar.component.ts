// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, OnDestroy, computed, inject } from '@angular/core';

import { MemoryUsage } from '../../core/services/system-status-api';
import { SystemStatusStateService } from '../../core/state/system-status-state.service';

type LedState = 'ok' | 'error' | 'pending' | 'unknown';

type MemoryReading = { detail: string; percent: string };

const BYTES_PER_GB = 1024 ** 3;
const NOT_AVAILABLE = 'N/A';

function formatMemory(usage: MemoryUsage | null | undefined): MemoryReading | null {
  if (!usage || usage.total_bytes <= 0) {
    return null;
  }
  const used = (usage.used_bytes / BYTES_PER_GB).toFixed(1);
  const total = (usage.total_bytes / BYTES_PER_GB).toFixed(1);
  const percent = Math.round((usage.used_bytes / usage.total_bytes) * 100);
  return { detail: `${used}/${total} GB`, percent: `${percent}%` };
}

@Component({
  selector: 'app-status-bar',
  templateUrl: './status-bar.component.html',
  styleUrl: './status-bar.component.scss',
})
export class StatusBarComponent implements OnDestroy {
  private readonly state = inject(SystemStatusStateService);
  readonly notAvailable = NOT_AVAILABLE;

  readonly backend = computed<{ led: LedState; label: string }>(() => {
    switch (this.state.connection()) {
      case 'connected':
        return { led: 'ok', label: 'Connected' };
      case 'disconnected':
        return { led: 'error', label: 'Disconnected' };
      default:
        return { led: 'pending', label: 'Checking' };
    }
  });

  readonly providers = computed(() => {
    const status = this.state.status();
    return status ? `${status.providers.active}` : NOT_AVAILABLE;
  });

  readonly providersTitle = computed(() => {
    const status = this.state.status();
    return status
      ? `${status.providers.active} of ${status.providers.total} cloud providers have an active access key and a usable model catalog`
      : 'Provider availability is unknown';
  });

  readonly local = computed<{ led: LedState; label: string }>(() => {
    const status = this.state.status();
    if (status) {
      return status.ollama === 'reachable'
        ? { led: 'ok', label: 'Reachable' }
        : { led: 'error', label: 'Unreachable' };
    }
    return this.state.connection() === 'loading'
      ? { led: 'pending', label: 'Checking' }
      : { led: 'unknown', label: 'Unknown' };
  });

  readonly cpu = computed(() => {
    const value = this.state.status()?.cpu_percent;
    return typeof value === 'number' ? `${Math.round(value)}%` : NOT_AVAILABLE;
  });

  readonly ram = computed(() => formatMemory(this.state.status()?.ram));
  readonly vram = computed(() => formatMemory(this.state.status()?.vram));

  constructor() {
    this.state.start();
  }

  ngOnDestroy(): void {
    this.state.stop();
  }
}
