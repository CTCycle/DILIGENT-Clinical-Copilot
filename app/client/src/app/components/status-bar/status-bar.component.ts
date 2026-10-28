// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, ElementRef, HostListener, OnDestroy, computed, inject, signal } from '@angular/core';

import { RuntimeSettings } from '../../core/models/types';
import { MemoryUsage } from '../../core/services/system-status-api';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { SystemStatusStateService } from '../../core/state/system-status-state.service';

type LedState = 'ok' | 'warning' | 'error' | 'pending' | 'unknown';

const MODEL_ROLES: { label: string; key: keyof RuntimeSettings & `${string}Model` }[] = [
  { label: 'Clinical', key: 'clinicalModel' },
  { label: 'Text extraction', key: 'textExtractionModel' },
  { label: 'Revision', key: 'revisionModel' },
  { label: 'Timeline', key: 'timelineModel' },
];

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
  private readonly modelConfig = inject(ModelConfigStateService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
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

  readonly modelsOpen = signal(false);

  readonly modelRoles = computed<{ role: string; model: string | null }[]>(() => {
    const settings = this.modelConfig.settings();
    if (!settings) {
      return MODEL_ROLES.map(({ label }) => ({ role: label, model: null }));
    }
    return MODEL_ROLES.map(({ label, key }) => ({
      role: label,
      model: (settings.useCloudServices ? settings.cloudModel : settings[key]) || null,
    }));
  });

  readonly models = computed<{ led: LedState; label: string }>(() => {
    const roles = this.modelRoles();
    const assigned = roles.filter((entry) => entry.model).length;
    if (assigned === roles.length) return { led: 'ok', label: 'All assigned' };
    return assigned > 0
      ? { led: 'warning', label: 'Partial' }
      : { led: 'error', label: 'None assigned' };
  });

  readonly cpu = computed(() => {
    const value = this.state.status()?.cpu_percent;
    return typeof value === 'number' ? `${Math.round(value)}%` : NOT_AVAILABLE;
  });

  readonly ram = computed(() => formatMemory(this.state.status()?.ram));
  readonly vram = computed(() => formatMemory(this.state.status()?.vram));

  constructor() {
    this.state.start();
    if (this.modelConfig.status() === 'idle') {
      this.modelConfig.load().catch(() => undefined);
    }
  }

  toggleModels(): void {
    this.modelsOpen.update((open) => !open);
  }

  @HostListener('document:pointerdown', ['$event'])
  closeOnOutsidePointer(event: Event): void {
    if (this.modelsOpen() && !this.host.nativeElement.querySelector('.models-anchor')?.contains(event.target as Node)) {
      this.modelsOpen.set(false);
    }
  }

  @HostListener('document:keydown.escape')
  closeOnEscape(): void {
    this.modelsOpen.set(false);
  }

  ngOnDestroy(): void {
    this.state.stop();
  }
}
