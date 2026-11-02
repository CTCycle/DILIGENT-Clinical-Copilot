// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, ElementRef, HostListener, OnDestroy, computed, inject, signal } from '@angular/core';

import { RuntimeSettings } from '../../core/models/types';
import { formatUnknownError } from '../../core/utils';
import { MemoryUsage, startOllama, stopOllama } from '../../core/services/system-status-api';
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

  readonly localControl = computed(() => this.state.status()?.ollama_control ?? null);

  readonly local = computed<{ led: LedState; label: string }>(() => {
    const status = this.state.status();
    if (this.ollamaBusy()) return { led: 'pending', label: 'Starting' };
    if (status) {
      const control = status.ollama_control;
      switch (control.state) {
        case 'running':
          return { led: 'ok', label: 'Running' };
        case 'starting':
          return { led: 'pending', label: 'Starting' };
        case 'error':
          return { led: 'error', label: 'Error' };
        default:
          return control.can_start
            ? { led: 'warning', label: 'Stopped' }
            : { led: 'error', label: 'Unreachable' };
      }
    }
    return this.state.connection() === 'loading'
      ? { led: 'pending', label: 'Checking' }
      : { led: 'unknown', label: 'Unknown' };
  });

  readonly localStarting = computed(() => this.ollamaBusy() || this.localControl()?.state === 'starting');

  readonly localDetail = computed(() => {
    const control = this.localControl();
    if (this.localStarting()) return 'Launching the local Ollama server…';
    if (!control) return 'Status unavailable while the backend is disconnected.';
    if (control.state === 'running') {
      return control.managed ? 'Running, started by DILIGENT.' : 'Running, managed outside DILIGENT.';
    }
    if (control.error) return control.error;
    if (!control.executable_found) return 'Ollama is not installed or not on PATH.';
    if (!control.can_start) return 'Ollama is configured on another host and cannot be started here.';
    return 'Ollama is not running.';
  });

  readonly ollamaBusy = signal(false);
  readonly ollamaMessage = signal<{ kind: 'success' | 'error'; text: string } | null>(null);
  readonly localOpen = signal(false);

  readonly modelsOpen = signal(false);

  readonly modelRoles = computed<{ role: string; model: string | null }[]>(() => {
    const settings = this.modelConfig.settings();
    if (!settings) {
      return MODEL_ROLES.map(({ label }) => ({ role: label, model: null }));
    }
    return MODEL_ROLES.map(({ label, key }) => ({
      role: label,
      model: settings[key] || (settings.useCloudServices ? settings.cloudModel : null) || null,
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

  toggleLocal(): void {
    this.ollamaMessage.set(null);
    this.localOpen.update((open) => !open);
  }

  async startLocal(): Promise<void> {
    await this.runOllamaAction(() => startOllama(), 'Ollama is running.', true);
  }

  async stopLocal(): Promise<void> {
    await this.runOllamaAction(() => stopOllama(), 'Ollama stopped.', false);
  }

  private async runOllamaAction(
    action: () => Promise<unknown>,
    success: string,
    showBusy: boolean,
  ): Promise<void> {
    this.ollamaMessage.set(null);
    this.ollamaBusy.set(showBusy);
    try {
      await action();
      this.ollamaMessage.set({ kind: 'success', text: success });
    } catch (error) {
      this.ollamaMessage.set({ kind: 'error', text: formatUnknownError(error, 'Ollama action failed.') });
    } finally {
      this.ollamaBusy.set(false);
      await this.state.refresh();
    }
  }

  @HostListener('document:pointerdown', ['$event'])
  closeOnOutsidePointer(event: Event): void {
    const target = event.target as Node;
    const host = this.host.nativeElement;
    if (this.modelsOpen() && !host.querySelector('.models-anchor')?.contains(target)) {
      this.modelsOpen.set(false);
    }
    if (this.localOpen() && !host.querySelector('.local-anchor')?.contains(target)) {
      this.localOpen.set(false);
    }
  }

  @HostListener('document:keydown.escape')
  closeOnEscape(): void {
    this.modelsOpen.set(false);
    this.localOpen.set(false);
  }

  ngOnDestroy(): void {
    this.state.stop();
  }
}
