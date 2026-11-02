// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { OllamaControl, SystemStatusResponse } from '../../core/services/system-status-api';
import { ModelConfigStateService } from '../../core/state/model-config-state.service';
import { SystemStatusStateService } from '../../core/state/system-status-state.service';
import { StatusBarComponent } from './status-bar.component';

function control(overrides: Partial<OllamaControl>): OllamaControl {
  return { state: 'stopped', managed: false, can_start: true, can_stop: false, executable_found: true, error: null, ...overrides };
}

describe('StatusBarComponent local Ollama popover', () => {
  let fixture: ComponentFixture<StatusBarComponent>;

  async function open(ollama: OllamaControl): Promise<HTMLElement> {
    const status = signal<SystemStatusResponse | null>({
      providers: { active: 0, total: 0 },
      ollama: ollama.state === 'running' ? 'reachable' : 'unreachable',
      ollama_control: ollama,
      cpu_percent: null,
      ram: null,
      vram: null,
    });
    TestBed.configureTestingModule({
      imports: [StatusBarComponent],
      providers: [
        {
          provide: SystemStatusStateService,
          useValue: { status, connection: signal('connected'), start: vi.fn(), stop: vi.fn(), refresh: vi.fn() },
        },
        { provide: ModelConfigStateService, useValue: { status: signal('ready'), settings: signal(null), load: vi.fn() } },
      ],
    });
    fixture = TestBed.createComponent(StatusBarComponent);
    fixture.detectChanges();
    fixture.componentInstance.toggleLocal();
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  const action = (root: HTMLElement) => root.querySelector<HTMLButtonElement>('.local-action');

  afterEach(() => {
    fixture?.destroy();
    TestBed.resetTestingModule();
  });

  it('offers Start when stopped', async () => {
    const root = await open(control({}));
    expect(action(root)?.textContent?.trim()).toBe('Start');
    expect(action(root)?.disabled).toBe(false);
  });

  it('disables the action while the backend reports starting', async () => {
    const root = await open(control({ state: 'starting', can_start: false }));
    expect(action(root)?.disabled).toBe(true);
    expect(action(root)?.textContent).toContain('Starting');
  });

  it('shows the error and a Retry action after a failed launch', async () => {
    const root = await open(control({ state: 'error', error: 'Ollama exited unexpectedly.' }));
    expect(root.querySelector('.local-detail')?.textContent).toContain('Ollama exited unexpectedly.');
    expect(action(root)?.textContent?.trim()).toBe('Retry');
  });

  it('offers Stop only for a DILIGENT-managed running instance', async () => {
    const root = await open(control({ state: 'running', managed: true, can_start: false, can_stop: true }));
    expect(action(root)?.textContent?.trim()).toBe('Stop');
  });

  it('offers no action for an externally managed running instance', async () => {
    const root = await open(control({ state: 'running', can_start: false }));
    expect(action(root)).toBeNull();
  });
});
