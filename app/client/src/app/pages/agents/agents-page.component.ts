// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

import {
  LucideArchive,
  LucideArchiveRestore,
  LucideClipboardCheck,
  LucideCopy,
  LucideFileCheck,
  LucideLock,
  LucideLockOpen,
  LucidePencil,
  LucidePlus,
  LucideStar,
  LucideStethoscope,
  LucideTrash2,
} from '@lucide/angular';

import { Agent, CAPABILITIES, CAPABILITY_LABELS, Capability } from '../../core/models/agent-types';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { formatAppDateTime } from '../../core/utils/date-formatting';
import { ConfirmDialogComponent } from '../../components/confirm-dialog/confirm-dialog.component';
import { IconButtonComponent } from '../../components/icon-button/icon-button.component';
import { ProvidersResourcesComponent } from './providers-resources/providers-resources.component';

type StudioTab = 'agents' | 'providers';
const TABS: ReadonlyArray<{ id: StudioTab; label: string }> = [
  { id: 'agents', label: 'My Agents' },
  { id: 'providers', label: 'Providers & Resources' },
];

@Component({
  selector: 'app-agents-page',
  standalone: true,
  imports: [
    RouterLink,
    ConfirmDialogComponent,
    IconButtonComponent,
    ProvidersResourcesComponent,
    LucideArchive,
    LucideArchiveRestore,
    LucideClipboardCheck,
    LucideCopy,
    LucideFileCheck,
    LucideLock,
    LucideLockOpen,
    LucidePencil,
    LucidePlus,
    LucideStar,
    LucideStethoscope,
    LucideTrash2,
  ],
  templateUrl: './agents-page.component.html',
  styleUrl: './agents-page.component.scss',
})
export class AgentsPageComponent implements OnInit {
  readonly state = inject(AgentsStateService);
  private readonly router = inject(Router);

  readonly tabs = TABS;
  readonly capabilities = CAPABILITIES;
  readonly capabilityLabels = CAPABILITY_LABELS;
  readonly tab = signal<StudioTab>('agents');
  readonly providersVisited = signal(false);
  readonly search = signal('');
  readonly pendingDelete = signal<Agent | null>(null);
  readonly busy = signal(false);
  readonly message = signal('');
  readonly messageIsError = signal(false);

  readonly visibleAgents = computed(() => {
    const query = this.search().trim().toLowerCase();
    return this.state
      .agents()
      .filter((agent) => !query || agent.name.toLowerCase().includes(query))
      .sort((a, b) => Number(b.is_default) - Number(a.is_default) || a.name.localeCompare(b.name));
  });

  ngOnInit(): void {
    this.state.load({ force: true }).catch(() => undefined);
  }

  formatDate(value: string | null): string {
    return formatAppDateTime(value, 'Unknown date');
  }

  hasCapability(agent: Agent, capability: Capability): boolean {
    return agent.capabilities.includes(capability);
  }

  /** Why the action is unavailable for this agent, or null when it can run. */
  defaultBlockedReason(agent: Agent): string | null {
    if (agent.is_default) return 'Already the default agent';
    if (agent.is_archived) return 'Restore this agent before making it the default';
    if (!agent.is_active) return 'Reactivate this agent before making it the default';
    return null;
  }

  archiveBlockedReason(agent: Agent): string | null {
    return agent.is_default && !agent.is_archived
      ? 'Make another agent the default before archiving this one'
      : null;
  }

  deleteBlockedReason(agent: Agent): string | null {
    if (agent.is_default) return 'The default agent cannot be deleted';
    if (agent.is_locked) return 'Unlock this agent before deleting it';
    if (agent.reference_count > 0) {
      return `Used by ${agent.reference_count} saved run${agent.reference_count === 1 ? '' : 's'}; archive it instead`;
    }
    return null;
  }

  selectTab(id: StudioTab): void {
    this.tab.set(id);
    if (id === 'providers') this.providersVisited.set(true);
  }

  onTabKeydown(event: KeyboardEvent, index: number): void {
    const last = this.tabs.length - 1;
    const next =
      event.key === 'ArrowRight' ? (index + 1) % this.tabs.length
        : event.key === 'ArrowLeft' ? (index - 1 + this.tabs.length) % this.tabs.length
        : event.key === 'Home' ? 0
        : event.key === 'End' ? last
        : null;
    if (next === null) return;
    event.preventDefault();
    this.selectTab(this.tabs[next].id);
    queueMicrotask(() => document.getElementById(`agent-studio-tab-${this.tabs[next].id}`)?.focus());
  }

  edit(agent: Agent): void {
    void this.router.navigate(['/agents', agent.id]);
  }

  async duplicate(agent: Agent): Promise<void> {
    await this.run(() => this.state.duplicate(agent.id), `Duplicated "${agent.name}".`);
  }

  async setDefault(agent: Agent): Promise<void> {
    await this.run(() => this.state.setDefault(agent.id), `"${agent.name}" is now the default agent.`);
  }

  async toggleLock(agent: Agent): Promise<void> {
    await this.run(
      () => this.state.setLifecycle(agent.id, { is_locked: !agent.is_locked }),
      agent.is_locked ? `Unlocked "${agent.name}".` : `Locked "${agent.name}".`,
    );
  }

  async toggleArchive(agent: Agent): Promise<void> {
    await this.run(
      () => this.state.setLifecycle(agent.id, { is_archived: !agent.is_archived }),
      agent.is_archived ? `Restored "${agent.name}".` : `Archived "${agent.name}".`,
    );
  }

  async confirmDelete(): Promise<void> {
    const agent = this.pendingDelete();
    this.pendingDelete.set(null);
    if (agent) await this.run(() => this.state.remove(agent.id), `Deleted "${agent.name}".`);
  }

  private async run<T>(action: () => Promise<T>, success: string): Promise<T | null> {
    if (this.busy()) return null;
    this.busy.set(true);
    this.message.set('');
    try {
      const result = await action();
      this.messageIsError.set(false);
      this.message.set(success);
      return result;
    } catch (error) {
      this.messageIsError.set(true);
      this.message.set((error instanceof Error ? error.message : String(error)).replace(/^\[ERROR\]\s*/, ''));
      return null;
    } finally {
      this.busy.set(false);
    }
  }
}
