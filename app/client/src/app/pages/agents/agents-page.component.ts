// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

import {
  LucideClipboardCheck,
  LucideFileCheck,
  LucideLock,
  LucidePencil,
  LucidePlus,
  LucideStar,
  LucideStethoscope,
} from '@lucide/angular';

import { Agent, CAPABILITIES, CAPABILITY_LABELS, Capability } from '../../core/models/agent-types';
import { AgentsStateService } from '../../core/state/agents-state.service';
import { formatAppDateTime } from '../../core/utils/date-formatting';
import { ConfirmDialogComponent } from '../../components/confirm-dialog/confirm-dialog.component';
import { OverflowMenuComponent, OverflowMenuItem } from '../../components/overflow-menu/overflow-menu.component';

@Component({
  selector: 'app-agents-page',
  standalone: true,
  imports: [
    RouterLink,
    ConfirmDialogComponent,
    OverflowMenuComponent,
    LucideClipboardCheck,
    LucideFileCheck,
    LucideLock,
    LucidePencil,
    LucidePlus,
    LucideStar,
    LucideStethoscope,
  ],
  templateUrl: './agents-page.component.html',
  styleUrl: './agents-page.component.scss',
})
export class AgentsPageComponent implements OnInit {
  readonly state = inject(AgentsStateService);
  private readonly router = inject(Router);

  readonly capabilities = CAPABILITIES;
  readonly capabilityLabels = CAPABILITY_LABELS;
  readonly search = signal('');
  readonly pendingDelete = signal<Agent | null>(null);
  readonly busy = signal(false);
  readonly message = signal('');
  readonly messageIsError = signal(false);

  readonly visibleAgents = computed(() => {
    const query = this.search().trim().toLowerCase();
    return this.state
      .agents()
      .filter((agent) => !query || this.state.displayName(agent).toLowerCase().includes(query))
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

  defaultDescription(agent: Agent): string {
    if (!agent.capabilities.length) {
      return 'Each workflow selects an eligible fallback when one is available.';
    }
    const workflows = agent.capabilities.map((capability) => this.capabilityLabels[capability]);
    return `Preferred for ${workflows.join(', ')}. Other workflows select an eligible fallback when available.`;
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

  menuItems(agent: Agent): OverflowMenuItem[] {
    return [
      { id: 'duplicate', label: 'Duplicate' },
      {
        id: 'default',
        label: agent.capabilities.length ? 'Set as default' : 'Set as default (workflows need a fallback)',
        reason: this.defaultBlockedReason(agent),
      },
      { id: 'lock', label: agent.is_locked ? 'Unlock' : 'Lock' },
      { id: 'archive', label: agent.is_archived ? 'Restore' : 'Archive', reason: this.archiveBlockedReason(agent) },
      { id: 'delete', label: 'Delete', danger: true, reason: this.deleteBlockedReason(agent) },
    ];
  }

  onMenu(agent: Agent, action: string): void {
    switch (action) {
      case 'duplicate': void this.duplicate(agent); break;
      case 'default': void this.setDefault(agent); break;
      case 'lock': void this.toggleLock(agent); break;
      case 'archive': void this.toggleArchive(agent); break;
      case 'delete': this.pendingDelete.set(agent); break;
    }
  }

  edit(agent: Agent): void {
    void this.router.navigate(['/agents', agent.id]);
  }

  async duplicate(agent: Agent): Promise<void> {
    await this.run(() => this.state.duplicate(agent.id), `Duplicated "${this.state.displayName(agent)}".`);
  }

  async setDefault(agent: Agent): Promise<void> {
    const message = `"${this.state.displayName(agent)}" is now the default agent.`;
    await this.run(
      () => this.state.setDefault(agent.id),
      agent.capabilities.length ? message : `${message} ${this.defaultDescription(agent)}`,
    );
  }

  async toggleLock(agent: Agent): Promise<void> {
    await this.run(
      () => this.state.setLifecycle(agent.id, { is_locked: !agent.is_locked }),
      `${agent.is_locked ? 'Unlocked' : 'Locked'} "${this.state.displayName(agent)}".`,
    );
  }

  async toggleArchive(agent: Agent): Promise<void> {
    await this.run(
      () => this.state.setLifecycle(agent.id, { is_archived: !agent.is_archived }),
      `${agent.is_archived ? 'Restored' : 'Archived'} "${this.state.displayName(agent)}".`,
    );
  }

  async confirmDelete(): Promise<void> {
    const agent = this.pendingDelete();
    this.pendingDelete.set(null);
    if (agent) await this.run(() => this.state.remove(agent.id), `Deleted "${this.state.displayName(agent)}".`);
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
