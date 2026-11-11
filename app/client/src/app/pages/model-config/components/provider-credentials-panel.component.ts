// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output, signal } from '@angular/core';

import { AccessKeyProvider } from '../../../core/models/types';
import { AccessKeyModalComponent } from './access-key-modal.component';

export type CredentialProviderOption = {
  provider: AccessKeyProvider;
  label: string;
  description: string;
  logo: { src: string; alt: string } | null;
};

/** Cloud credential cards; "Manage keys" opens the shared access-key modal. */
@Component({
  selector: 'app-provider-credentials-panel',
  standalone: true,
  imports: [AccessKeyModalComponent],
  templateUrl: './provider-credentials-panel.component.html',
  styleUrl: './provider-credentials-panel.component.scss',
})
export class ProviderCredentialsPanelComponent {
  readonly providers = input.required<readonly CredentialProviderOption[]>();
  readonly selectedProvider = input<AccessKeyProvider | null>(null);
  readonly disabled = input(false);
  /** Modal title per credential scope; falls back to the option label. */
  readonly providerLabels = input<Partial<Record<AccessKeyProvider, string>>>({});

  /** Emitted when keys were created, activated or deleted in the modal. */
  readonly keysChanged = output<void>();

  readonly openProvider = signal<AccessKeyProvider | null>(null);
  readonly modalProviderLabel = computed(() => {
    const provider = this.openProvider() || 'openai';
    return this.providerLabels()[provider]
      || this.providers().find((option) => option.provider === provider)?.label
      || provider;
  });

  openKeys(provider: AccessKeyProvider): void {
    this.openProvider.set(provider);
  }

  closeKeys(): void {
    this.openProvider.set(null);
  }
}
