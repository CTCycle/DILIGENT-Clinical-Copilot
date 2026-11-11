// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { CloudProvider } from '../../../core/models/types';
import { MODEL_FILTERS } from '../model-catalog';
import { ModelFilterKey } from '../model-config.types';

/**
 * Provider catalog shell: heading, search, provider select, refresh, local filters,
 * catalog status notes and footer. The model table is projected by the host.
 */
@Component({
  selector: 'app-provider-catalog-panel',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './provider-catalog-panel.component.html',
  styleUrl: './provider-catalog-panel.component.scss',
})
export class ProviderCatalogPanelComponent {
  readonly useCloud = input(false);
  /** Whole panel is dimmed/aria-disabled (saving or loading). */
  readonly disabled = input(false);
  /** Provider select and refresh are disabled (saving, loading or catalog request in flight). */
  readonly actionsDisabled = input(false);
  readonly searchQuery = input('');
  readonly provider = input<CloudProvider | null>(null);
  readonly providerOptions = input<readonly { value: CloudProvider; label: string }[]>([]);
  readonly activeFilters = input.required<Record<ModelFilterKey, boolean>>();
  readonly localCatalogStatus = input('not_loaded');
  readonly localCatalogMessage = input<string | null>(null);
  readonly cloudCatalogStatus = input<string | null>(null);
  readonly cloudCatalogMessage = input('');
  readonly rangeStart = input(0);
  readonly rangeEnd = input(0);
  readonly totalCount = input(0);

  readonly searchQueryChange = output<string>();
  readonly providerChange = output<CloudProvider>();
  readonly refreshRequested = output<void>();
  readonly filterToggled = output<ModelFilterKey>();
  readonly filtersCleared = output<void>();

  readonly filters = MODEL_FILTERS;
  readonly noFilterActive = computed(() =>
    this.filters.every((filter) => !this.activeFilters()[filter.key]),
  );
}
