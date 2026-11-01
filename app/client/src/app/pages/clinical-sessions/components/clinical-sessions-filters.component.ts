// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, ElementRef, HostListener, computed, inject, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideCalendar,
  LucideChevronDown,
  LucideRefreshCw,
  LucideSearch,
  LucideX,
} from '@lucide/angular';

import {
  InspectionSessionDateField,
  InspectionSessionSort,
  InspectionSessionStatus,
} from '../../../core/models/inspection-types';
import { SESSION_STATUSES, sessionStatusLabel } from '../../../core/utils/session-status';
import { CACHE_WINDOW_DAYS, SessionDatePreset } from '../clinical-session-filters';
import { ClinicalSessionsStore } from '../clinical-sessions-store';

type Panel = 'status' | 'date';

const DATE_PRESETS: ReadonlyArray<{ value: SessionDatePreset; label: string }> = [
  { value: 'any', label: 'Any date' },
  { value: 'today', label: 'Today' },
  { value: 'last7', label: 'Last 7 days' },
  { value: 'last30', label: 'Last 30 days' },
  { value: 'custom', label: 'Custom range' },
];

const DATE_FIELDS: ReadonlyArray<{ value: InspectionSessionDateField; label: string }> = [
  { value: 'created', label: 'Created' },
  { value: 'updated', label: 'Updated' },
  { value: 'visit', label: 'Visit date' },
];

@Component({
  selector: 'app-clinical-sessions-filters',
  standalone: true,
  imports: [FormsModule, LucideCalendar, LucideChevronDown, LucideRefreshCw, LucideSearch, LucideX],
  templateUrl: './clinical-sessions-filters.component.html',
  styleUrl: './clinical-sessions-filters.component.scss',
})
export class ClinicalSessionsFiltersComponent {
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly store = inject(ClinicalSessionsStore);
  readonly refresh = output<void>();

  readonly statusOptions = SESSION_STATUSES;
  readonly statusLabel = sessionStatusLabel;
  readonly datePresets = DATE_PRESETS;
  readonly dateFields = DATE_FIELDS;
  readonly cacheDays = CACHE_WINDOW_DAYS;
  readonly sortOptions: ReadonlyArray<{ value: InspectionSessionSort; label: string }> = [
    { value: 'updated_desc', label: 'Recently updated' },
    { value: 'created_desc', label: 'Newest first' },
    { value: 'created_asc', label: 'Oldest first' },
  ];
  readonly openPanel = signal<Panel | null>(null);

  readonly filters = this.store.filters;
  readonly statusSummary = computed(() => {
    const count = this.filters().statuses.length;
    return count ? `Status · ${count}` : 'Status';
  });
  readonly hasChips = computed(() => (
    this.filters().statuses.length > 0 || this.filters().date.preset !== 'any'
  ));
  readonly dateChipLabel = computed(() => {
    const { field, preset, from, to } = this.filters().date;
    const fieldLabel = DATE_FIELDS.find((option) => option.value === field)?.label ?? '';
    const range = preset === 'custom'
      ? `${from || '…'} – ${to || '…'}`
      : DATE_PRESETS.find((option) => option.value === preset)?.label ?? '';
    return `${fieldLabel}: ${range}`;
  });

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: Event): void {
    if (this.openPanel() && !this.host.nativeElement.contains(event.target as Node)) {
      this.openPanel.set(null);
    }
  }

  @HostListener('keydown.escape')
  onEscape(): void {
    this.openPanel.set(null);
  }

  togglePanel(panel: Panel): void {
    this.openPanel.update((current) => (current === panel ? null : panel));
  }

  isStatusSelected(status: InspectionSessionStatus): boolean {
    return this.filters().statuses.includes(status);
  }

  /** Statuses combine with OR; toggling one never affects the others. */
  toggleStatus(status: InspectionSessionStatus): void {
    const current = this.filters().statuses;
    const next = current.includes(status)
      ? current.filter((value) => value !== status)
      : SESSION_STATUSES.filter((value) => value === status || current.includes(value));
    this.store.setFilters({ statuses: next });
  }

  clearStatuses(): void {
    this.store.setFilters({ statuses: [] });
  }

  setQuery(query: string): void {
    this.store.setFilters({ query });
  }

  setSort(sort: InspectionSessionSort): void {
    this.store.setFilters({ sort });
  }

  setDatePreset(preset: SessionDatePreset): void {
    this.store.setFilters({ date: { ...this.filters().date, preset } });
  }

  setDateField(field: InspectionSessionDateField): void {
    this.store.setFilters({ date: { ...this.filters().date, field } });
  }

  setCustomDate(bound: 'from' | 'to', value: string): void {
    this.store.setFilters({ date: { ...this.filters().date, preset: 'custom', [bound]: value } });
  }

  clearDate(): void {
    this.store.setFilters({ date: { ...this.filters().date, preset: 'any', from: '', to: '' } });
  }
}
