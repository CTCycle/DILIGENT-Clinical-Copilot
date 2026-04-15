import { signal } from '@angular/core';

const PAGE_LIMIT = 10;

export type InspectionPagedResponse<TItem> = {
  items: TItem[];
  total: number;
};

export type InspectionPagedLoader<TItem> = (params: {
  search: string;
  offset: number;
  limit: number;
}) => Promise<InspectionPagedResponse<TItem>>;

export class InspectionPagedResource<TItem> {
  readonly items = signal<TItem[]>([]);
  readonly total = signal(0);
  readonly offset = signal(0);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly searchInput = signal('');

  constructor(
    private readonly loader: InspectionPagedLoader<TItem>,
    private readonly fallbackErrorMessage: string,
  ) {}

  async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const payload = await this.loader({
        search: this.searchInput(),
        offset: this.offset(),
        limit: PAGE_LIMIT,
      });
      this.items.set(payload.items);
      this.total.set(payload.total);
    } catch (error) {
      this.items.set([]);
      this.total.set(0);
      this.error.set(error instanceof Error ? error.message : this.fallbackErrorMessage);
    } finally {
      this.loading.set(false);
    }
  }

  async updateSearch(value: string): Promise<void> {
    this.searchInput.set(value);
    this.offset.set(0);
    await this.load();
  }

  async page(direction: -1 | 1): Promise<void> {
    const next = this.offset() + direction * PAGE_LIMIT;
    if (next < 0 || next >= this.total()) {
      return;
    }
    this.offset.set(next);
    await this.load();
  }
}

