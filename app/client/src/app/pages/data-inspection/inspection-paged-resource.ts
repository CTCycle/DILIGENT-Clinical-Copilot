import { computed, signal } from '@angular/core';

const DEFAULT_BATCH_SIZE = 50;
const LOAD_MORE_THRESHOLD_PX = 180;
const DEFAULT_ROW_HEIGHT_PX = 44;
const DEFAULT_OVERSCAN_ROWS = 8;

export type InspectionPagedResponse<TItem> = {
  items: TItem[];
  total: number;
  offset?: number;
  limit?: number;
};

export type InspectionPagedLoader<TItem> = (params: {
  search: string;
  offset: number;
  limit: number;
}) => Promise<InspectionPagedResponse<TItem>>;

export class InspectionPagedResource<TItem> {
  readonly items = signal<TItem[]>([]);
  readonly total = signal(0);
  readonly loading = signal(false);
  readonly loadingMore = signal(false);
  readonly hasMore = signal(true);
  readonly error = signal<string | null>(null);
  readonly searchInput = signal('');
  readonly scrollTop = signal(0);
  readonly viewportHeight = signal(0);

  readonly visibleStartIndex = computed(() => {
    const itemCount = this.items().length;
    if (itemCount <= 0) return 0;
    const first = Math.floor(this.scrollTop() / this.rowHeightPx) - this.overscanRows;
    return Math.max(0, first);
  });

  readonly visibleEndIndex = computed(() => {
    const itemCount = this.items().length;
    if (itemCount <= 0) return 0;
    const rowsInView = Math.ceil(this.viewportHeight() / this.rowHeightPx) + this.overscanRows * 2;
    return Math.min(itemCount, this.visibleStartIndex() + Math.max(rowsInView, this.overscanRows * 2));
  });

  readonly visibleItems = computed(() =>
    this.items().slice(this.visibleStartIndex(), this.visibleEndIndex()),
  );

  readonly topPaddingPx = computed(() => this.visibleStartIndex() * this.rowHeightPx);

  readonly bottomPaddingPx = computed(() => {
    const itemCount = this.items().length;
    const hiddenRows = Math.max(0, itemCount - this.visibleEndIndex());
    return hiddenRows * this.rowHeightPx;
  });

  private nextOffset = 0;

  constructor(
    private readonly loader: InspectionPagedLoader<TItem>,
    private readonly fallbackErrorMessage: string,
    private readonly batchSize: number = DEFAULT_BATCH_SIZE,
    private readonly rowHeightPx: number = DEFAULT_ROW_HEIGHT_PX,
    private readonly overscanRows: number = DEFAULT_OVERSCAN_ROWS,
  ) {}

  async loadInitial(): Promise<void> {
    this.nextOffset = 0;
    this.items.set([]);
    this.total.set(0);
    this.hasMore.set(true);
    await this.loadMoreInternal(true);
  }

  async loadMore(): Promise<void> {
    await this.loadMoreInternal(false);
  }

  async updateSearch(value: string): Promise<void> {
    this.searchInput.set(value);
    await this.loadInitial();
  }

  handleScrollEvent(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    this.scrollTop.set(target.scrollTop);
    this.viewportHeight.set(target.clientHeight);

    if (!this.hasMore() || this.loading() || this.loadingMore()) {
      return;
    }
    const remaining = target.scrollHeight - (target.scrollTop + target.clientHeight);
    if (remaining <= LOAD_MORE_THRESHOLD_PX) {
      void this.loadMore();
    }
  }

  private async loadMoreInternal(isInitialLoad: boolean): Promise<void> {
    if (!isInitialLoad && (!this.hasMore() || this.loadingMore() || this.loading())) {
      return;
    }

    if (isInitialLoad) {
      this.loading.set(true);
      this.error.set(null);
    } else {
      this.loadingMore.set(true);
    }

    try {
      const payload = await this.loader({
        search: this.searchInput().trim(),
        offset: this.nextOffset,
        limit: this.batchSize,
      });
      const current = isInitialLoad ? [] : this.items();
      const nextItems = payload.items ?? [];
      this.items.set([...current, ...nextItems]);
      this.total.set(payload.total ?? this.items().length);
      this.nextOffset = this.items().length;
      this.hasMore.set(this.nextOffset < this.total());
      if (isInitialLoad && this.items().length === 0) {
        this.hasMore.set(false);
      }
    } catch (error) {
      if (isInitialLoad) {
        this.items.set([]);
        this.total.set(0);
        this.hasMore.set(false);
      }
      this.error.set(error instanceof Error ? error.message : this.fallbackErrorMessage);
    } finally {
      if (isInitialLoad) {
        this.loading.set(false);
      } else {
        this.loadingMore.set(false);
      }
    }
  }
}

