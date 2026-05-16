import { signal } from '@angular/core';

export class InspectionDetailResource<TData> {
  readonly data = signal<TData | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  async load(request: () => Promise<TData>, fallbackErrorMessage: string): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    this.data.set(null);
    try {
      this.data.set(await request());
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : fallbackErrorMessage);
    } finally {
      this.loading.set(false);
    }
  }

  close(): void {
    this.data.set(null);
    this.error.set(null);
    this.loading.set(false);
  }
}
