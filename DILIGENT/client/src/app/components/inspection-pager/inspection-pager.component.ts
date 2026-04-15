import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-inspection-pager',
  standalone: true,
  template: `
    <div class="inspection-pager">
      <div class="inspection-pager-actions">
        <button
          class="btn btn-secondary inspection-mini-btn"
          type="button"
          (click)="previous.emit()"
          [disabled]="offset() === 0"
        >
          Previous
        </button>
        <span class="inspection-pager-range">
          {{ rangeStart() }}-{{ rangeEnd() }} / {{ total() }}
        </span>
        <button
          class="btn btn-secondary inspection-mini-btn"
          type="button"
          (click)="next.emit()"
          [disabled]="offset() + pageSize() >= total()"
        >
          Next
        </button>
      </div>
    </div>
  `,
})
export class InspectionPagerComponent {
  readonly offset = input.required<number>();
  readonly total = input.required<number>();
  readonly pageSize = input(10);

  readonly previous = output<void>();
  readonly next = output<void>();

  rangeStart(): number {
    return this.total() <= 0 ? 0 : this.offset() + 1;
  }

  rangeEnd(): number {
    return Math.min(this.offset() + this.pageSize(), this.total());
  }
}
