import {
  AfterViewInit,
  Component,
  ElementRef,
  HostListener,
  Input,
  OnDestroy,
  ViewChild,
  inject,
  signal,
} from '@angular/core';

let nextPopoverId = 0;
type HelpPopoverPlacement = 'auto' | 'top' | 'bottom' | 'left' | 'right';

@Component({
  selector: 'app-help-popover',
  standalone: true,
  templateUrl: './help-popover.component.html',
  styleUrl: './help-popover.component.scss',
})
export class HelpPopoverComponent implements AfterViewInit, OnDestroy {
  @Input() title = 'More about this control';
  @Input() body = '';
  @Input() ariaLabel = 'More information';
  @Input() placement: HelpPopoverPlacement = 'auto';

  @ViewChild('trigger') private trigger?: ElementRef<HTMLButtonElement>;
  @ViewChild('panel') private panel?: ElementRef<HTMLElement>;

  readonly open = signal(false);
  readonly panelPosition = signal({ top: 0, left: 0 });
  readonly panelId = `guidance-popover-${++nextPopoverId}`;
  private readonly host = inject(ElementRef<HTMLElement>);
  private readonly onViewportChange = (): void => this.positionPanel();

  ngAfterViewInit(): void {
    globalThis.addEventListener('resize', this.onViewportChange);
    globalThis.addEventListener('scroll', this.onViewportChange, true);
  }

  ngOnDestroy(): void {
    globalThis.removeEventListener('resize', this.onViewportChange);
    globalThis.removeEventListener('scroll', this.onViewportChange, true);
  }

  toggle(): void {
    if (this.open()) {
      this.close();
      return;
    }
    this.open.set(true);
    queueMicrotask(() => {
      this.positionPanel();
      this.panel?.nativeElement.focus();
    });
    globalThis.setTimeout(() => {
      if (!this.open()) return;
      this.positionPanel();
      this.panel?.nativeElement.focus();
    }, 0);
  }

  close(): void {
    if (!this.open()) return;
    this.open.set(false);
    queueMicrotask(() => this.trigger?.nativeElement.focus());
  }

  @HostListener('document:pointerdown', ['$event'])
  handleOutsidePointer(event: PointerEvent): void {
    if (!this.open()) return;
    const target = event.target;
    if (target instanceof Node && !this.host.nativeElement.contains(target)) {
      this.close();
    }
  }

  @HostListener('document:keydown', ['$event'])
  handleKeydown(event: KeyboardEvent): void {
    if (!this.open()) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      this.close();
    }
  }

  private positionPanel(): void {
    if (!this.open() || !this.trigger || !this.panel) return;
    const triggerRect = this.trigger.nativeElement.getBoundingClientRect();
    const panelElement = this.panel.nativeElement;
    const viewportMargin = 12;
    const gap = 8;
    const panelWidth = Math.min(340, Math.max(240, window.innerWidth - (viewportMargin * 2)));
    const panelHeight = panelElement.offsetHeight || 180;
    const spaceBelow = window.innerHeight - triggerRect.bottom - viewportMargin;
    const spaceAbove = triggerRect.top - viewportMargin;
    const top = spaceBelow >= panelHeight + gap || spaceBelow >= spaceAbove
      ? triggerRect.bottom + gap
      : triggerRect.top - panelHeight - gap;
    const autoPosition = {
      top: Math.max(viewportMargin, Math.min(top, window.innerHeight - panelHeight - viewportMargin)),
      left: Math.min(
        Math.max(viewportMargin, triggerRect.left),
        Math.max(viewportMargin, window.innerWidth - panelWidth - viewportMargin),
      ),
    };
    if (this.placement === 'auto') {
      this.panelPosition.set(autoPosition);
      return;
    }

    const preferredPosition = this.placement === 'top'
      ? { top: triggerRect.top - panelHeight - gap, left: triggerRect.left }
      : this.placement === 'right'
        ? { top: triggerRect.top, left: triggerRect.right + gap }
        : this.placement === 'left'
          ? { top: triggerRect.top, left: triggerRect.left - panelWidth - gap }
          : { top: triggerRect.bottom + gap, left: triggerRect.left };
    const hasRoom = this.placement === 'top'
      ? preferredPosition.top >= viewportMargin
      : this.placement === 'right'
        ? preferredPosition.left + panelWidth <= window.innerWidth - viewportMargin
        : this.placement === 'left'
          ? preferredPosition.left >= viewportMargin
          : preferredPosition.top + panelHeight <= window.innerHeight - viewportMargin;
    const left = Math.min(
      Math.max(viewportMargin, triggerRect.left),
      Math.max(viewportMargin, window.innerWidth - panelWidth - viewportMargin),
    );
    this.panelPosition.set({
      top: hasRoom
        ? Math.max(viewportMargin, Math.min(preferredPosition.top, window.innerHeight - panelHeight - viewportMargin))
        : autoPosition.top,
      left: hasRoom
        ? Math.max(viewportMargin, Math.min(preferredPosition.left, window.innerWidth - panelWidth - viewportMargin))
        : left,
    });
  }
}
