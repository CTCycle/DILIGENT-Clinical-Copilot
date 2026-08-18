import { Component, EventEmitter, Input, Output, inject, signal } from '@angular/core';

import { GuidanceStateService } from './guidance-state.service';
import { GuidanceId } from './guidance.types';

@Component({
  selector: 'app-feature-tip',
  standalone: true,
  templateUrl: './feature-tip.component.html',
  styleUrl: './feature-tip.component.scss',
})
export class FeatureTipComponent {
  @Input({ required: true }) guidanceId!: GuidanceId;
  @Input() version = 1;
  @Input() title = '';
  @Input() body = '';
  @Input() actionLabel = 'Show me';
  @Input() secondaryLabel = '';
  @Output() primaryAction = new EventEmitter<void>();
  @Output() secondaryAction = new EventEmitter<void>();

  readonly visible = signal(false);
  private readonly stateService = inject(GuidanceStateService);
  private initialized = false;

  ngOnInit(): void {
    this.initialized = true;
    if (this.stateService.shouldShow(this.guidanceId, this.version)) {
      this.visible.set(true);
      this.stateService.markSeen(this.guidanceId, this.version);
    }
  }

  dismiss(): void {
    if (!this.initialized) return;
    this.stateService.dismiss(this.guidanceId, this.version);
    this.visible.set(false);
  }

  showMore(): void {
    this.stateService.markSeen(this.guidanceId, this.version);
    this.primaryAction.emit();
  }

  openSecondaryAction(): void {
    this.stateService.markSeen(this.guidanceId, this.version);
    this.secondaryAction.emit();
  }
}
