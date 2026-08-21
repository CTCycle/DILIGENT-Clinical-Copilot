import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, inject } from '@angular/core';
import { Router } from '@angular/router';
import { LucideSparkles } from '@lucide/angular';

import { ModalShellComponent } from '../../components/modal-shell/modal-shell.component';
import { DILI_ASSESSMENT_TOUR, TIPS_AND_TRICKS } from './guidance-content';
import { GuidanceTourService } from './guidance-tour.service';
import { TipAction } from './guidance.types';

@Component({
  selector: 'app-tips-and-tricks',
  standalone: true,
  imports: [CommonModule, ModalShellComponent, LucideSparkles],
  templateUrl: './tips-and-tricks.component.html',
  styleUrl: './tips-and-tricks.component.scss',
})
export class TipsAndTricksComponent {
  @Input() isOpen = false;
  @Output() closed = new EventEmitter<void>();

  readonly tips = TIPS_AND_TRICKS;
  private readonly router = inject(Router);
  private readonly tourService = inject(GuidanceTourService);

  close(): void {
    this.closed.emit();
  }

  handleAction(action: TipAction | undefined): void {
    if (!action) return;
    this.close();
    queueMicrotask(() => {
      if (action === 'tour') this.tourService.restart(DILI_ASSESSMENT_TOUR);
      if (action === 'configurations') void this.router.navigateByUrl('/model-config');
      if (action === 'sessions') void this.router.navigateByUrl('/clinical-sessions');
      if (action === 'data') void this.router.navigateByUrl('/data');
    });
  }
}
