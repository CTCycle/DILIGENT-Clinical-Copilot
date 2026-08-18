import { Component, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';

import {
  AppStateService,
  PageId,
  resolvePageIdFromPath,
  resolvePathFromPage,
} from './core/state/app-state.service';
import { DiliJobTrackerService } from './core/services/dili-job-tracker.service';
import { InspectionUpdateJobTrackerService } from './core/state/inspection-update-job-tracker.service';
import { NavSidebarComponent } from './components/nav-sidebar/nav-sidebar.component';
import { GuidedTourComponent } from './core/guidance/guided-tour.component';
import { TipsAndTricksComponent } from './core/guidance/tips-and-tricks.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, NavSidebarComponent, GuidedTourComponent, TipsAndTricksComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  readonly stateService = inject(AppStateService);
  readonly tipsOpen = signal(false);
  private readonly diliJobTracker = inject(DiliJobTrackerService);
  private readonly inspectionUpdateTracker = inject(InspectionUpdateJobTrackerService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  constructor() {
    void this.diliJobTracker;
    void this.inspectionUpdateTracker;
    this.router.events
      .pipe(
        filter((event): event is NavigationEnd => event instanceof NavigationEnd),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((event) => {
        this.stateService.setActivePage(resolvePageIdFromPath(event.urlAfterRedirects));
      });
  }

  navigateToPage(pageId: PageId): void {
    const nextPath = resolvePathFromPage(pageId);
    if (window.location.pathname !== nextPath) {
      void this.router.navigateByUrl(nextPath);
    }
    this.stateService.setActivePage(pageId);
  }

  openTips(): void {
    this.tipsOpen.set(true);
  }

}
