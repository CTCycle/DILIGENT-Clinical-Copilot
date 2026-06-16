import { Component, DestroyRef, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';

import {
  AppStateService,
  PageId,
  resolvePageIdFromPath,
  resolvePathFromPage,
} from './core/state/app-state.service';
import { NavSidebarComponent } from './components/nav-sidebar/nav-sidebar.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, NavSidebarComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  readonly stateService = inject(AppStateService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  constructor() {
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

}
