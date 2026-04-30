import { Component, HostListener, inject } from '@angular/core';
import { Router, RouterOutlet } from '@angular/router';

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

  navigateToPage(pageId: PageId): void {
    const nextPath = resolvePathFromPage(pageId);
    if (window.location.pathname !== nextPath) {
      void this.router.navigateByUrl(nextPath);
    }
    this.stateService.setActivePage(pageId);
  }

  @HostListener('window:popstate')
  onPopState(): void {
    this.stateService.setActivePage(resolvePageIdFromPath(window.location.pathname));
  }
}
