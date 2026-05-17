import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Output, inject } from '@angular/core';

import { AppStateService, PageId } from '../../core/state/app-state.service';

@Component({
  selector: 'app-nav-sidebar',
  imports: [CommonModule],
  templateUrl: './nav-sidebar.component.html',
  styleUrl: './nav-sidebar.component.scss',
})
export class NavSidebarComponent {
  @Output() navigate = new EventEmitter<PageId>();

  readonly stateService = inject(AppStateService);

  readonly navItems: Array<{ pageId: PageId; label: string }> = [
    { pageId: 'dili-agent', label: 'DILI Agent' },
    { pageId: 'clinical-sessions', label: 'Clinical Sessions' },
    { pageId: 'data-inspection', label: 'Data Inspection' },
    { pageId: 'model-config', label: 'Model Configurations' },
  ];

  onNavigate(pageId: PageId): void {
    this.navigate.emit(pageId);
  }

  onNavTabKeydown(event: KeyboardEvent, pageId: PageId): void {
    const index = this.navItems.findIndex((item) => item.pageId === pageId);
    if (index < 0) {
      return;
    }

    let nextIndex: number | null = null;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        nextIndex = (index + 1) % this.navItems.length;
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        nextIndex = (index - 1 + this.navItems.length) % this.navItems.length;
        break;
      case 'Home':
        nextIndex = 0;
        break;
      case 'End':
        nextIndex = this.navItems.length - 1;
        break;
      default:
        break;
    }

    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    this.onNavigate(this.navItems[nextIndex].pageId);
  }

  toggleTheme(): void {
    this.stateService.toggleTheme();
  }

  get isDarkTheme(): boolean {
    return this.stateService.state().theme === 'dark';
  }

  get themeAriaLabel(): string {
    return this.isDarkTheme ? 'Switch to light theme' : 'Switch to dark theme';
  }
}
