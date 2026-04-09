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
    { pageId: 'data-inspection', label: 'Data Inspection' },
    { pageId: 'model-config', label: 'Model Configurations' },
  ];

  onNavigate(pageId: PageId): void {
    this.navigate.emit(pageId);
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
