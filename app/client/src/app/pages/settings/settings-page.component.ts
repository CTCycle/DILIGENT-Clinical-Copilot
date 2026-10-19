// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-settings-page',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, RouterOutlet],
  templateUrl: './settings-page.component.html',
  styleUrl: './settings-page.component.scss',
})
export class SettingsPageComponent {
  readonly navItems = [
    { path: '/settings/general', label: 'General' },
    { path: '/settings/models', label: 'Models' },
    { path: '/settings/data', label: 'Data Processing' },
    { path: '/settings/integrations', label: 'Integrations' },
    { path: '/settings/matching', label: 'Drug Matching' },
    { path: '/settings/advanced', label: 'Advanced' },
  ] as const;
}
