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
    { path: '/settings/general', label: 'General', description: 'Application runtime behavior' },
    { path: '/settings/models', label: 'Models', description: 'Providers, roles, reasoning, and RAG' },
    { path: '/settings/data', label: 'Data Processing', description: 'Drug-name ingestion limits' },
    { path: '/settings/integrations', label: 'Integrations', description: 'LiverTox and RxNav runtime controls' },
    { path: '/settings/advanced', label: 'Advanced', description: 'Runtime timeouts and limits' },
  ] as const;
}
