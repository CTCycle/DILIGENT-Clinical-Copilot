// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./pages/dili-agent/dili-agent-page.component').then((m) => m.DiliAgentPageComponent), title: 'DILI Agent | DILIGENT' },
  { path: 'report/:sessionId', loadComponent: () => import('./pages/dili-report/dili-report-page.component').then((m) => m.DiliReportPageComponent), title: 'DILI Report | DILIGENT' },
  { path: 'clinical-sessions', loadComponent: () => import('./pages/clinical-sessions/clinical-sessions-page.component').then((m) => m.ClinicalSessionsPageComponent), title: 'Clinical Sessions | DILIGENT' },
  { path: 'agents', loadComponent: () => import('./pages/agents/agents-page.component').then((m) => m.AgentsPageComponent), title: 'Agents | DILIGENT' },
  {
    path: 'agents/:id',
    loadComponent: () => import('./pages/agents/agent-editor.component').then((m) => m.AgentEditorComponent),
    canDeactivate: [(component: { canLeave(): boolean | Promise<boolean> }) => component.canLeave()],
    title: 'Agent Editor | DILIGENT',
  },
  { path: 'data', loadComponent: () => import('./pages/data-inspection/data-inspection-page.component').then((m) => m.DataInspectionPageComponent), title: 'Knowledge | DILIGENT' },
  { path: 'sessions/:sessionId/timetable/:timelineId', loadComponent: () => import('./pages/patient-timetable/patient-timetable-page.component').then((m) => m.PatientTimetablePageComponent), title: 'Patient Timeline | DILIGENT' },
  { path: 'sessions/:sessionId/timetable', loadComponent: () => import('./pages/patient-timetable/patient-timetable-page.component').then((m) => m.PatientTimetablePageComponent), title: 'Patient Timeline | DILIGENT' },
  {
    path: 'settings',
    loadComponent: () => import('./pages/settings/settings-page.component').then((m) => m.SettingsPageComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'general' },
      {
        path: 'general',
        loadComponent: () => import('./pages/settings/components/operational-settings-page.component').then((m) => m.OperationalSettingsPageComponent),
        data: { settingsSection: 'general' },
        title: 'General Settings | DILIGENT',
      },
      {
        path: 'data',
        loadComponent: () => import('./pages/settings/components/operational-settings-page.component').then((m) => m.OperationalSettingsPageComponent),
        data: { settingsSection: 'data' },
        title: 'Data Processing Settings | DILIGENT',
      },
      {
        path: 'integrations',
        loadComponent: () => import('./pages/settings/components/operational-settings-page.component').then((m) => m.OperationalSettingsPageComponent),
        data: { settingsSection: 'integrations' },
        title: 'Integration Settings | DILIGENT',
      },
      {
        path: 'providers',
        loadComponent: () => import('./pages/settings/components/providers-resources/providers-resources.component').then((m) => m.ProvidersResourcesComponent),
        title: 'Providers & Models Settings | DILIGENT',
      },
      {
        path: 'matching',
        loadComponent: () => import('./pages/settings/components/operational-settings-page.component').then((m) => m.OperationalSettingsPageComponent),
        data: { settingsSection: 'matching' },
        title: 'Drug Matching Settings | DILIGENT',
      },
      {
        path: 'advanced',
        loadComponent: () => import('./pages/settings/components/operational-settings-page.component').then((m) => m.OperationalSettingsPageComponent),
        data: { settingsSection: 'advanced' },
        title: 'Advanced Settings | DILIGENT',
      },
      {
        path: 'database',
        loadComponent: () => import('./pages/settings/components/database-settings.component').then((m) => m.DatabaseSettingsComponent),
        title: 'Database Settings | DILIGENT',
      },
    ],
  },
  { path: '**', redirectTo: '' },
];
