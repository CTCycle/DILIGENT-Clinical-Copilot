import { Routes } from '@angular/router';

import { DiliAgentPageComponent } from './pages/dili-agent/dili-agent-page.component';
import { DataInspectionPageComponent } from './pages/data-inspection/data-inspection-page.component';
import { ClinicalSessionsPageComponent } from './pages/clinical-sessions/clinical-sessions-page.component';
import { ModelConfigPageComponent } from './pages/model-config/model-config-page.component';
import { PatientTimetablePageComponent } from './pages/patient-timetable/patient-timetable-page.component';
import { SettingsPageComponent } from './pages/settings/settings-page.component';
import { OperationalSettingsPageComponent } from './pages/settings/components/operational-settings-page.component';

export const routes: Routes = [
  { path: '', component: DiliAgentPageComponent, title: 'DILI Agent | DILIGENT' },
  { path: 'clinical-sessions', component: ClinicalSessionsPageComponent, title: 'Clinical Sessions | DILIGENT' },
  { path: 'data', component: DataInspectionPageComponent, title: 'Knowledge Base | DILIGENT' },
  { path: 'sessions/:sessionId/timetable/:timelineId', component: PatientTimetablePageComponent, title: 'Patient Timeline | DILIGENT' },
  { path: 'sessions/:sessionId/timetable', component: PatientTimetablePageComponent, title: 'Patient Timeline | DILIGENT' },
  {
    path: 'settings',
    component: SettingsPageComponent,
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'general' },
      {
        path: 'general',
        component: OperationalSettingsPageComponent,
        data: { settingsSection: 'general' },
        title: 'General Settings | DILIGENT',
      },
      {
        path: 'models',
        component: ModelConfigPageComponent,
        title: 'Model Settings | DILIGENT',
      },
      {
        path: 'data',
        component: OperationalSettingsPageComponent,
        data: { settingsSection: 'data' },
        title: 'Data Processing Settings | DILIGENT',
      },
      {
        path: 'integrations',
        component: OperationalSettingsPageComponent,
        data: { settingsSection: 'integrations' },
        title: 'Integration Settings | DILIGENT',
      },
      {
        path: 'advanced',
        component: OperationalSettingsPageComponent,
        data: { settingsSection: 'advanced' },
        title: 'Advanced Settings | DILIGENT',
      },
    ],
  },
  { path: 'model-config', pathMatch: 'full', redirectTo: 'settings/models' },
  { path: '**', redirectTo: '' },
];
