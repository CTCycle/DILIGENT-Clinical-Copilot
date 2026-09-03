import { Routes } from '@angular/router';

import { DiliAgentPageComponent } from './pages/dili-agent/dili-agent-page.component';
import { DataInspectionPageComponent } from './pages/data-inspection/data-inspection-page.component';
import { ClinicalSessionsPageComponent } from './pages/clinical-sessions/clinical-sessions-page.component';
import { ModelConfigPageComponent } from './pages/model-config/model-config-page.component';
import { PatientTimetablePageComponent } from './pages/patient-timetable/patient-timetable-page.component';

export const routes: Routes = [
  { path: '', component: DiliAgentPageComponent, title: 'DILI Agent | DILIGENT' },
  { path: 'clinical-sessions', component: ClinicalSessionsPageComponent, title: 'Clinical Sessions | DILIGENT' },
  { path: 'data', component: DataInspectionPageComponent, title: 'Knowledge Base | DILIGENT' },
  { path: 'sessions/:sessionId/timetable/:timelineId', component: PatientTimetablePageComponent, title: 'Patient Timeline | DILIGENT' },
  { path: 'sessions/:sessionId/timetable', component: PatientTimetablePageComponent, title: 'Patient Timeline | DILIGENT' },
  { path: 'model-config', component: ModelConfigPageComponent, title: 'Configuration | DILIGENT' },
  { path: '**', redirectTo: '' },
];
