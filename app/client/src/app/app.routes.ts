import { Routes } from '@angular/router';

import { DiliAgentPageComponent } from './pages/dili-agent/dili-agent-page.component';
import { DataInspectionPageComponent } from './pages/data-inspection/data-inspection-page.component';
import { ClinicalSessionsPageComponent } from './pages/clinical-sessions/clinical-sessions-page.component';
import { ModelConfigPageComponent } from './pages/model-config/model-config-page.component';
import { PatientTimetablePageComponent } from './pages/patient-timetable/patient-timetable-page.component';

export const routes: Routes = [
  { path: '', component: DiliAgentPageComponent },
  { path: 'clinical-sessions', component: ClinicalSessionsPageComponent },
  { path: 'data', component: DataInspectionPageComponent },
  { path: 'sessions/:sessionId/timetable', component: PatientTimetablePageComponent },
  { path: 'model-config', component: ModelConfigPageComponent },
  { path: '**', redirectTo: '' },
];
