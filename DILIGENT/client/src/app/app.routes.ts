import { Routes } from '@angular/router';

import { DiliAgentPageComponent } from './pages/dili-agent/dili-agent-page.component';
import { DataInspectionPageComponent } from './pages/data-inspection/data-inspection-page.component';
import { ModelConfigPageComponent } from './pages/model-config/model-config-page.component';

export const routes: Routes = [
  { path: '', component: DiliAgentPageComponent },
  { path: 'data', component: DataInspectionPageComponent },
  { path: 'model-config', component: ModelConfigPageComponent },
  { path: '**', redirectTo: '' },
];
