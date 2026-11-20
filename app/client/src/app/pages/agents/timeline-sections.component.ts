// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import { AgentModelRef, TimelineConfig } from '../../core/models/agent-types';
import { AgentModelFieldComponent } from './agent-model-field.component';
import { AgentModelsSectionComponent } from './agent-models-section.component';
import { ModelCatalog, ProviderOption } from './agent-form';

/** Model of the Timeline capability; the extraction itself is scripted. */
@Component({
  selector: 'app-timeline-sections',
  standalone: true,
  imports: [AgentModelFieldComponent, AgentModelsSectionComponent],
  styleUrl: './agent-sections.scss',
  template: `
    <fieldset class="plain" [disabled]="disabled()">
      <app-agent-models-section heading="Model" headingId="timeline-model-heading" [disabled]="disabled()"
        [refs]="[config().model]">
        <app-agent-model-field label="Timeline model" [ref]="config().model"
          [providerOptions]="providerOptions()" [catalog]="catalog()" (refChange)="setModel($event)" />
      </app-agent-models-section>
    </fieldset>
  `,
})
export class TimelineSectionsComponent {
  readonly config = input.required<TimelineConfig>();
  readonly providerOptions = input<readonly ProviderOption[]>([]);
  readonly catalog = input<ModelCatalog>({});
  readonly disabled = input(false);
  readonly configChange = output<TimelineConfig>();

  setModel(ref: AgentModelRef): void {
    const draft = structuredClone(this.config());
    draft.model = ref;
    this.configChange.emit(draft);
  }
}
