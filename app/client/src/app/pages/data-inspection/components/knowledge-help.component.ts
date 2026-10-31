// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';

import { ModalShellComponent } from '../../../components/modal-shell/modal-shell.component';

type KnowledgeHelpSection = {
  id: string;
  title: string;
  items: readonly string[];
};

const KNOWLEDGE_HELP_SECTIONS: readonly KnowledgeHelpSection[] = [
  {
    id: 'sources',
    title: 'What each source contributes',
    items: [
      'Drug Catalog (RxNav): the RxNorm drug list used to recognise drug names in clinical notes.',
      'LiverTox: NIH hepatotoxicity monographs, used as bibliographic evidence for each detected drug.',
      'DILIrank: FDA drug-level DILI concern categories. Only drugs linked to a trusted LiverTox or RxNorm identity count as clinical evidence.',
      'Terminology: maps brand names, synonyms and international names to one canonical drug identity.',
      'RAG: a searchable index of your own documents, used for retrieval-based context.',
    ],
  },
  {
    id: 'updating',
    title: 'Individual updates versus Update all',
    items: [
      'Each source view has its own update button. Use it to refresh one source without touching the others.',
      'Update all runs Drug Catalog, LiverTox, Terminology and DILIrank in that order, because later sources link to identities created by earlier ones.',
      'In Update all you can switch off terminology mapping for a faster run. RAG is always updated separately from its own view.',
      'Only one structured update can run at a time. Start buttons stay disabled while a job is active.',
    ],
  },
  {
    id: 'terminology',
    title: 'Terminology mapping',
    items: [
      'Normalization turns names such as brand names, abbreviations and spelling variants into a single drug identity, so LiverTox and DILIrank evidence can be matched.',
      'Contributing sources: RxNorm, DrugCentral (CC BY-SA 4.0), ChEMBL (CC BY-SA 3.0), Swissmedic (open data, attribution required) and project-curated aliases.',
      'Update terminology after a Drug Catalog refresh, when drugs are not recognised, or when you enable an additional source.',
      'Skip unchanged releases is the usual mode; Full re-import rebuilds everything from the sources.',
    ],
  },
  {
    id: 'pipeline',
    title: 'How sources feed the DILI analysis',
    items: [
      'Medication names found in the note are resolved through Terminology, then linked to Drug Catalog entries.',
      'LiverTox and DILIrank provide the hepatotoxicity evidence shown in the report. RAG adds document context.',
      'Out-of-date or missing sources reduce matches but never change stored sessions.',
    ],
  },
  {
    id: 'status',
    title: 'Reading update status',
    items: [
      'Running: the job is active and the bar shows its share of work. Queued sources wait for earlier ones.',
      'Completed: the source finished and its tables were refreshed.',
      'Skipped or excluded: the source was not part of the run, or its release was unchanged.',
      'Failed: the update stopped. In Update all, the sources after the failed one are skipped. Fix the cause and run it again.',
      'Cancelled: stop was requested. The running source finishes safely and the rest do not start.',
    ],
  },
];

@Component({
  selector: 'app-knowledge-help',
  standalone: true,
  imports: [ModalShellComponent],
  template: `
    <app-modal-shell
      [isOpen]="isOpen()"
      title="Knowledge help"
      subtitle="How the data sources work and how to keep them up to date."
      titleId="knowledge-help-title"
      ariaLabelledBy="knowledge-help-title"
      dialogClassName="modal-container inspection-modal knowledge-help-modal"
      closeLabel="Close Knowledge help"
      (close)="close.emit()"
    >
      <div class="knowledge-help">
        @for (section of sections; track section.id; let first = $first) {
          <details class="knowledge-help-section" [open]="first">
            <summary>{{ section.title }}</summary>
            <ul>
              @for (item of section.items; track item) {
                <li>{{ item }}</li>
              }
            </ul>
          </details>
        }
      </div>
    </app-modal-shell>
  `,
  styles: `
    :host {
      display: contents;
    }

    .knowledge-help {
      display: flex;
      flex-direction: column;
    }

    .knowledge-help-section {
      padding: var(--space-sm) 0;
    }

    .knowledge-help-section + .knowledge-help-section {
      border-top: 1px solid var(--color-border-subtle);
    }

    .knowledge-help-section summary {
      cursor: pointer;
      color: var(--color-text-primary);
      font-size: var(--font-md);
      font-weight: 700;
      border-radius: var(--radius-sm);
    }

    .knowledge-help-section summary:focus-visible {
      outline: 2px solid var(--color-brand-light);
      outline-offset: 2px;
    }

    .knowledge-help-section ul {
      margin: var(--space-sm) 0 0;
      padding-left: var(--space-lg);
      color: var(--color-text-secondary);
      font-size: var(--font-sm);
      line-height: 1.5;
    }

    .knowledge-help-section li + li {
      margin-top: var(--space-xs);
    }
  `,
})
export class KnowledgeHelpComponent {
  readonly isOpen = input(false);
  readonly close = output<void>();
  readonly sections = KNOWLEDGE_HELP_SECTIONS;
}
