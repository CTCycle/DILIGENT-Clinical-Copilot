// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ModalShellComponent } from '../../../components/modal-shell/modal-shell.component';
import { HelpPopoverComponent } from '../../../core/guidance/help-popover.component';
import { ModelConfigStateResponse } from '../../../core/models/types';
import {
  DraftRagSettings,
  RagBooleanKey,
  RagFloatKey,
  RagNumberKey,
  RagSettingsSectionKey,
  RagTextKey,
} from '../model-config.types';

const RERANKER_PROFILE_OPTIONS = [
  { value: 'lightweight-balanced-v1', label: 'Balanced' },
  { value: 'lightweight-lexical-v1', label: 'Lexical' },
  { value: 'lightweight-phrase-v1', label: 'Phrase' },
] as const;

const RAG_INDEX_METRIC_OPTIONS = [
  { value: 'cosine', label: 'Cosine' },
  { value: 'euclidean', label: 'Euclidean' },
  { value: 'dot', label: 'Dot product' },
] as const;

const RAG_INDEX_TYPE_OPTIONS = [
  { value: 'IVF_FLAT', label: 'IVF flat' },
  { value: 'HNSW', label: 'HNSW' },
  { value: 'FLAT', label: 'Flat' },
] as const;

@Component({
  selector: 'app-rag-resources-panel',
  standalone: true,
  imports: [FormsModule, ModalShellComponent, HelpPopoverComponent],
  templateUrl: './rag-resources-panel.component.html',
  styleUrl: './rag-resources-panel.component.scss',
})
export class RagResourcesPanelComponent {
  readonly draft = input.required<DraftRagSettings>();
  readonly modalOpen = input(false);
  readonly activeSection = input<RagSettingsSectionKey>('general');
  readonly disabled = input(false);
  readonly status = input('');
  readonly statusIsError = input(false);
  readonly modelLabel = input('');
  readonly embeddingRuntime = input<ModelConfigStateResponse['embedding_runtime'] | null>(null);
  readonly embeddingIndex = input<ModelConfigStateResponse['embedding_index'] | null>(null);

  readonly openRequested = output<void>();
  readonly closeRequested = output<void>();
  readonly sectionChange = output<RagSettingsSectionKey>();
  readonly numberChange = output<{ key: RagNumberKey; value: string }>();
  readonly floatChange = output<{ key: RagFloatKey; value: string }>();
  readonly textChange = output<{ key: RagTextKey; value: string }>();
  readonly booleanChange = output<{ key: RagBooleanKey; value: boolean }>();

  readonly rerankerProfileOptions = RERANKER_PROFILE_OPTIONS;
  readonly ragIndexMetricOptions = RAG_INDEX_METRIC_OPTIONS;
  readonly ragIndexTypeOptions = RAG_INDEX_TYPE_OPTIONS;
}
