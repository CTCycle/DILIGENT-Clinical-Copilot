import { CommonModule } from '@angular/common';
import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  computed,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AccessKeyProvider, AccessKeyRecord } from '../../../core/models/types';
import {
  activateAccessKey,
  createAccessKey,
  deleteAccessKey,
  fetchAccessKeys,
} from '../../../core/services/model-config-api';
import { ModalShellComponent } from '../../../components/modal-shell/modal-shell.component';
import { StatusMessageComponent } from '../../../components/status-message/status-message.component';

const MASKED_KEY_LABEL = '********************';
const MIN_ACCESS_KEY_LENGTH = 16;

function obfuscateFingerprint(value: string): string {
  const fingerprint = (value || '').trim();
  if (fingerprint.length <= 10) {
    return `fp: ${fingerprint || 'unknown'}`;
  }
  return `fp: ${fingerprint.slice(0, 6)}...${fingerprint.slice(-4)}`;
}

function formatTimestamp(value: string | null): string {
  if (!value) {
    return 'Not used';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return 'Not used';
  }
  return parsed.toLocaleString();
}

@Component({
  selector: 'app-access-key-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalShellComponent, StatusMessageComponent],
  templateUrl: './access-key-modal.component.html',
  styleUrl: './access-key-modal.component.scss',
})
export class AccessKeyModalComponent implements OnChanges {
  @Input() isOpen = false;
  @Input() provider: AccessKeyProvider = 'openai';
  @Input() providerLabel = 'OpenAI';
  @Output() closed = new EventEmitter<void>();
  @Output() keysChanged = new EventEmitter<void>();

  readonly keys = signal<AccessKeyRecord[]>([]);
  readonly isLoading = signal(false);
  readonly isSaving = signal(false);
  readonly newKeyValue = signal('');
  readonly errorMessage = signal('');
  readonly visibleRows = signal<Record<number, boolean>>({});
  private loadSequence = 0;

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['isOpen'] || changes['provider']) && this.isOpen) {
      void this.loadKeys();
    }
  }

  readonly sortedKeys = computed(() => [...this.keys()].sort(
      (left, right) => Number(right.is_active) - Number(left.is_active) || right.id - left.id,
    ));

  readonly hasKeys = computed(() => this.keys().length > 0);

  async loadKeys(): Promise<void> {
    const loadId = ++this.loadSequence;
    this.isLoading.set(true);
    this.errorMessage.set('');
    try {
      const keys = await fetchAccessKeys(this.provider);
      if (loadId === this.loadSequence) {
        this.keys.set(keys);
      }
    } catch (error) {
      if (loadId === this.loadSequence) {
        this.errorMessage.set(error instanceof Error ? error.message : 'Unable to load access keys.');
      }
    } finally {
      if (loadId === this.loadSequence) {
        this.isLoading.set(false);
      }
    }
  }

  async addKey(): Promise<void> {
    const candidate = this.newKeyValue().trim();
    if (!candidate) {
      this.errorMessage.set('Please paste a key before adding.');
      return;
    }
    if (candidate.length < MIN_ACCESS_KEY_LENGTH) {
      this.errorMessage.set(`Access keys must be at least ${MIN_ACCESS_KEY_LENGTH} characters.`);
      return;
    }
    this.isSaving.set(true);
    this.errorMessage.set('');
    try {
      await createAccessKey(this.provider, candidate);
      this.newKeyValue.set('');
      await this.loadKeys();
      this.keysChanged.emit();
    } catch (error) {
      this.errorMessage.set(error instanceof Error ? error.message : 'Unable to add access key.');
    } finally {
      this.isSaving.set(false);
    }
  }

  async activateKey(keyId: number): Promise<void> {
    this.isSaving.set(true);
    this.errorMessage.set('');
    try {
      const activated = await activateAccessKey(keyId, this.provider);
      this.keys.update((items) => items.map((item) => ({
        ...item,
        is_active: item.id === activated.id,
        updated_at: item.id === activated.id ? activated.updated_at : item.updated_at,
        last_used_at: item.id === activated.id ? activated.last_used_at : item.last_used_at,
      })));
      this.keysChanged.emit();
    } catch (error) {
      this.errorMessage.set(error instanceof Error ? error.message : 'Unable to activate access key.');
    } finally {
      this.isSaving.set(false);
    }
  }

  async deleteKey(keyId: number): Promise<void> {
    this.isSaving.set(true);
    this.errorMessage.set('');
    try {
      await deleteAccessKey(keyId, this.provider);
      this.keys.update((items) => items.filter((item) => item.id !== keyId));
      this.visibleRows.update((rows) => {
        const next = { ...rows };
        delete next[keyId];
        return next;
      });
      this.keysChanged.emit();
    } catch (error) {
      this.errorMessage.set(error instanceof Error ? error.message : 'Unable to delete access key.');
    } finally {
      this.isSaving.set(false);
    }
  }

  toggleVisibility(keyId: number): void {
    this.visibleRows.update((rows) => ({ ...rows, [keyId]: !rows[keyId] }));
  }

  fingerprintLabel(item: AccessKeyRecord): string {
    return this.visibleRows()[item.id] ? obfuscateFingerprint(item.fingerprint) : MASKED_KEY_LABEL;
  }

  lastUsedLabel(item: AccessKeyRecord): string {
    return formatTimestamp(item.last_used_at);
  }

  activateActionLabel(item: AccessKeyRecord): string {
    return item.is_active ? 'Key is active' : 'Activate key';
  }

  visibilityActionLabel(item: AccessKeyRecord): string {
    return this.visibleRows()[item.id] ? 'Hide fingerprint' : 'Show fingerprint';
  }
}


