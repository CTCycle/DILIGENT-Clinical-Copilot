import { CommonModule } from '@angular/common';
import {
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
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

  keys: AccessKeyRecord[] = [];
  isLoading = false;
  isSaving = false;
  newKeyValue = '';
  errorMessage = '';
  visibleRows: Record<number, boolean> = {};
  private loadSequence = 0;

  constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  ngOnChanges(changes: SimpleChanges): void {
    if ((changes['isOpen'] || changes['provider']) && this.isOpen) {
      void this.loadKeys();
    }
  }

  get sortedKeys(): AccessKeyRecord[] {
    return [...this.keys].sort(
      (left, right) => Number(right.is_active) - Number(left.is_active) || right.id - left.id,
    );
  }

  get hasKeys(): boolean {
    return this.keys.length > 0;
  }

  get maskedKeyLabel(): string {
    return MASKED_KEY_LABEL;
  }

  async loadKeys(): Promise<void> {
    const loadId = ++this.loadSequence;
    this.isLoading = true;
    this.errorMessage = '';
    try {
      const keys = await fetchAccessKeys(this.provider);
      if (loadId === this.loadSequence) {
        this.keys = keys;
        this.changeDetectorRef.markForCheck();
      }
    } catch (error) {
      if (loadId === this.loadSequence) {
        this.errorMessage = error instanceof Error ? error.message : 'Unable to load access keys.';
        this.changeDetectorRef.markForCheck();
      }
    } finally {
      if (loadId === this.loadSequence) {
        this.isLoading = false;
        this.changeDetectorRef.detectChanges();
      }
    }
  }

  async addKey(): Promise<void> {
    const candidate = this.newKeyValue.trim();
    if (!candidate) {
      this.errorMessage = 'Please paste a key before adding.';
      return;
    }
    if (candidate.length < MIN_ACCESS_KEY_LENGTH) {
      this.errorMessage = `Access keys must be at least ${MIN_ACCESS_KEY_LENGTH} characters.`;
      return;
    }
    this.isSaving = true;
    this.errorMessage = '';
    try {
      const created = await createAccessKey(this.provider, candidate);
      if (!created.is_active) {
        await activateAccessKey(created.id, this.provider);
      }
      this.newKeyValue = '';
      await this.loadKeys();
    } catch (error) {
      this.errorMessage = error instanceof Error ? error.message : 'Unable to add access key.';
    } finally {
      this.isSaving = false;
    }
  }

  async activateKey(keyId: number): Promise<void> {
    this.isSaving = true;
    this.errorMessage = '';
    try {
      const activated = await activateAccessKey(keyId, this.provider);
      this.keys = this.keys.map((item) => ({
        ...item,
        is_active: item.id === activated.id,
        updated_at: item.id === activated.id ? activated.updated_at : item.updated_at,
        last_used_at: item.id === activated.id ? activated.last_used_at : item.last_used_at,
      }));
      this.changeDetectorRef.detectChanges();
    } catch (error) {
      this.errorMessage = error instanceof Error ? error.message : 'Unable to activate access key.';
    } finally {
      this.isSaving = false;
    }
  }

  async deleteKey(keyId: number): Promise<void> {
    this.isSaving = true;
    this.errorMessage = '';
    try {
      await deleteAccessKey(keyId, this.provider);
      this.keys = this.keys.filter((item) => item.id !== keyId);
      const next = { ...this.visibleRows };
      delete next[keyId];
      this.visibleRows = next;
      this.changeDetectorRef.detectChanges();
    } catch (error) {
      this.errorMessage = error instanceof Error ? error.message : 'Unable to delete access key.';
    } finally {
      this.isSaving = false;
    }
  }

  toggleVisibility(keyId: number): void {
    this.visibleRows = { ...this.visibleRows, [keyId]: !this.visibleRows[keyId] };
  }

  fingerprintLabel(item: AccessKeyRecord): string {
    return this.visibleRows[item.id] ? obfuscateFingerprint(item.fingerprint) : MASKED_KEY_LABEL;
  }

  lastUsedLabel(item: AccessKeyRecord): string {
    return formatTimestamp(item.last_used_at);
  }

  activateActionLabel(item: AccessKeyRecord): string {
    return item.is_active ? 'Key is active' : 'Activate key';
  }

  visibilityActionLabel(item: AccessKeyRecord): string {
    return this.visibleRows[item.id] ? 'Hide fingerprint' : 'Show fingerprint';
  }
}


