import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { AccessKeyProvider, AccessKeyRecord } from '../../../core/models/types';
import {
  activateAccessKey,
  createAccessKey,
  deleteAccessKey,
  fetchAccessKeys,
} from '../../../core/services/api';
import { ModalShellComponent } from '../../../components/modal-shell/modal-shell.component';
import { StatusMessageComponent } from '../../../components/status-message/status-message.component';

const MASKED_KEY_LABEL = '********************';

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
    this.isLoading = true;
    this.errorMessage = '';
    try {
      this.keys = await fetchAccessKeys(this.provider);
    } catch (error) {
      this.errorMessage = error instanceof Error ? error.message : 'Unable to load access keys.';
    } finally {
      this.isLoading = false;
    }
  }

  async addKey(): Promise<void> {
    const candidate = this.newKeyValue.trim();
    if (!candidate) {
      this.errorMessage = 'Please paste a key before adding.';
      return;
    }
    this.isSaving = true;
    this.errorMessage = '';
    try {
      await createAccessKey(this.provider, candidate);
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
}

