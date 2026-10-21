// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  DatabaseBackend,
  DatabaseConfiguration,
  DatabaseConfigurationRequest,
  DatabaseConfigurationState,
  DatabaseConnectionTest,
} from '../../../core/models/database-settings';
import {
  applyDatabaseConfiguration,
  fetchDatabaseConfiguration,
  testDatabaseConfiguration,
} from '../../../core/services/database-settings-api';

@Component({
  selector: 'app-database-settings',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './database-settings.component.html',
  styleUrl: './database-settings.component.scss',
})
export class DatabaseSettingsComponent implements OnInit {
  readonly configuration = signal<DatabaseConfiguration | null>(null);
  readonly state = signal<DatabaseConfigurationState | null>(null);
  readonly loading = signal(true);
  readonly testing = signal(false);
  readonly saving = signal(false);
  readonly error = signal<string | null>(null);
  readonly testResult = signal<DatabaseConnectionTest | null>(null);
  readonly password = signal('');
  readonly backendOptions: { value: DatabaseBackend; label: string }[] = [
    { value: 'sqlite', label: 'SQLite' },
    { value: 'postgresql', label: 'PostgreSQL' },
  ];

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const response = await fetchDatabaseConfiguration();
      this.state.set(response);
      this.configuration.set({ ...response.configuration });
      this.password.set('');
    } catch (error: unknown) {
      this.error.set(this.errorMessage(error));
    } finally {
      this.loading.set(false);
    }
  }

  updateBackend(backend: DatabaseBackend): void {
    const current = this.configuration();
    if (!current) return;
    this.configuration.set({ ...current, backend });
    this.testResult.set(null);
  }

  patchConfiguration(patch: Partial<DatabaseConfiguration>): void {
    const current = this.configuration();
    if (current) this.configuration.set({ ...current, ...patch });
    this.testResult.set(null);
  }

  async testConnection(): Promise<void> {
    const payload = this.buildRequest();
    if (!payload) return;
    this.testing.set(true);
    this.error.set(null);
    this.testResult.set(null);
    try {
      this.testResult.set(await testDatabaseConfiguration(payload));
    } catch (error: unknown) {
      this.error.set(this.errorMessage(error));
    } finally {
      this.testing.set(false);
    }
  }

  async save(): Promise<void> {
    const payload = this.buildRequest();
    if (!payload) return;
    this.saving.set(true);
    this.error.set(null);
    try {
      await applyDatabaseConfiguration(payload);
      // The API reloads database-backed settings. A page reload refreshes
      // client state held by every other settings and workspace service.
      window.location.reload();
    } catch (error: unknown) {
      this.error.set(this.errorMessage(error));
      this.saving.set(false);
    }
  }

  private buildRequest(): DatabaseConfigurationRequest | null {
    const current = this.configuration();
    if (!current) return null;
    if (current.backend === 'sqlite' && !current.sqlite_path?.trim()) {
      this.error.set('Choose a SQLite database file path.');
      return null;
    }
    if (current.backend === 'postgresql') {
      if (!current.host?.trim() || !current.database_name?.trim() || !current.username?.trim()) {
        this.error.set('Enter the PostgreSQL host, database name, and username.');
        return null;
      }
    }
    return {
      backend: current.backend,
      sqlite_path: current.backend === 'sqlite' ? current.sqlite_path : null,
      engine: current.engine || 'postgresql+psycopg',
      host: current.backend === 'postgresql' ? current.host : null,
      port: current.port || 5432,
      database_name: current.backend === 'postgresql' ? current.database_name : null,
      username: current.backend === 'postgresql' ? current.username : null,
      password: this.password() || null,
      keep_password: true,
      ssl: current.backend === 'postgresql' && current.ssl,
      ssl_ca: current.backend === 'postgresql' ? current.ssl_ca : null,
      connect_timeout: current.connect_timeout,
      write_batch_size: current.write_batch_size,
      read_page_size: current.read_page_size,
    };
  }

  activeBackendLabel(): string {
    const backend = this.state()?.active_backend;
    return backend === 'postgresql' ? 'PostgreSQL' : 'SQLite';
  }

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : 'Database configuration could not be loaded.';
  }
}
