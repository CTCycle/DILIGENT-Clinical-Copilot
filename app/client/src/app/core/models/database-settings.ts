export type DatabaseBackend = 'sqlite' | 'postgresql';

export interface DatabaseConfiguration {
  backend: DatabaseBackend;
  sqlite_path: string | null;
  engine: string | null;
  host: string | null;
  port: number | null;
  database_name: string | null;
  username: string | null;
  password_configured: boolean;
  ssl: boolean;
  ssl_ca: string | null;
  connect_timeout: number;
  write_batch_size: number;
  read_page_size: number;
}

export interface DatabaseConfigurationState {
  configuration: DatabaseConfiguration;
  active_backend: DatabaseBackend;
  status: 'active';
  runtime_switching: boolean;
}

export interface DatabaseConfigurationRequest {
  backend: DatabaseBackend;
  sqlite_path: string | null;
  engine: string;
  host: string | null;
  port: number;
  database_name: string | null;
  username: string | null;
  password: string | null;
  keep_password: boolean;
  ssl: boolean;
  ssl_ca: string | null;
  connect_timeout: number;
  write_batch_size: number;
  read_page_size: number;
}

export interface DatabaseConnectionTest {
  reachable: boolean;
  schema_ready: boolean;
  message: string;
}

export interface DatabaseSwitchResult {
  active_backend: DatabaseBackend;
  settings_reloaded: boolean;
  message: string;
}
