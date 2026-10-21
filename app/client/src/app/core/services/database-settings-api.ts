import { API_BASE_URL } from '../constants';
import {
  DatabaseConfigurationRequest,
  DatabaseConfigurationState,
  DatabaseConnectionTest,
  DatabaseSwitchResult,
} from '../models/database-settings';
import { requestJson } from './http-api';

const DATABASE_API = `${API_BASE_URL}/database`;

export function fetchDatabaseConfiguration(): Promise<DatabaseConfigurationState> {
  return requestJson<DatabaseConfigurationState>(`${DATABASE_API}/configuration`, {
    method: 'GET',
    cache: 'no-store',
  });
}

export function testDatabaseConfiguration(
  payload: DatabaseConfigurationRequest,
): Promise<DatabaseConnectionTest> {
  return requestJson<DatabaseConnectionTest>(`${DATABASE_API}/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 45);
}

export function applyDatabaseConfiguration(
  payload: DatabaseConfigurationRequest,
): Promise<DatabaseSwitchResult> {
  return requestJson<DatabaseSwitchResult>(`${DATABASE_API}/configuration`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 180);
}
