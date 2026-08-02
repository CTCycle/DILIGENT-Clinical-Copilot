import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AccessKeyRecord } from '../../../core/models/types';
import { AccessKeyModalComponent } from './access-key-modal.component';

describe('AccessKeyModalComponent', () => {
  let fixture: ComponentFixture<AccessKeyModalComponent>;
  let component: AccessKeyModalComponent;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  const key = (id: number, isActive: boolean): AccessKeyRecord => ({
    id,
    provider: 'openai',
    fingerprint: `fingerprint-${id}`,
    is_active: isActive,
    created_at: '2026-07-01T10:00:00Z',
    updated_at: '2026-07-01T10:00:00Z',
    last_used_at: null,
  });

  const jsonResponse = (body: unknown) => ({
    ok: true,
    status: 200,
    headers: new Headers({ 'content-type': 'application/json' }),
    text: async () => JSON.stringify(body),
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AccessKeyModalComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(AccessKeyModalComponent);
    component = fixture.componentInstance;
    fetchSpy = vi.spyOn(globalThis, 'fetch');
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it('loads keys into signal state and keeps active keys first', async () => {
    fetchSpy.mockResolvedValue(jsonResponse([key(1, false), key(2, true)]) as Response);

    await component.loadKeys();

    expect(component.isLoading()).toBe(false);
    expect(component.hasKeys()).toBe(true);
    expect(component.sortedKeys().map((item) => item.id)).toEqual([2, 1]);
  });

  it('validates short key input without starting a save', async () => {
    component.newKeyValue.set('too-short');

    await component.addKey();

    expect(component.errorMessage()).toBe('Access keys must be at least 16 characters.');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('activates, deletes, and signal-updates key rows and visibility', async () => {
    const inactive = key(1, false);
    const active = key(2, true);
    component.keys.set([inactive, active]);
    const emitSpy = vi.spyOn(component.keysChanged, 'emit');
    fetchSpy
      .mockResolvedValueOnce(jsonResponse({ ...inactive, is_active: true, updated_at: '2026-07-02T10:00:00Z' }) as Response)
      .mockResolvedValueOnce(jsonResponse({ status: 'ok', deleted: true }) as Response);

    component.toggleVisibility(1);
    expect(component.fingerprintLabel(inactive)).toContain('fp:');

    await component.activateKey(1);
    expect(component.keys().find((item) => item.id === 1)?.is_active).toBe(true);
    expect(component.keys().find((item) => item.id === 2)?.is_active).toBe(false);

    await component.deleteKey(1);
    expect(component.keys().map((item) => item.id)).toEqual([2]);
    expect(component.isSaving()).toBe(false);
    expect(emitSpy).toHaveBeenCalledTimes(2);
  });
});
