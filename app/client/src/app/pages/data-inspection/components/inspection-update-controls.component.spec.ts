import { describe, expect, it } from 'vitest';

import { InspectionUpdateControlsComponent } from './inspection-update-controls.component';

describe('InspectionUpdateControlsComponent', () => {
  it('emits a bounded integer concurrency value', () => {
    const component = new InspectionUpdateControlsComponent();
    let emitted: { key: string; value: unknown } | null = null;
    component.configChange.subscribe((value) => {
      emitted = value;
    });

    component.updateNumber('rxnav_max_concurrency', 80, 1, 64, true);

    expect(emitted).toEqual({ key: 'rxnav_max_concurrency', value: 64 });
  });

  it('emits the selected archive handling mode', () => {
    const component = new InspectionUpdateControlsComponent();
    let emitted: { key: string; value: unknown } | null = null;
    component.configChange.subscribe((value) => {
      emitted = value;
    });

    component.updateBoolean('redownload', true);

    expect(emitted).toEqual({ key: 'redownload', value: true });
  });

  it('reads displayed values from the loaded configuration', () => {
    const component = new InspectionUpdateControlsComponent();
    component.config = {
      livertox_monograph_max_workers: 6,
      livertox_archive: 'livertox_NBK547852.tar.gz',
      redownload: false,
    };

    expect(component.numberValue('livertox_monograph_max_workers', 4)).toBe(6);
    expect(component.stringValue('livertox_archive')).toBe('livertox_NBK547852.tar.gz');
    expect(component.booleanValue('redownload', true)).toBe(false);
  });
});
