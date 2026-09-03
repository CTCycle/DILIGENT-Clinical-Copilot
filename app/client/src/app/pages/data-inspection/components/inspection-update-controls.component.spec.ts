import { describe, expect, it } from 'vitest';

import { InspectionUpdateControlsComponent } from './inspection-update-controls.component';

describe('InspectionUpdateControlsComponent', () => {
  it('updates a numeric field while preserving the other JSON overrides', () => {
    const component = new InspectionUpdateControlsComponent();
    component.configText = JSON.stringify({
      rxnav_request_timeout: 30,
      rxnav_max_concurrency: 4,
    });
    let emitted = '';
    component.configTextChange.subscribe((value) => {
      emitted = value;
    });

    component.updateNumber('rxnav_max_concurrency', 8, 1, 64, true);

    expect(JSON.parse(emitted)).toEqual({
      rxnav_request_timeout: 30,
      rxnav_max_concurrency: 8,
    });
  });

  it('uses loaded defaults to recover from invalid advanced JSON', () => {
    const component = new InspectionUpdateControlsComponent();
    component.config = {
      livertox_monograph_max_workers: 4,
      livertox_archive: 'livertox_NBK547852.tar.gz',
      redownload: false,
    };
    component.configText = '{ invalid';
    let emitted = '';
    component.configTextChange.subscribe((value) => {
      emitted = value;
    });

    component.updateBoolean('redownload', true);

    expect(JSON.parse(emitted)).toEqual({
      livertox_monograph_max_workers: 4,
      livertox_archive: 'livertox_NBK547852.tar.gz',
      redownload: true,
    });
  });

  it('reports malformed or non-object JSON', () => {
    const component = new InspectionUpdateControlsComponent();

    component.configText = '[]';
    expect(component.jsonError()).toBe('Overrides must be a JSON object.');

    component.configText = '{ invalid';
    expect(component.jsonError()).toBe('Invalid JSON overrides.');
  });
});
