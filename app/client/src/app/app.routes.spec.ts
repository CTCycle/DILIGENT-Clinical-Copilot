// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { routes } from './app.routes';

describe('application routes', () => {
  it('keeps model settings under Settings without the removed UI alias', () => {
    const settingsRoute = routes.find((route) => route.path === 'settings');

    expect(settingsRoute?.children?.some((route) => route.path === 'models')).toBe(true);
    expect(routes.some((route) => route.path === 'model-config')).toBe(false);
  });
});
