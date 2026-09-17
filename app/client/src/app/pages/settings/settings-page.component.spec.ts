import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { beforeEach, describe, expect, it } from 'vitest';

import { SettingsPageComponent } from './settings-page.component';

describe('SettingsPageComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SettingsPageComponent],
      providers: [provideRouter([])],
    }).compileComponents();
  });

  it('renders the persistent settings navigation without duplicate model configuration', () => {
    const fixture = TestBed.createComponent(SettingsPageComponent);
    fixture.detectChanges();

    const links = Array.from(
      fixture.nativeElement.querySelectorAll('.settings-navigation-link') as NodeListOf<HTMLAnchorElement>,
    );
    expect(links).toHaveLength(5);
    expect(links.map((link) => link.textContent?.trim())).toEqual([
      expect.stringContaining('General'),
      expect.stringContaining('Models'),
      expect.stringContaining('Data Processing'),
      expect.stringContaining('Integrations'),
      expect.stringContaining('Advanced'),
    ]);
    expect(fixture.nativeElement.textContent).toContain('.env');
  });
});
