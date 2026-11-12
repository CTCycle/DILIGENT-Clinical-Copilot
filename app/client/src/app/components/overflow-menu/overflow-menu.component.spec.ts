// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { describe, expect, it } from 'vitest';

import { OverflowMenuComponent, OverflowMenuItem } from './overflow-menu.component';

@Component({
  standalone: true,
  imports: [OverflowMenuComponent],
  template: `<app-overflow-menu label="More" [items]="items" (selected)="picked.push($event)" />
    <p id="outside">outside</p>`,
})
class HostComponent {
  items: OverflowMenuItem[] = [
    { id: 'a', label: 'Alpha' },
    { id: 'b', label: 'Beta', reason: 'Not now' },
  ];
  picked: string[] = [];
}

describe('OverflowMenuComponent', () => {
  let fixture: ComponentFixture<HostComponent>;
  const el = (): HTMLElement => fixture.nativeElement;
  const trigger = (): HTMLButtonElement => el().querySelector('.trigger')!;
  const items = (): HTMLButtonElement[] => Array.from(el().querySelectorAll('[role="menuitem"]'));

  it('opens, emits the chosen id, disables items with a reason and closes on outside click', () => {
    fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    expect(items()).toHaveLength(0);

    trigger().click();
    fixture.detectChanges();
    expect(trigger().getAttribute('aria-expanded')).toBe('true');
    expect(items().map((i) => i.disabled)).toEqual([false, true]);
    expect(items()[1].title).toBe('Not now');

    items()[0].click();
    fixture.detectChanges();
    expect(fixture.componentInstance.picked).toEqual(['a']);
    expect(items()).toHaveLength(0);

    trigger().click();
    fixture.detectChanges();
    el().querySelector<HTMLElement>('#outside')!.click();
    fixture.detectChanges();
    expect(items()).toHaveLength(0);
  });
});
