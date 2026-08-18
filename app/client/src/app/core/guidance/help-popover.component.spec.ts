import { ComponentFixture, TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { HelpPopoverComponent } from './help-popover.component';

describe('HelpPopoverComponent', () => {
  let fixture: ComponentFixture<HelpPopoverComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [HelpPopoverComponent] }).compileComponents();
    fixture = TestBed.createComponent(HelpPopoverComponent);
    fixture.componentRef.setInput('title', 'Test help');
    fixture.componentRef.setInput('body', 'A short explanation.');
    fixture.detectChanges();
  });

  afterEach(() => {
    fixture.destroy();
  });

  it('opens from the trigger, exposes a dialog, and closes with Escape while restoring focus', async () => {
    const trigger = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    trigger.focus();
    trigger.click();
    fixture.detectChanges();
    await Promise.resolve();
    fixture.detectChanges();

    expect(fixture.componentInstance.open()).toBe(true);
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).not.toBeNull();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    fixture.detectChanges();
    await Promise.resolve();

    expect(fixture.componentInstance.open()).toBe(false);
    expect(document.activeElement).toBe(trigger);
  });

  it('closes when a pointer lands outside the popover host', async () => {
    const trigger = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    trigger.click();
    fixture.detectChanges();
    await Promise.resolve();

    document.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    fixture.detectChanges();

    expect(fixture.componentInstance.open()).toBe(false);
  });
});
