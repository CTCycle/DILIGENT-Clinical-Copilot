import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InspectionCatalogStatusComponent } from './inspection-catalog-status.component';

describe('InspectionCatalogStatusComponent', () => {
  let fixture: ComponentFixture<InspectionCatalogStatusComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InspectionCatalogStatusComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(InspectionCatalogStatusComponent);
    fixture.componentRef.setInput('loading', false);
    fixture.componentRef.setInput('loadingMessage', 'Loading catalog...');
    fixture.componentRef.setInput('loadingMore', false);
    fixture.componentRef.setInput('loadingMoreMessage', 'Loading more rows...');
    fixture.componentRef.setInput('hasMore', false);
  });

  it('renders catalog progress messages without changing their CSS contract', () => {
    fixture.componentRef.setInput('loadingMore', true);
    fixture.componentRef.setInput('hasMore', true);
    fixture.detectChanges();

    const messages = Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLElement>('.inspection-loading-note'),
      (element) => element.textContent?.trim(),
    );
    expect(messages).toEqual(['Loading more rows...', 'Scroll to load more...']);
  });

  it('renders the supplied error with the existing error class', () => {
    fixture.componentRef.setInput('error', 'Catalog failed.');
    fixture.detectChanges();

    const error = (fixture.nativeElement as HTMLElement).querySelector<HTMLElement>('.inspection-error-text');
    expect(error?.textContent?.trim()).toBe('Catalog failed.');
  });
});
