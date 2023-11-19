import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LexiconViewComponent } from './lexicon-view.component';

describe('LexiconViewComponent', () => {
  let component: LexiconViewComponent;
  let fixture: ComponentFixture<LexiconViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LexiconViewComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(LexiconViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
