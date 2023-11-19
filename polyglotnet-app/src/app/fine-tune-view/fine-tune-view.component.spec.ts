import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FineTuneViewComponent } from './fine-tune-view.component';

describe('FineTuneViewComponent', () => {
  let component: FineTuneViewComponent;
  let fixture: ComponentFixture<FineTuneViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FineTuneViewComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(FineTuneViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
