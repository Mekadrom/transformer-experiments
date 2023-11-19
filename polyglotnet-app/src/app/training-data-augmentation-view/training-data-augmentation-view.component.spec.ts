import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingDataAugmentationViewComponent } from './training-data-augmentation-view.component';

describe('TrainingDataAugmentationViewComponent', () => {
  let component: TrainingDataAugmentationViewComponent;
  let fixture: ComponentFixture<TrainingDataAugmentationViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingDataAugmentationViewComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TrainingDataAugmentationViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
