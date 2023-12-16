import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingDataAugmentationComponent } from './training-data-augmentation.component';

describe('TrainingDataAugmentationComponent', () => {
  let component: TrainingDataAugmentationComponent;
  let fixture: ComponentFixture<TrainingDataAugmentationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingDataAugmentationComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TrainingDataAugmentationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
