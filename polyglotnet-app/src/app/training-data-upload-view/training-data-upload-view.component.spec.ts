import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingDataUploadViewComponent } from './training-data-upload-view.component';

describe('TrainingDataUploadViewComponent', () => {
  let component: TrainingDataUploadViewComponent;
  let fixture: ComponentFixture<TrainingDataUploadViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingDataUploadViewComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TrainingDataUploadViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
