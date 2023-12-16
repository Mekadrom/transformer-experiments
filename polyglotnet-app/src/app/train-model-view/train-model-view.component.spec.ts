import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainModelViewComponent } from './train-model-view.component';

describe('TrainModelViewComponent', () => {
    let component: TrainModelViewComponent;
    let fixture: ComponentFixture<TrainModelViewComponent>;

    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [TrainModelViewComponent],
        }).compileComponents();

        fixture = TestBed.createComponent(TrainModelViewComponent);
        component = fixture.componentInstance;
        fixture.detectChanges();
    });

    it('should create', () => {
        expect(component).toBeTruthy();
    });
});
