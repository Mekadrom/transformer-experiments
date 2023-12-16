import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HyperParametersViewComponent } from './hyper-parameters-view.component';

describe('HyperParametersViewComponent', () => {
    let component: HyperParametersViewComponent;
    let fixture: ComponentFixture<HyperParametersViewComponent>;

    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [HyperParametersViewComponent],
        }).compileComponents();

        fixture = TestBed.createComponent(HyperParametersViewComponent);
        component = fixture.componentInstance;
        fixture.detectChanges();
    });

    it('should create', () => {
        expect(component).toBeTruthy();
    });
});
