import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OauthViewComponent } from './oauth-view.component';

describe('OauthViewComponent', () => {
    let component: OauthViewComponent;
    let fixture: ComponentFixture<OauthViewComponent>;

    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [OauthViewComponent],
        }).compileComponents();

        fixture = TestBed.createComponent(OauthViewComponent);
        component = fixture.componentInstance;
        fixture.detectChanges();
    });

    it('should create', () => {
        expect(component).toBeTruthy();
    });
});
