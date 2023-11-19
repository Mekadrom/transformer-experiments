import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AutoRedirectComponent } from './auto-redirect.component';

describe('AutoRedirectComponent', () => {
  let component: AutoRedirectComponent;
  let fixture: ComponentFixture<AutoRedirectComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutoRedirectComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AutoRedirectComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
