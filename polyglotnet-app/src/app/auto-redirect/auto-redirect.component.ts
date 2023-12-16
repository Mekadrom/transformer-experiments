import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';

import { cookies_constants } from '../constants/cookies-constants';
import { utils } from '../utils/utils';

@Component({
    selector: 'app-auto-redirect',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './auto-redirect.component.html',
    styleUrl: './auto-redirect.component.scss',
})
export class AutoRedirectComponent {
    constructor(
        private cookieService: CookieService,
        private router: Router,
    ) {}

    ngOnInit(): void {
        // check cookies for token id
        const sessionId = this.cookieService.get(cookies_constants.sessionId);
        const token = this.cookieService.get(cookies_constants.authorization);
        if (!sessionId || !token || utils.isExpired(token)) {
            this.router.navigate(['/login']);
        } else {
            const lastStep = this.cookieService.get(cookies_constants.lastStep);

            if (lastStep) {
                // redirect to last step saved in cookies
                this.router.navigate([`/step${lastStep}`]);
            } else {
                // redirect to step 1
                this.router.navigate(['/step1']);
            }
        }
    }
}
