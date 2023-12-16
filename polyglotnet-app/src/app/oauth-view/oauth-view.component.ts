import { SocialAuthService } from '@abacritt/angularx-social-login';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import {
    GoogleSigninButtonModule,
    SocialUser,
} from '@abacritt/angularx-social-login';
import { CookieService } from 'ngx-cookie-service';

import { environment } from '../environments/environment';
import { cookies_constants } from '../constants/cookies-constants';

@Component({
    selector: 'app-oauth-view',
    standalone: true,
    imports: [CommonModule, GoogleSigninButtonModule],
    templateUrl: './oauth-view.component.html',
    styleUrl: './oauth-view.component.scss',
})
export class OauthViewComponent implements OnInit {
    constructor(
        private socialAuthService: SocialAuthService,
        private cookieService: CookieService,
        private router: Router,
    ) {}

    ngOnInit(): void {
        this.socialAuthService.authState.subscribe((user) => {
            if (user) {
                this.cookieService.set(
                    cookies_constants.authorization,
                    `Bearer ${user.idToken}`,
                );
                this.cookieService.set(cookies_constants.sessionId, user.id);
                this.router.navigate(['/']);
            }
        });
    }

    refreshToken(): void {
        this.socialAuthService.refreshAuthToken(environment.googleClientId);
    }
}
