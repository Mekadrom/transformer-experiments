import { SocialAuthService } from '@abacritt/angularx-social-login';
import { Injectable } from '@angular/core';
import {
    HttpEvent,
    HttpInterceptor,
    HttpHandler,
    HttpRequest,
} from '@angular/common/http';
import { CookieService } from 'ngx-cookie-service';
import { Observable } from 'rxjs';

import { environment } from '../environments/environment';
import { cookies_constants } from '../constants/cookies-constants';
import { utils } from '../utils/utils';

@Injectable()
export class AuthInterceptorService implements HttpInterceptor {
    constructor(
        private cookieService: CookieService,
        private socialAuthService: SocialAuthService,
    ) {
        socialAuthService.authState.subscribe((user) => {
            if (user) {
                this.cookieService.set(
                    cookies_constants.authorization,
                    `Bearer ${user.idToken}`,
                );
                this.cookieService.set(cookies_constants.sessionId, user.id);
            }
        });
    }

    intercept(
        req: HttpRequest<any>,
        next: HttpHandler,
    ): Observable<HttpEvent<any>> {
        if (!req.url.startsWith(environment.apiBaseUrl)) {
            return next.handle(req);
        }

        let token: string = this.cookieService.get(
            cookies_constants.authorization,
        );
        if (utils.isExpired(token)) {
            console.log('token expired, refreshing');
            this.socialAuthService.refreshAccessToken(
                environment.googleClientId,
            );
            token = this.cookieService.get(cookies_constants.authorization);
        }
        let sessionId = this.cookieService.get(cookies_constants.sessionId);
        if (!sessionId) {
            console.log('no session id, getting from token');
            sessionId = utils.getSub(token);
        }
        req = req.clone({
            setHeaders: {
                Authorization: `Bearer ${token}`,
                'X-Session-Id': sessionId,
            },
        });
        return next.handle(req);
    }
}
