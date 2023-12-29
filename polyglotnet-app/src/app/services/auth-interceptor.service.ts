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
                    `${user.idToken}`,
                );
                this.cookieService.set(cookies_constants.sessionId, user.id);
            }
        });
    }

    intercept(
        req: HttpRequest<any>,
        next: HttpHandler,
    ): Observable<HttpEvent<any>> {
        console.log('intercepting')
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
        if (environment.production) {
            req = req.clone({
                setHeaders: {
                    'Authorization': `Bearer ${token}`,
                    'X-Session-Id': sessionId,
                    'Content-Type': 'application/json',
                },
            });
        } else {
            req = req.clone({
                setHeaders: {
                    'Authorization': `Bearer ${token}`,
                    'X-Session-Id': sessionId,
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
            });
        }
        return next.handle(req);
    }
}
