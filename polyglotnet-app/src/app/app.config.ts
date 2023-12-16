import {
    GoogleLoginProvider,
    SocialAuthServiceConfig,
} from '@abacritt/angularx-social-login';
import { HTTP_INTERCEPTORS, provideHttpClient } from '@angular/common/http';
import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { environment } from './environments/environment';
import { AuthInterceptorService } from './services/auth-interceptor.service';

export const appConfig: ApplicationConfig = {
    providers: [
        provideRouter(routes),
        provideHttpClient(),
        {
            provide: 'SocialAuthServiceConfig',
            useValue: {
                autoLogin: true,
                providers: [
                    {
                        id: GoogleLoginProvider.PROVIDER_ID,
                        provider: new GoogleLoginProvider(
                            environment.googleClientId,
                        ),
                    },
                ],
                onError: (err) => {
                    console.error(err);
                },
            } as SocialAuthServiceConfig,
        },
        {
            provide: HTTP_INTERCEPTORS,
            useClass: AuthInterceptorService,
            multi: true,
        },
    ],
};
