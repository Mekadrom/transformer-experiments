import { ApplicationConfig } from '@angular/core';
import { provideRouter } from '@angular/router';
import { GoogleLoginProvider, SocialAuthServiceConfig  } from '@abacritt/angularx-social-login';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    {
      provide: 'SocialAuthServiceConfig',
      useValue: {
        autoLogin: false,
        providers: [
          {
            id: GoogleLoginProvider.PROVIDER_ID,
            provider: new GoogleLoginProvider(
              '1061772888483-8q2b5q4d0q3q2j5n1e2b5e4g2f1h5q.apps.googleusercontent.com'
            )
          }
        ]
      } as SocialAuthServiceConfig,
    }
  ]
};
