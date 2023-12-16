import { SocialAuthService } from '@abacritt/angularx-social-login';
import { CommonModule } from '@angular/common';
import { Component, isDevMode, Renderer2, OnInit } from '@angular/core';
import {
    Router,
    RouterLink,
    RouterLinkActive,
    RouterOutlet,
} from '@angular/router';
import { CookieService } from 'ngx-cookie-service';
import { cookies_constants } from './constants/cookies-constants';

@Component({
    selector: 'app-root',
    standalone: true,
    imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss',
})
export class AppComponent {
    title = 'polyglotnet-app';
    homeTabPages = [
        '/',
        '/login',
        '/step1',
        '/step2',
        '/step3',
        '/step4',
        '/step5',
        '/step6',
        '/step7',
        '/inference',
    ];

    constructor(
        private renderer: Renderer2,
        private router: Router,
        private socialAuthService: SocialAuthService,
        private cookieService: CookieService,
    ) {}

    navigate(e: MouseEvent, id: string): void {
        if (isDevMode()) {
            console.log(e);
        }

        this.renderer.selectRootElement(id).click();
    }

    getCurrentRoute(): string {
        return this.router.url;
    }

    logout(): void {
        this.socialAuthService.signOut();
        this.cookieService.delete(cookies_constants.authorization);
        this.cookieService.delete(cookies_constants.sessionId);
        this.router.navigate(['/login']);
    }
}
