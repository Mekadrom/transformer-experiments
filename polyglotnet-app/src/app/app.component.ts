import { Component, isDevMode, Renderer2 } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router , RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive, RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'polyglotnet-app';

  constructor(
    private renderer: Renderer2,
    private router: Router,
  ) { }

  navigate(e: MouseEvent, id: string): void {
    if (isDevMode()) {
      console.log(e);
    }

    const targetElement = e.target as HTMLElement;
    this.renderer.selectRootElement(id).click();
  }

  getCurrentRoute(): string {
    return this.router.url;
  }
}
