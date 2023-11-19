import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
  selector: 'app-auto-redirect',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './auto-redirect.component.html',
  styleUrl: './auto-redirect.component.scss'
})
export class AutoRedirectComponent {
  constructor(private router: Router) { }

  ngOnInit(): void {
    
  }
}
