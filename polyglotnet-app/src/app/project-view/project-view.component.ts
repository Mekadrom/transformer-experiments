import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';

import { cookies_constants } from '../constants/cookies-constants';
import { DataService } from '../services/data.service';
import { Project } from '../models/models';
import { StepProgressComponent } from '../step-progress/step-progress.component';
import { utils } from '../utils/utils';

@Component({
    selector: 'app-project-view',
    standalone: true,
    imports: [CommonModule, StepProgressComponent],
    templateUrl: './project-view.component.html',
    styleUrl: './project-view.component.scss',
})
export class ProjectViewComponent implements OnInit {
    projects: Project[] = [];

    constructor(private cookieService: CookieService, public dataService: DataService, private router: Router) { }

    ngOnInit(): void {
        if (utils.isNeedsAuth(this.cookieService)) {
            this.router.navigate(['/login']);
            return;
        }

        this.dataService.fetchProjects().subscribe((projects) => {
            this.projects = projects;
            if (projects.length === 0) {
                return;
            }
            this.dataService.setActiveProject(projects[0]);
        });

        this.cookieService.set(cookies_constants.lastStep, '0');
    }

    selectProject(project: Project): void {
        this.dataService.setActiveProject(project);
    }

    nextStep(): void {
        this.router.navigate(['/step2']);
    }

    createProject(): void {
        const newProject = {name: 'New project', description: ''} as Project;
        this.dataService.setActiveProject(newProject);
        this.dataService.upsertProject().subscribe((project) => {
            this.dataService.fetchProjects().subscribe((projects) => {
                this.projects = projects;
                this.dataService.setActiveProject(project);
            });
        });
    }

    deleteProject(): void {
        if (!this.dataService.getActiveProject()) {
            throw new Error('No active project');
        }
        this.dataService.deleteProject().subscribe(() => {
            this.dataService.fetchProjects().subscribe((projects) => {
                this.projects = projects;
                if (projects.length === 0) {
                    this.dataService.setActiveProject(null);
                    return;
                }
                this.dataService.setActiveProject(projects[0]);
            });
        });
    }
}
