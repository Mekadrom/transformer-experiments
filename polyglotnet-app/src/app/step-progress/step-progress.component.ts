import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatDividerModule } from '@angular/material/divider';
import { Router } from '@angular/router';

import { DataService } from '../services/data.service';

@Component({
    selector: 'app-step-progress',
    standalone: true,
    imports: [CommonModule, MatDividerModule],
    templateUrl: './step-progress.component.html',
    styleUrl: './step-progress.component.scss',
})
export class StepProgressComponent {
    steps: string[] = [
        'Project',
        'Training Data',
        'Lexicon',
        'Hyperparameters',
        'Train Model',
        'Fine Tune',
    ];

    constructor(private router: Router, private dataService: DataService) { }

    isActiveStep(index: number): boolean {
        const activeStep = +this.router.url.slice(-1);
        return index + 1 === activeStep;
    }

    isCompletedStep(index: number): boolean {
        const activeStep = +this.router.url.slice(-1);
        return index + 1 < activeStep;
    }

    isDisabledStep(index: number): boolean {
        return this.dataService.getMaxValidStep() < index + 1;
    }

    isInferenceActive(): boolean {
        return this.router.url === '/inference';
    }

    isInferenceDisabled(): boolean {
        return this.dataService.getMaxValidStep() < 7;
    }

    onStepClick(path: string): void {
        if (this.isDisabledStep(+path.slice(-1) - 1)) {
            return;
        }
        if (path === 'inference' && this.isInferenceDisabled()) {
            return;
        }
        this.router.navigate([`/${path}`]);
    }
}
