import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

import { StepProgressComponent } from '../step-progress/step-progress.component';

@Component({
  selector: 'app-dataset-view',
  standalone: true,
  imports: [CommonModule, StepProgressComponent],
  templateUrl: './dataset-view.component.html',
  styleUrl: './dataset-view.component.scss'
})
export class DatasetViewComponent {

}
