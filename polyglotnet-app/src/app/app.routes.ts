import { Routes } from '@angular/router';

import { AutoRedirectComponent } from './auto-redirect/auto-redirect.component';
import { ProjectViewComponent } from './project-view/project-view.component';
import { LexiconViewComponent } from './lexicon-view/lexicon-view.component';
import { TrainingDataUploadViewComponent } from './training-data-upload-view/training-data-upload-view.component';
import { TrainingDataAugmentationViewComponent } from './training-data-augmentation-view/training-data-augmentation-view.component';
import { HyperParametersViewComponent } from './hyper-parameters-view/hyper-parameters-view.component';
import { TrainModelViewComponent } from './train-model-view/train-model-view.component';
import { FineTuneViewComponent } from './fine-tune-view/fine-tune-view.component';
import { InferenceViewComponent } from './inference-view/inference-view.component';
import { HelpViewComponent } from './help-view/help-view.component';

export const routes: Routes = [
    { path: '', component: AutoRedirectComponent },
    { path: 'step1', component: ProjectViewComponent },
    { path: 'step2', component: LexiconViewComponent },
    { path: 'step3', component: TrainingDataUploadViewComponent },
    { path: 'step4', component: TrainingDataAugmentationViewComponent },
    { path: 'step5', component: HyperParametersViewComponent },
    { path: 'step6', component: TrainModelViewComponent },
    { path: 'step7', component: FineTuneViewComponent },
    { path: 'inference', component: InferenceViewComponent },
    { path: 'help', component: HelpViewComponent },
];
