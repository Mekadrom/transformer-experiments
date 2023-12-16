import { Routes } from '@angular/router';

import { AutoRedirectComponent } from './auto-redirect/auto-redirect.component';
import { OauthViewComponent } from './oauth-view/oauth-view.component';
import { ProjectViewComponent } from './project-view/project-view.component';
import { LexiconViewComponent } from './lexicon-view/lexicon-view.component';
import { DatasetViewComponent } from './dataset-view/dataset-view.component';
import { HyperParametersViewComponent } from './hyper-parameters-view/hyper-parameters-view.component';
import { TrainModelViewComponent } from './train-model-view/train-model-view.component';
import { FineTuneViewComponent } from './fine-tune-view/fine-tune-view.component';
import { InferenceViewComponent } from './inference-view/inference-view.component';
import { HelpViewComponent } from './help-view/help-view.component';

export const routes: Routes = [
    { path: '', component: AutoRedirectComponent },
    { path: 'login', component: OauthViewComponent },
    { path: 'step1', component: ProjectViewComponent },
    { path: 'step2', component: DatasetViewComponent },
    { path: 'step3', component: LexiconViewComponent },
    { path: 'step4', component: HyperParametersViewComponent },
    { path: 'step5', component: TrainModelViewComponent },
    { path: 'step6', component: FineTuneViewComponent },
    { path: 'inference', component: InferenceViewComponent },
    { path: 'help', component: HelpViewComponent },
];
