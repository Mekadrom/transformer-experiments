import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../environments/environment';
import { Project, Lexicon, Dataset, Hyperparameters, TrainedModelRef } from '../models/models';
import { utils } from '../utils/utils';

@Injectable({
    providedIn: 'root',
})
export class DataService {
    project: Project | null = null;
    datasets: Dataset[] | null = null;
    lexicon: Lexicon | null = null;
    hyperparameters: Hyperparameters | null = null;
    trainedModelRef: TrainedModelRef | null = null;

    constructor(private httpClient: HttpClient) { }

    public getMaxValidStep(): number {
        return utils.getMaxValidStep(this.getActiveProject(), this.getDatasets(), this.getLexicon(), this.getActiveHyperparameters(), this.getActiveTrainedModelRef())
    }

    public fetchProjects(): Observable<Project[]> {
        return this.httpClient.get<Project[]>(`${environment.apiBaseUrl}/projects`);
    }

    public getActiveProject(): Project | null {
        return this.project;
    }

    public setActiveProject(project: Project | null): void {
        this.project = project;
    }

    public upsertProject(): Observable<Project> {
        return this.httpClient.post<Project>(`${environment.apiBaseUrl}/projects`, this.project);
    }

    public deleteProject(): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        return this.httpClient.delete(`${environment.apiBaseUrl}/projects/${this.project.projectKey}`);
    }

    public fetchDatasets(): Observable<Dataset[]> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        return this.httpClient.get<Dataset[]>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/datasets`);
    }

    public getDatasets(): Dataset[] | null {
        return this.datasets;
    }

    public setDatasets(datasets: Dataset[]): void {
        this.datasets = datasets;
    }

    public fetchDatasetData(dataset: Dataset): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.datasets) {
            throw new Error('No datasets loaded');
        }
        return this.httpClient.get<any>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/datasets/${dataset.datasetKey}`);
    }

    public upsertDataset(dataset: Dataset): Observable<Dataset> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.datasets) {
            throw new Error('No datasets loaded');
        }
        return this.httpClient.post<Dataset>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/datasets`, dataset);
    }

    public deleteDataset(dataset: Dataset): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.datasets) {
            throw new Error('No datasets loaded');
        }
        return this.httpClient.delete(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/datasets/${dataset.datasetKey}`);
    }

    public appendToDataset(dataset: Dataset, newData: any): Observable<Dataset> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        return this.httpClient.patch<Dataset>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/datasets/${dataset.datasetKey}/append`, newData);
    }

    public fetchLexicon(): Observable<Lexicon> {
        return this.httpClient.get<Lexicon>(`${environment.apiBaseUrl}/lexicons`);
    }

    public getLexicon(): Lexicon | null {
        return this.lexicon;
    }

    public setLexicon(lexicon: Lexicon): void {
        this.lexicon = lexicon;
    }

    public upsertLexicon(): Observable<Lexicon> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.lexicon) {
            throw new Error('No lexicon selected');
        }
        return this.httpClient.post<Lexicon>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/lexicon`, this.lexicon);
    }

    public fetchHyperparameterConfigurations(): Observable<Hyperparameters[]> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.datasets) {
            throw new Error('No datasets loaded');
        }
        return this.httpClient.get<Hyperparameters[]>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/hyperparameters`);
    }

    public getActiveHyperparameters(): Hyperparameters | null {
        return this.hyperparameters;
    }

    public setActiveHyperparameters(hyperparameters: Hyperparameters): void {
        this.hyperparameters = hyperparameters;
    }

    public upsertHyperparameters(): Observable<Hyperparameters> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.hyperparameters) {
            throw new Error('No hyperparameters selected');
        }
        return this.httpClient.post<Hyperparameters>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/hyperparameters`, this.hyperparameters);
    }

    public deleteHyperparameters(hyperparameters: Hyperparameters): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!hyperparameters) {
            throw new Error('No hyperparameters selected');
        }
        return this.httpClient.delete(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/hyperparameters/${hyperparameters.hyperparametersKey}`);
    }

    public trainModel(): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.hyperparameters) {
            throw new Error('No hyperparameters selected');
        }
        return this.httpClient.post(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/train`, this.hyperparameters.hyperparametersKey);
    }

    public fetchTrainedModelRefs(): Observable<TrainedModelRef[]> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        return this.httpClient.get<TrainedModelRef[]>(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/models`);
    }

    public getActiveTrainedModelRef(): TrainedModelRef | null {
        return this.trainedModelRef;
    }

    public setActiveTrainedModelRef(trainedModelRef: TrainedModelRef): void {
        this.trainedModelRef = trainedModelRef;
    }

    public retrainModel(): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.hyperparameters) {
            throw new Error('No hyperparameters selected');
        }
        if (!this.trainedModelRef) {
            throw new Error('No trained model selected');
        }
        return this.httpClient.post(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/models/${this.trainedModelRef.trainedModelKey}/train`, this.hyperparameters.hyperparametersKey);
    }

    public inference(input: string): Observable<any> {
        if (!this.project) {
            throw new Error('No project selected');
        }
        if (!this.hyperparameters) {
            throw new Error('No hyperparameters selected');
        }
        if (!this.trainedModelRef) {
            throw new Error('No trained model selected');
        }
        return this.httpClient.post(`${environment.apiBaseUrl}/projects/${this.project.projectKey}/models/${this.trainedModelRef.trainedModelKey}/inference`, {input: input});
    }
}
