import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DataServiceService {
  constructor() { }

  public getProjects(): Observable<string[]> {
    return new Observable<string[]>(observer => {
      observer.next(['e2c58417-d33e-44a3-b6a4-2e8562df43c9', 'd19394d5-2158-40ea-8d51-54ad4ed2f4cc', '9fa0d020-0c4c-4b4a-bd92-21cc86c0c68b']);
      observer.complete();
    });
  }

  public getDataSetsForProject(projectId: string): Observable<any> {
    const dict: any = {'e2c58417-d33e-44a3-b6a4-2e8562df43c9': '{"data": {"train": [{"src": "good afternoon", "tgt": "guten tag"}]}}', 'd19394d5-2158-40ea-8d51-54ad4ed2f4cc': '{"data": {"train":[{"src": "goodbye", "tgt": "auf wiedersehen"}]}}', '9fa0d020-0c4c-4b4a-bd92-21cc86c0c68b': '{"data": {"train": [{"src": "hello", "tgt": "hallo"}]}}'};
    return new Observable<any>(observer => {
      observer.next(JSON.parse(dict[projectId]));
      observer.complete();
    });
  }
}
