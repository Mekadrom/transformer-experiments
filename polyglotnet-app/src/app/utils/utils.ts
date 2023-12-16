import { CookieService } from 'ngx-cookie-service';

import { cookies_constants } from '../constants/cookies-constants';
import { Project, Lexicon, Dataset, Hyperparameters, TrainedModelRef } from '../models/models';

export const utils = {
    getSub: function (token: string) {
        if (!token) {
            return null;
        }
        if (token.startsWith('Bearer ')) {
            token = token.split(' ')[1];
        }
        const sub = JSON.parse(atob(token.split('.')[1])).sub;
        return sub;
    },
    isExpired: function (token: string) {
        if (!token) {
            return true;
        }
        if (token.startsWith('Bearer ')) {
            token = token.split(' ')[1];
        }
        const expiry = JSON.parse(atob(token.split('.')[1])).exp;
        return Math.floor(new Date().getTime() / 1000) >= expiry;
    },
    isNeedsAuth: function (cookieService: CookieService) {
        return !cookieService.check(cookies_constants.authorization) || utils.isExpired(cookieService.get(cookies_constants.authorization));
    },
    getMaxValidStep: function (project: Project | null, datasets: Dataset[] | null, lexicon: Lexicon | null, hyperparameters: Hyperparameters | null, trainedModelRef: TrainedModelRef | null): number {
        if (!project) {
            return 1;
        }
        if (!datasets) {
            return 2;
        }
        if (!lexicon) {
            return 3;
        }
        if (!hyperparameters) {
            return 4;
        }
        if (!trainedModelRef) {
            return 5;
        }
        // 6 and 7 both open up if there is a trained model for this project
        return 7;
    }
};
