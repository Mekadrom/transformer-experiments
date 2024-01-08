import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';

import { cookies_constants } from '../constants/cookies-constants';
import { DataService } from '../services/data.service';
import { StepProgressComponent } from '../step-progress/step-progress.component';
import { Lexicon, LexiconEntry } from '../models/models';
import { utils } from '../utils/utils';

@Component({
    selector: 'app-lexicon-view',
    standalone: true,
    imports: [CommonModule, StepProgressComponent],
    templateUrl: './lexicon-view.component.html',
    styleUrl: './lexicon-view.component.scss',
})
export class LexiconViewComponent {
    selectedWord: LexiconEntry | null = null;

    constructor(private cookieService: CookieService, public dataService: DataService, private router: Router) { }

    ngOnInit(): void {
        if (utils.isNeedsAuth(this.cookieService)) {
            this.router.navigate(['/login']);
            return;
        }

        this.dataService.fetchLexicon().subscribe((lexicon) => {
            this.dataService.setLexicon(lexicon);
        });

        this.cookieService.set(cookies_constants.lastStep, '0');
    }

    addWord(): void {
        const newWord = {word: '', partOfSpeech: 'Noun', notes: '', whitelist: [], blacklist: []} as LexiconEntry;
        this.dataService.getLexicon()?.data.push(newWord);
        this.dataService.upsertLexicon().subscribe((lexicon) => {
            this.dataService.setLexicon(lexicon);
        });
    }

    deleteWord(): void {
        console.log('deleteWord');
    }

    selectWord(word: LexiconEntry): void {
        console.log('selectWord');
    }
}
