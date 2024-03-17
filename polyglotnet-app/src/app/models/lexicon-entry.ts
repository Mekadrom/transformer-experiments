export default interface LexiconEntry {
    word: string;
    partOfSpeech: string;
    notes: string;
    whitelist: String[];
    blacklist: String[];
}
