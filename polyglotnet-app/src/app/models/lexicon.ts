export default interface Lexicon {
    projectKey: string; // a uuid reference to the project that owns this lexicon. one-to-one relationship (one project has one lexicon)
    srcLanguage: string; // the name of the src language for this lexicon/project
    tgtLanguage: string; // the name of the tgt language for this lexicon/project
    data: any; // json string containing the lexicon data. list of <tgt> <pos> (part of speech) pairs. the source language does not matter to the lexicon, it is only used for display purposes. the lexicon is used for generating training data.
};
