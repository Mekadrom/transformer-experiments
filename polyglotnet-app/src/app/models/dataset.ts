export default interface Dataset {
    name: string; // the name for this dataset (for display and logging purposes)
    description: string; // a description for this dataset, basically a place for users to store comments on the datasets
    projectKey: string; // a uuid reference to the project that owns this dataset. many-to-one relationship (one project has many datasets).
    data: any | null; // empty when dataset is not actively being viewed in order to save memory, populated when viewed for editing or when uploading a new dataset
    datasetKey: string | null; // a uuid reference to this dataset. null when new dataset
};
