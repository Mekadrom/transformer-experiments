export default interface TrainedModelRef {
    name: string; // the name for this trained model (for display and logging purposes)
    notes: string; // a place for notes
    projectKey: string; // a uuid reference to the project that owns this trained model. many-to-one relationship (one project has many trained models)
    trainedModelKey: string; // a uuid reference to this trained model
    hyperparametersKey: string; // the hyperparameters that were used to train this model initially. for finetuned models, a separate trained model ref is created with the new hyperparameters
};
