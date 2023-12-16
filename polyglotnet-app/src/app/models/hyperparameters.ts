export default interface Hyperparameters {
    name: string; // a name for this set of hyperparameters
    description: string; // a description for this set of hyperparameters, basically a place for users to store comments for later reference
    projectKey: string | null; // a uuid reference to the project that owns this set of hyperparameters. many-to-one relationship (one project has many hyperparameters). null when new hyperparameters
    hyperparametersKey: string | null; // a uuid reference to this set of hyperparameters. null when new hyperparameters

    // now for the parameters themselves

    // tokenizer parameters
    tokenizer: "BPE" | "WordPiece" | "SentencePiece";
    src_vocab_size: number;
    tgt_vocab_size: number;
    shared_vocab: boolean;
    max_length: number;
    min_length: number;
    max_length_ratio: number;

    // model parameters
    d_model: number;
    n_heads: number;
    d_queries: number;
    d_values: number;
    d_inner: number;
    n_encoder_layers: number;
    n_decoder_layers: number;
    dropout: number; // technically a training parameter but is used when initializing the model so technically part of the architecture even if not relevant to inference
    positional_encoding_type: "sinusoidal" | "rotary" | "rotary_learned";
    rotary_positional_encoding_dim: number;

    // training parameters
    tokens_in_batch: number;
    target_tokens_per_batch: number;
    n_steps: number;
    warmup_steps: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    label_smoothing: number;
};

// default parameters. these will be updated as better on average parameters are found
export const defaultHyperparameters: Hyperparameters = {
    name: "New Hyperparameters",
    description: "",
    projectKey: null,
    hyperparametersKey: null,

    // now for the parameters themselves

    // tokenizer parameters
    tokenizer: "BPE",
    src_vocab_size: 37000,
    tgt_vocab_size: 0,
    shared_vocab: true,
    max_length: 160,
    min_length: 3,
    max_length_ratio: 1.5,

    // model parameters
    d_model: 512,
    n_heads: 8,
    d_queries: 64,
    d_values: 64,
    d_inner: 2048,
    n_encoder_layers: 6,
    n_decoder_layers: 6,
    dropout: 0.1, // technically a training parameter but is used when initializing the model so technically part of the architecture even if not relevant to inference
    positional_encoding_type: "sinusoidal",
    rotary_positional_encoding_dim: 0,

    // training parameters
    tokens_in_batch: 2000,
    target_tokens_per_batch: 25000,
    n_steps: 100000,
    warmup_steps: 8000,
    beta1: 0.9,
    beta2: 0.98,
    epsilon: 1e-9,
    label_smoothing: 0.1,
};
