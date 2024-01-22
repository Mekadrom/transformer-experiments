from modules.transformer import Transformer

class TransformerModelProvider:
    def __init__(self):
        super().__init__()

    def provide(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings, positional_encoding):
        return Transformer(
            args=args,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            tie_embeddings=tie_embeddings,
            positional_encoding=positional_encoding,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_queries=args.d_queries,
            d_values=args.d_values,
            d_inner=args.d_inner,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            dropout=args.dropout
        )
