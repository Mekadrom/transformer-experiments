from modules.transformer import Transformer

class TransformerModelProvider:
    def __init__(self):
        super().__init__()

    def provide(self, args, src_vocab_size, tgt_vocab_size, positional_encoding, tie_embeddings):
        return Transformer(
            args=args,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            positional_encoding=positional_encoding,
            tie_embeddings=tie_embeddings,
        )
