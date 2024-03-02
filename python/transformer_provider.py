from modules.transformer import Transformer

class TransformerModelProvider:
    def __init__(self):
        super().__init__()

    def provide(self, args, src_vocab_size, tgt_vocab_size, positional_encoding, use_shared_qkv, tie_embeddings):
        model = Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, positional_encoding=positional_encoding)

        model = model.to(args.device)
        model.init_weights(use_shared_qkv=use_shared_qkv, tie_embeddings=tie_embeddings)

        return model
