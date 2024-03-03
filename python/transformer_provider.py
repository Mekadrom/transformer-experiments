from modules.simple_transformer import SimpleTransformer
from modules.transformer import Transformer

class TransformerModelProvider:
    def provide(self, args, src_vocab_size, tgt_vocab_size, positional_encoding, use_shared_qkv, tie_embeddings):
        model = Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, positional_encoding=positional_encoding)

        model = model.to(args.device)
        model.init_weights(use_shared_qkv=use_shared_qkv, tie_embeddings=tie_embeddings)

        return model

    def provide_simple(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings):
        model = SimpleTransformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)

        model = model.to(args.device)
        model.init_weights(tie_embeddings=tie_embeddings)

        return model