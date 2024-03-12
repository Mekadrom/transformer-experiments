from modules.transformer import Transformer

class TransformerModelProvider:
    def provide(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings):
        model = Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)

        model = model.to(args.device)
        model.init_weights(tie_embeddings=tie_embeddings)

        return model
