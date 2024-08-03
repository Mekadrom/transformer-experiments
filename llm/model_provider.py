from modules.transformer import Decoder, init_weights

class LLMTransformerModelProvider:
    def provide_transformer(self, args, vocab_size, tie_embeddings):
        model = Decoder(args, vocab_size, use_cross_attn=False).to(args.decoder_device)

        init_weights(args, model, tie_embeddings=tie_embeddings)

        return model
