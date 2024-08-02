from modules import transformer

class LLMTransformerModelProvider:
    def provide_transformer(self, args, vocab_size, tie_embeddings) -> transformer.Decoder:
        model = transformer.Decoder(args, vocab_size, use_cross_attn=False).to(args.decoder_device)

        transformer.init_weights(args, model, tie_embeddings=tie_embeddings)

        return model
