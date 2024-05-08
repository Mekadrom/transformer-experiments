from modules.transformer import Decoder

import utils

class LLMTransformerModelProvider:
    def provide_transformer(self, args, vocab_size, tie_embeddings):
        model = Decoder(args, vocab_size)
        model = model.to(args.device)

        utils.init_transformer_weights(args, model, tie_embeddings=tie_embeddings)

        return model
