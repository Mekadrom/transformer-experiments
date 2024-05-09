from modules.rmsnorm import RMSNorm
from modules.transformer import Transformer
from modules.vae_transformer import VAETransformer

import torch.nn as nn
import utils

class TranslationTransformerModelProvider:
    def provide_transformer(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings):
        norm = RMSNorm if args.norm_type == 'rms' else nn.LayerNorm

        model = Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, norm=norm)

        model = model.to(args.device)
        utils.init_transformer_weights(args, model, tie_embeddings=tie_embeddings)

        return model
    
    def provide_vae_transformer(self, args, vocab_size):
        model = VAETransformer(args=args, vocab_size=vocab_size)

        model = model.to(args.device)
        utils.init_transformer_weights(args, model, args.vae_tie_embeddings)

        return model
