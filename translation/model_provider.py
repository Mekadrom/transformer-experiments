from modules import rmsnorm, transformer
from torch import nn

class TranslationTransformerModelProvider:
    def provide_transformer(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings) -> transformer.Transformer:
        norm = rmsnorm.RMSNorm if args.norm_type == 'rms' else nn.LayerNorm

        model = transformer.Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, norm=norm)

        model.encoder = model.encoder.to(args.encoder_device)
        model.decoder = model.decoder.to(args.decoder_device)
        
        return model
    