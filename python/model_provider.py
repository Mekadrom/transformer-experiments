from modules.transformer import Transformer
from modules.vae_transformer import VAETransformer

import torch.nn as nn

class TransformerModelProvider:
    def provide_transformer(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings):
        model = Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)

        model = model.to(args.device)
        self.init_transformer_weights(args, model, tie_embeddings=tie_embeddings)

        return model
    
    def provide_vae_transformer(self, args, vocab_size):
        model = VAETransformer(args=args, vocab_size=vocab_size)

        model = model.to(args.device)
        self.init_transformer_weights(args, model, args.vae_tie_embeddings)

        return model

    def init_transformer_weights(self, args, model, tie_embeddings=True):
        # Glorot uniform initialization with a gain of self.args.init_weights_gain
        for p in model.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                if args.init_weights_from in ['glorot_uniform', 'xavier_uniform']:
                    nn.init.xavier_uniform_(p, gain=args.init_weights_gain)
                elif args.init_weights_from in ['glorot_normal', 'xavier_normal']:
                    nn.init.xavier_normal_(p, gain=args.init_weights_gain)
                elif args.init_weights_from == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(p)
                elif args.init_weights_from == 'kaiming_normal':
                    nn.init.kaiming_normal_(p)
                elif args.init_weights_from == 'orthogonal':
                    nn.init.orthogonal_(p)
                else:
                    raise Exception(f"Unknown weight initialization method: {args.init_weights_from}")

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(model.encoder.embedding.weight, mean=0., std=args.d_model**-0.5)
        model.decoder.embedding.weight = model.encoder.embedding.weight

        if tie_embeddings:
            model.decoder.classifier.weight = model.decoder.embedding.weight

        print("Model initialized.")
