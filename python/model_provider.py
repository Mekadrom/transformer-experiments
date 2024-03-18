from modules.transformer import Transformer
from modules.vae_transformer import VAETransformer

import torch.nn as nn

class TransformerModelProvider:
    def provide_transformer(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings):
        model = Transformer(args=args, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)

        model = model.to(args.device)
        self.init_weights(model, tie_embeddings=tie_embeddings)

        return model
    
    def provide_vae_transformer(self, args, vocab_size):
        model = VAETransformer(args=args, vocab_size=vocab_size)

        model = model.to(args.device)
        self.init_weights(model)

        return model

    def init_transformer_weights(self, tie_embeddings=True):
        # Glorot uniform initialization with a gain of self.init_weights_gain
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                if self.args.init_weights_from in ['glorot_uniform', 'xavier_uniform']:
                    nn.init.xavier_uniform_(p, gain=self.args.init_weights_gain)
                elif self.args.init_weights_from in ['glorot_normal', 'xavier_normal']:
                    nn.init.xavier_normal_(p, gain=self.args.init_weights_gain)
                elif self.args.init_weights_from == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(p)
                elif self.args.init_weights_from == 'kaiming_normal':
                    nn.init.kaiming_normal_(p)
                elif self.args.init_weights_from == 'orthogonal':
                    nn.init.orthogonal_(p)
                else:
                    raise Exception(f"Unknown weight initialization method: {self.args.init_weights_from}")

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=self.args.d_model**-0.5)
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.classifier.weight = self.decoder.embedding.weight

        print("Model initialized.")
