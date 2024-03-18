from .transformer import Encoder, Decoder

import torch
import torch.nn as nn

class VAETransformer(nn.Module):
    def __init__(self, args, vocab_size):
        super(VAETransformer, self).__init__()

        self.args = args

        self.encoder = Encoder(args, vocab_size)
        self.decoder = Decoder(args, vocab_size)

        self.mu = nn.Linear(args.d_model, args.latent_size)
        self.logvar = nn.Linear(args.d_model, args.latent_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        encoder_sequences, _ = self.encoder(encoder_sequences, encoder_sequence_lengths)

        cls_token = encoder_sequences[:, 0, :]

        mu = self.mu(cls_token)
        logvar = self.logvar(cls_token)
        z = self.reparameterize(mu, logvar)
        decoder_sequences, _ = self.decoder(decoder_sequences, decoder_sequence_lengths, z, decoder_sequence_lengths)
        return decoder_sequences, mu, logvar
