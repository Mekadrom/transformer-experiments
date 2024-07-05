from .transformer import Encoder, Decoder

import torch
import torch.nn as nn

class VAETransformer(nn.Module):
    def __init__(self, args, vocab_size):
        super(VAETransformer, self).__init__()

        self.args = args

        self.encoder = Encoder(args, vocab_size)

        self.mu = nn.Linear(args.d_model, args.latent_size)
        self.logvar = nn.Linear(args.d_model, args.latent_size)
        self.decoder_extrapolator = nn.Linear(args.latent_size, args.d_model * args.latent_seq_len)

        self.decoder = Decoder(args, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask=None):
        encoder_sequences, _ = self.encoder(encoder_sequences, encoder_sequence_lengths, src_key_padding_mask)

        cls_token = encoder_sequences[:, 0, :]

        mu = self.mu(cls_token)
        logvar = self.logvar(cls_token)
        z = self.reparameterize(mu, logvar).unsqueeze(1) # shape: (batch_size, 1, latent_size) or (batch_size, sequence_length, latent_size)
        z_length = torch.ones(z.size(0), dtype=torch.long, device=z.device).unsqueeze(1) # shape: (batch_size, 1)

        z = self.decoder_extrapolator(z).view(z.size(0), -1, self.args.d_model) # shape: (batch_size, latent_seq_len, d_model)
        decoder_sequences, _ = self.decoder(decoder_sequences, decoder_sequence_lengths, z, z_length, torch.zeros([src_key_padding_mask.size(0), self.args.latent_seq_len]).bool().to(self.args.decoder_device), tgt_key_padding_mask, attn_mask)

        return decoder_sequences, mu, logvar
