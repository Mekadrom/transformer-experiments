from modules.multihead_attn import MultiHeadAttention
from modules.positionwise_fcn import PositionWiseFCNetwork
from modules.sum import Sum
from modules.liteconv import LiteConv
from modules import reparameterize

import admin_torch
import math
import torch
import torch.nn as nn
import utils

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(args, self_attn=True, in_decoder=False)
        # self.lite_conv_self_attn = LiteConv(args, self_attn=True, in_decoder=False)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_layers)
            self.fcn_residual = admin_torch.as_module(args.n_layers)
        else:
            self.self_attn_residual = Sum()
            self.fcn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args)

    def forward(self, encoder_sequences, encoder_sequence_lengths, key_padding_mask):
        self_attn, _, q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar = self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths, key_padding_mask)

        encoder_sequences = self.self_attn_residual(encoder_sequences, self_attn)

        fcn, gating_variances = self.fcn(encoder_sequences)

        encoder_sequences = self.fcn_residual(encoder_sequences, fcn)
            
        return encoder_sequences, q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar, gating_variances

class Encoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()

        self.args = args

        self.d_model = args.d_model * 2 if 't' in args.latent_repr_type else args.d_model

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.apply_dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.encoder_layers = self.make_encoder_layers(args.n_encoder_layers, args.encoder_param_sharing_type, args.m_encoder_independent_layers)

        if args.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(utils.get_positional_encoding(args))

    def make_encoder_layers(self, n_layers, param_sharing_type, m_independent_layers):
        def new_encoder_layer():
            return EncoderLayer(self.args)

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_encoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_encoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_encoder_layer())
        return nn.ModuleList(layers)

    def perform_embedding_transformation(self, encoder_sequences):
        embedded = self.embedding(encoder_sequences) * math.sqrt(self.args.d_model) # (N, pad_length, d_model)

        # if VAE tokens are used, split the embedding into mu and logvar and sample for forward pass
        # this happens before positional encoding and dropout for a good reason
        if 'latent_repr_type' in self.args and 't' in self.args.latent_repr_type:
            t_mu, t_logvar = torch.chunk(embedded, 2, dim=-1)
            embedded = reparameterize(t_mu, t_logvar)
        else:
            t_mu, t_logvar = None, None
        return embedded, t_mu, t_logvar

    def apply_positional_embedding(self, encoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return encoder_sequences + self.tensor_positional_encoding[:, :encoder_sequences.size(1), :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences, encoder_sequence_lengths, key_padding_mask, encoder_layer):
        return encoder_layer(encoder_sequences, encoder_sequence_lengths, key_padding_mask)

    def forward(self, encoder_sequences, encoder_sequence_lengths, key_padding_mask):
        encoder_sequences = self.perform_embedding_transformation(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences, t_mu, t_logvar = self.apply_positional_embedding(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        q_mus = []
        q_logvars = []
        k_mus = []
        k_logvars = []
        v_mus = []
        v_logvars = []
        gating_variances = []
        for encoder_layer in self.encoder_layers:
            encoder_sequences, q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar, gating_variance = self.apply_encoder_layer(encoder_sequences, encoder_sequence_lengths, key_padding_mask, encoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)
            if q_mu is not None:
                q_mus.append(q_mu)
                q_logvars.append(q_logvar)
            if k_mu is not None:
                k_mus.append(k_mu)
                k_logvars.append(k_logvar)
            if v_mu is not None:
                v_mus.append(v_mu)
                v_logvars.append(v_logvar)

        # post-LN
        encoder_sequences = self.layer_norm(encoder_sequences) # (N, pad_length, d_model)

        q_mus = torch.stack(q_mus, dim=0) if len(q_mus) > 0 else None
        q_logvars = torch.stack(q_logvars, dim=0) if len(q_logvars) > 0 else None
        k_mus = torch.stack(k_mus, dim=0) if len(k_mus) > 0 else None
        k_logvars = torch.stack(k_logvars, dim=0) if len(k_logvars) > 0 else None
        v_mus = torch.stack(v_mus, dim=0) if len(v_mus) > 0 else None
        v_logvars = torch.stack(v_logvars, dim=0) if len(v_logvars) > 0 else None

        return encoder_sequences, (t_mu, t_logvar), (q_mus, q_logvars), (k_mus, k_logvars), (v_mus, v_logvars), gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()

        self.args = args

        self.self_attn = MultiHeadAttention(args, self_attn=True, in_decoder=True)
        self.cross_attn = MultiHeadAttention(args, self_attn=False, in_decoder=True)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_layers)
            self.cross_attn_residual = admin_torch.as_module(args.n_layers)
            self.fcn_residual = admin_torch.as_module(args.n_layers)
        else:
            self.self_attn_residual = Sum()
            self.cross_attn_residual = Sum()
            self.fcn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask=None):
        if attn_mask is None:
            ones = torch.ones(decoder_sequences.size(1), decoder_sequences.size(1)).to(decoder_sequences.device)
            attn_mask = torch.triu(ones, diagonal=1).bool()

        self_attn, _, s_q_mu, s_q_logvar, s_k_mu, s_k_logvar, s_v_mu, s_v_logvar = self.self_attn(decoder_sequences, decoder_sequences, decoder_sequences, decoder_sequence_lengths, tgt_key_padding_mask, attn_mask)
        decoder_sequences = self.self_attn_residual(decoder_sequences, self_attn)

        cross_attn, _, c_q_mu, c_q_logvar, c_k_mu, c_k_logvar, c_v_mu, c_v_logvar = self.cross_attn(decoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask)
        decoder_sequences = self.cross_attn_residual(decoder_sequences, cross_attn)

        fcn, gating_variances = self.fcn(decoder_sequences)
        
        decoder_sequences = self.fcn_residual(decoder_sequences, fcn)

        return decoder_sequences, s_q_mu, s_q_logvar, s_k_mu, s_k_logvar, s_v_mu, s_v_logvar, c_q_mu, c_q_logvar, c_k_mu, c_k_logvar, c_v_mu, c_v_logvar, gating_variances

class Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Decoder, self).__init__()

        self.args = args

        self.d_model = args.d_model * 2 if 't' in args.latent_repr_type else args.d_model

        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.apply_dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.decoder_layers = self.make_decoder_layers(args.n_decoder_layers, args.decoder_param_sharing_type, args.m_decoder_independent_layers)
        self.classifier = nn.Linear(args.d_model, vocab_size)

        if args.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(utils.get_positional_encoding(args))

    def make_decoder_layers(self, n_layers, param_sharing_type, m_independent_layers):
        def new_decoder_layer():
            return DecoderLayer(self.args)
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(new_decoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(new_decoder_layer())
                else:
                    layers.append(layers[i - 1])
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(layers[res_idx])
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(layers[res_idx])
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    if hasattr(layers[res_idx], 'fcn_residual'):
                        new_layer.fcn_residual = layers[res_idx].fcn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    if hasattr(layers[res_idx], 'cross_attn_residual'):
                        new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_attn
                    new_layer.cross_attn = layers[res_idx].cross_attn
                    if hasattr(layers[res_idx], 'self_attn_residual'):
                        new_layer.self_attn_residual = layers[res_idx].self_attn_residual
                    if hasattr(layers[res_idx], 'cross_attn_residual'):
                        new_layer.cross_attn_residual = layers[res_idx].cross_attn_residual
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_decoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, decoder_sequences):
        embedded = self.embedding(decoder_sequences) * math.sqrt(self.d_model) # (N, pad_length, d_model)

        # if VAE tokens are used, split the embedding into mu and logvar and sample for forward pass
        # this happens before positional encoding and dropout for a good reason
        if 't' in self.args.latent_repr_type:
            t_mu, t_logvar = torch.chunk(embedded, 2, dim=-1)
            embedded = reparameterize(t_mu, t_logvar)
        else:
            t_mu, t_logvar = None, None
        return embedded, t_mu, t_logvar
    
    def apply_positional_embedding(self, decoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return decoder_sequences + self.tensor_positional_encoding[:, :decoder_sequences.size(1), :]
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask, decoder_layer):
        return decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask=None):
        decoder_sequences, t_mu, t_logvar = self.apply_embedding_transformation(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_positional_embedding(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        s_q_mus = []
        s_q_logvars = []
        s_k_mus = []
        s_k_logvars = []
        s_v_mus = []
        s_v_logvars = []
        c_q_mus = []
        c_q_logvars = []
        c_k_mus = []
        c_k_logvars = []
        c_v_mus = []
        c_v_logvars = []
        gating_variances = []
        for decoder_layer in self.decoder_layers:
            decoder_sequences, s_q_mu, s_q_logvar, s_k_mu, s_k_logvar, s_v_mu, s_v_logvar, c_q_mu, c_q_logvar, c_k_mu, c_k_logvar, c_v_mu, c_v_logvar, gating_variance = self.apply_decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)
            if s_q_mu is not None:
                s_q_mus.append(s_q_mu)
                s_q_logvars.append(s_q_logvar)
            if s_k_mu is not None:
                s_k_mus.append(s_k_mu)
                s_k_logvars.append(s_k_logvar)
            if s_v_mu is not None:
                s_v_mus.append(s_v_mu)
                s_v_logvars.append(s_v_logvar)
            if c_q_mu is not None:
                c_q_mus.append(c_q_mu)
                c_q_logvars.append(c_q_logvar)
            if c_k_mu is not None:
                c_k_mus.append(c_k_mu)
                c_k_logvars.append(c_k_logvar)
            if c_v_mu is not None:
                c_v_mus.append(c_v_mu)
                c_v_logvars.append(c_v_logvar)

        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)
        decoder_sequences = self.classifier(decoder_sequences)  # (N, pad_length, vocab_size)

        s_q_mus = torch.stack(s_q_mus, dim=0) if len(s_q_mus) > 0 else None
        s_q_logvars = torch.stack(s_q_logvars, dim=0) if len(s_q_logvars) > 0 else None
        s_k_mus = torch.stack(s_k_mus, dim=0) if len(s_k_mus) > 0 else None
        s_k_logvars = torch.stack(s_k_logvars, dim=0) if len(s_k_logvars) > 0 else None
        s_v_mus = torch.stack(s_v_mus, dim=0) if len(s_v_mus) > 0 else None
        s_v_logvars = torch.stack(s_v_logvars, dim=0) if len(s_v_logvars) > 0 else None
        c_q_mus = torch.stack(c_q_mus, dim=0) if len(c_q_mus) > 0 else None
        c_q_logvars = torch.stack(c_q_logvars, dim=0) if len(c_q_logvars) > 0 else None
        c_k_mus = torch.stack(c_k_mus, dim=0) if len(c_k_mus) > 0 else None
        c_k_logvars = torch.stack(c_k_logvars, dim=0) if len(c_k_logvars) > 0 else None
        c_v_mus = torch.stack(c_v_mus, dim=0) if len(c_v_mus) > 0 else None
        c_v_logvars = torch.stack(c_v_logvars, dim=0) if len(c_v_logvars) > 0 else None

        return decoder_sequences, (t_mu, t_logvar), (s_q_mus, s_q_logvars), (s_k_mus, s_k_logvars), (s_v_mus, s_v_logvars), (c_q_mus, c_q_logvars), (c_k_mus, c_k_logvars), (c_v_mus, c_v_logvars), gating_variances

class Transformer(nn.Module):
    def __init__(self, args, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()
        self.args = args

        self.encoder = Encoder(args, src_vocab_size)
        self.decoder = Decoder(args, tgt_vocab_size)

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask=None):
        encoder_sequences, e_t_vars, e_q_vars, e_k_vars, e_v_vars, encoder_gating_variances = self.encoder(encoder_sequences, encoder_sequence_lengths, src_key_padding_mask) # (N, encoder_sequence_pad_length, d_model)
        decoder_sequences, d_t_vars, d_s_q_vars, d_s_k_vars, d_s_v_vars, d_c_q_vars, d_c_k_vars, d_c_v_vars, decoder_gating_variances = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, src_key_padding_mask, tgt_key_padding_mask, attn_mask) # (N, decoder_sequence_pad_length, vocab_size)
        return decoder_sequences, e_t_vars, e_q_vars, e_k_vars, e_v_vars, encoder_gating_variances, d_t_vars, d_s_q_vars, d_s_k_vars, d_s_v_vars, d_c_q_vars, d_c_k_vars, d_c_v_vars, decoder_gating_variances
