from .concat_attn import ConcatenateAttention
from .linearconv_attn import LinearConvAttention
from .multihead_attn import MultiHeadAttention
from .positionwise_fcn import PositionWiseFCNetwork
from .sum import Sum
from .sum_attn import SumAttention

import admin_torch
import math
import torch.nn as nn
import utils

def get_attn_layer(args, self_attn, in_decoder):
    mha = MultiHeadAttention(args, args.d_model, in_decoder=in_decoder)

    apply_nonlinearty = 'nonlinear' in args.conv_incl_type

    if args.conv_incl_type == 'none' or args.convs_per_attn == 0:
        return mha
    elif args.conv_incl_type == 'sum':
        return SumAttention(args, mha, LinearConvAttention(args, in_decoder), apply_nonlinearity=apply_nonlinearty)
    elif args.conv_incl_type == 'concat':
        return ConcatenateAttention(args, mha, LinearConvAttention(args, in_decoder), apply_nonlinearity=apply_nonlinearty)
    else:
        raise Exception(f"Unknown convolutional inclusion type: {args.conv_incl_type}")

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.self_attn = get_attn_layer(args, True, False)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_layers)
            self.fcn_residual = admin_torch.as_module(args.n_layers)
        else:
            self.self_attn_residual = Sum()
            self.fcn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args)

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        encoder_sequences = self.self_attn_residual(encoder_sequences, self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths)[0])
        residual = encoder_sequences
        encoder_sequences, gating_variances = self.fcn(sequences=encoder_sequences)
        encoder_sequences = self.fcn_residual(residual, encoder_sequences)
        return encoder_sequences, gating_variances

class Encoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(vocab_size, args.d_model)
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
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].mha
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_encoder_layer()
                    new_layer.self_attn = layers[res_idx].mha
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_encoder_layer())
        return nn.ModuleList(layers)

    def perform_embedding_transformation(self, encoder_sequences):
        d_model = self.embedding.weight.size(1)
        return self.embedding(encoder_sequences) * math.sqrt(d_model) # (N, pad_length, d_model)

    def apply_positional_embedding(self, encoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return encoder_sequences + self.tensor_positional_encoding[:, :encoder_sequences.size(1), :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences, encoder_sequence_lengths, encoder_layer):
        return encoder_layer(encoder_sequences, encoder_sequence_lengths)

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        encoder_sequences = self.perform_embedding_transformation(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_positional_embedding(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        gating_variances = []
        for encoder_layer in self.encoder_layers:
            encoder_sequences, gating_variance = self.apply_encoder_layer(encoder_sequences, encoder_sequence_lengths, encoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        # post-LN
        encoder_sequences = self.layer_norm(encoder_sequences) # (N, pad_length, d_model)

        return encoder_sequences, gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()

        self.args = args

        self.self_attn = get_attn_layer(args, True, True)
        self.cross_attn = MultiHeadAttention(args, args.d_model, in_decoder=True)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_layers)
            self.cross_attn_residual = admin_torch.as_module(args.n_layers)
            self.fcn_residual = admin_torch.as_module(args.n_layers)
        else:
            self.self_attn_residual = Sum()
            self.cross_attn_residual = Sum()
            self.fcn_residual = Sum()

        self.fcn = PositionWiseFCNetwork(args)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        decoder_sequences = self.self_attn_residual(decoder_sequences, self.self_attn(decoder_sequences, decoder_sequences, decoder_sequences, decoder_sequence_lengths)[0]) # (N, pad_length, d_model), trash attention_weights
        decoder_sequences = self.cross_attn_residual(decoder_sequences, self.cross_attn(decoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths)[0]) # (N, pad_length, d_model)
        residual = decoder_sequences
        decoder_sequences, gating_variances = self.fcn(sequences=decoder_sequences)
        decoder_sequences = self.fcn_residual(residual, decoder_sequences) # (N, pad_length, d_model)
        return decoder_sequences, gating_variances

class Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Decoder, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(vocab_size, args.d_model)
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
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.fcn = layers[res_idx].fcn
                    layers.append(new_layer)
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(new_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_mha
                    new_layer.cross_attn = layers[res_idx].cross_mha
                    new_layer.cross_mca = layers[res_idx].cross_mca
                    layers.append(new_layer)
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    new_layer = new_decoder_layer()
                    new_layer.self_attn = layers[res_idx].self_mha
                    new_layer.cross_attn = layers[res_idx].cross_mha
                    new_layer.cross_mca = layers[res_idx].cross_mca
                    layers.append(new_layer)
            elif param_sharing_type == 'all':
                layers.append(layers[0])
            else:
                layers.append(new_decoder_layer())
        return nn.ModuleList(layers)

    def apply_embedding_transformation(self, decoder_sequences):
        d_model = self.embedding.weight.size(1)
        return self.embedding(decoder_sequences) * math.sqrt(d_model) # (N, pad_length, d_model)
    
    def apply_positional_embedding(self, decoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return decoder_sequences + self.tensor_positional_encoding[:, :decoder_sequences.size(1), :]
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer):
        return decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        decoder_sequences = self.apply_embedding_transformation(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_positional_embedding(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        gating_variances = []
        for decoder_layer in self.decoder_layers:
            decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)
        decoder_sequences = self.classifier(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences, gating_variances

class Transformer(nn.Module):
    def __init__(self, args, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()

        self.args = args

        self.encoder = Encoder(args, src_vocab_size)
        self.decoder = Decoder(args, tgt_vocab_size)

    def init_weights(self, tie_embeddings=True):
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
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.args.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.classifier.weight = self.decoder.embedding.weight

        print("Model initialized.")

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        encoder_sequences, encoder_gating_variances = self.encoder(encoder_sequences, encoder_sequence_lengths) # (N, encoder_sequence_pad_length, d_model)
        decoder_sequences, decoder_gating_variances = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths) # (N, decoder_sequence_pad_length, vocab_size)
        return decoder_sequences, encoder_gating_variances, decoder_gating_variances
