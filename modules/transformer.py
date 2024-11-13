from modules import embedding_mlp, millions_moe, positionwise_fcn, multihead_attn, sum
from torch import nn
from typing import List

import admin_torch
import math
import torch
import utils

def init_weights(args, model: nn.Module, tie_embeddings):
    # glorot uniform initialization with a gain of self.args.init_weights_gain
    for p in model.parameters():
        # glorot initialization needs at least two dimensions on the tensor
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

    # share weights between the embedding layers and the logit layer
    if isinstance(model, Transformer):
        if isinstance(model.encoder.embedding, nn.Embedding):
            nn.init.normal_(model.encoder.embedding.weight, mean=0., std=args.d_model ** -0.5)
            if tie_embeddings:
                model.decoder.embedding.weight = model.encoder.embedding.weight
                model.decoder.classifier.weight = model.decoder.embedding.weight
        elif isinstance(model.encoder.embedding, embedding_mlp.EmbeddingMLP):
            nn.init.normal_(model.encoder.embedding.embedding.weight, mean=0., std=args.d_model ** -0.5)
            model.decoder.embedding.embedding.weight = model.encoder.embedding.embedding.weight

            if tie_embeddings:
                model.decoder.classifier[-1].weight = model.decoder.embedding.embedding.weight
    elif isinstance(model, Decoder):
        if isinstance(model.encoder.embedding, nn.Embedding):
            if tie_embeddings:
                model.classifier.weight = model.embedding.weight
        elif isinstance(model.encoder.embedding, embedding_mlp.EmbeddingMLP):
            if tie_embeddings:
                model.classifier.weight = model.embedding.embedding.weight

    print("Model initialized.")

class EncoderLayer(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(EncoderLayer, self).__init__()

        self.self_attn: multihead_attn.MultiHeadAttention = multihead_attn.MultiHeadAttention(args, device=args.encoder_device, self_attn=True, in_decoder=False, norm=norm)

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_encoder_layers)
            self.fcn_residual = admin_torch.as_module(args.n_encoder_layers)
        else:
            self.self_attn_residual = sum.Sum()
            self.fcn_residual = sum.Sum()

        moe = None
        if args.moe_type == 'millions':
            moe = millions_moe.MillionsMoE(args)
        else:
            self.fcn = positionwise_fcn.PositionWiseFCNetwork(args, norm=norm)

        if moe is not None and bool(args.moe_replace):
            self.fcn = moe
        elif moe is not None:
            self.fcn = nn.Sequential(moe, self.fcn)

    def forward(self, encoder_sequences, key_padding_mask):
        self_attn, _ = self.self_attn(encoder_sequences, encoder_sequences, encoder_sequences, key_padding_mask)

        encoder_sequences = self.self_attn_residual(encoder_sequences, self_attn)
        fcn, gating_variances = self.fcn(encoder_sequences)
        encoder_sequences = self.fcn_residual(encoder_sequences, fcn)
            
        return encoder_sequences, gating_variances

class Encoder(nn.Module):
    def __init__(self, args, vocab_size, norm=nn.LayerNorm):
        super(Encoder, self).__init__()

        self.args = args
        self.vocab_size = vocab_size

        if args.embedding_compression_dim != 0:
            self.embedding = embedding_mlp.EmbeddingMLP(vocab_size, args.embedding_compression_dim, args.d_model, utils.get_activation_function(args.embedding_compression_activation) if args.embedding_compression_activation != 'none' else nn.Identity)
        else:
            self.embedding = nn.Embedding(vocab_size, args.d_model)

        self.apply_dropout = nn.Dropout(args.dropout)
        self.norm = norm(args.d_model, args.norm_eps)
        self.encoder_layers = self.make_encoder_layers(args.n_encoder_layers, args.encoder_param_sharing_type, args.m_encoder_independent_layers, norm=norm)

        if args.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(utils.get_tensor_positional_encoding(args, args.encoder_device))

    def make_encoder_layers(self, n_layers, param_sharing_type, m_independent_layers, norm=nn.LayerNorm) -> List[EncoderLayer]:
        def new_encoder_layer():
            return EncoderLayer(self.args, norm=norm)

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

    def perform_embedding_transformation(self, encoder_sequences : torch.Tensor) -> torch.Tensor:
        return self.embedding(encoder_sequences) * math.sqrt(self.args.d_model) # (N, pad_length, d_model)

    def apply_positional_embedding(self, encoder_sequences):
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return encoder_sequences + self.tensor_positional_encoding[:, :encoder_sequences.size(1), :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences: torch.Tensor, key_padding_mask: torch.Tensor, encoder_layer: nn.Module) -> torch.Tensor:
        return encoder_layer(encoder_sequences, key_padding_mask)

    def forward(self, encoder_sequences: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        assert torch.all(encoder_sequences < self.vocab_size), f"Encoder input is out of bounds: {torch.max(encoder_sequences)} >= {self.vocab_size}"

        encoder_sequences = encoder_sequences.to(self.args.encoder_device)
        key_padding_mask = key_padding_mask.to(self.args.encoder_device)

        encoder_sequences = self.perform_embedding_transformation(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_positional_embedding(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        gating_variances = []
        for encoder_layer in self.encoder_layers:
            encoder_sequences, gating_variance = self.apply_encoder_layer(encoder_sequences, key_padding_mask, encoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        # post-LN
        encoder_sequences = self.norm(encoder_sequences) # (N, pad_length, d_model)

        return encoder_sequences, gating_variances

class DecoderLayer(nn.Module):
    def __init__(self, args, use_cross_attn=True, norm=nn.LayerNorm):
        super(DecoderLayer, self).__init__()

        self.args = args

        self.self_attn = multihead_attn.MultiHeadAttention(args, device=args.decoder_device, self_attn=True, in_decoder=True, norm=norm)

        if use_cross_attn:
            self.cross_attn = multihead_attn.MultiHeadAttention(args, device=args.decoder_device, self_attn=False, in_decoder=True, norm=norm)
        else:
            self.cross_attn = None

        if args.use_admin:
            self.self_attn_residual = admin_torch.as_module(args.n_decoder_layers)
            self.cross_attn_residual = admin_torch.as_module(args.n_decoder_layers)
            self.fcn_residual = admin_torch.as_module(args.n_decoder_layers)
        else:
            self.self_attn_residual = sum.Sum()
            self.cross_attn_residual = sum.Sum()
            self.fcn_residual = sum.Sum()

        moe = None
        if args.moe_type == 'millions':
            moe = millions_moe.MillionsMoE(args)
        else:
            self.fcn = positionwise_fcn.PositionWiseFCNetwork(args, norm=norm)

        if moe is not None and bool(args.moe_replace):
            self.fcn = moe
        elif moe is not None:
            self.fcn = nn.Sequential(moe, self.fcn)

    def forward(self, decoder_sequences: torch.Tensor, encoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        self_attn, _ = self.self_attn(decoder_sequences, decoder_sequences, decoder_sequences, tgt_key_padding_mask)
        decoder_sequences = self.self_attn_residual(decoder_sequences, self_attn)

        if self.cross_attn is not None:
            cross_attn, _ = self.cross_attn(decoder_sequences, encoder_sequences, encoder_sequences, src_key_padding_mask)
            decoder_sequences = self.cross_attn_residual(decoder_sequences, cross_attn)

        fcn, gating_variances = self.fcn(decoder_sequences)
        decoder_sequences = self.fcn_residual(decoder_sequences, fcn)

        return decoder_sequences, gating_variances

class Decoder(nn.Module):
    def __init__(self, args, vocab_size, use_cross_attn=True, norm=nn.LayerNorm):
        super(Decoder, self).__init__()

        self.args = args
        self.vocab_size = vocab_size
        self.use_cross_attn = use_cross_attn

        if args.embedding_compression_dim != 0:
            self.embedding = embedding_mlp.EmbeddingMLP(vocab_size, args.embedding_compression_dim, args.d_model, utils.get_activation_function(args.embedding_compression_activation) if args.embedding_compression_activation != 'none' else nn.Identity)
        else:
            self.embedding = nn.Embedding(vocab_size, args.d_model)
            
        self.apply_dropout = nn.Dropout(args.dropout)
        self.layer_norm = norm(args.d_model, args.norm_eps)
        self.decoder_layers = self.make_decoder_layers(args.n_decoder_layers, args.decoder_param_sharing_type, args.m_decoder_independent_layers, norm=norm)

        if args.embedding_compression_dim != 0:
            self.classifier = nn.Sequential(
                nn.Linear(args.d_model, args.embedding_compression_dim),
                utils.create_activation_function(args.embedding_compression_dim, args.embedding_compression_activation) if args.embedding_compression_activation != 'none' else nn.Identity(),
                nn.Linear(args.embedding_compression_dim, vocab_size)
            )
        else:
            self.classifier = nn.Linear(args.d_model, vocab_size)

        if args.positional_encoding_type != 'rotary':
            self.tensor_positional_encoding = nn.Parameter(utils.get_tensor_positional_encoding(args, args.decoder_device))

    def make_decoder_layers(self, n_layers, param_sharing_type, m_independent_layers, norm=nn.LayerNorm) -> List[DecoderLayer]:
        def new_decoder_layer():
            return DecoderLayer(self.args, use_cross_attn=self.use_cross_attn, norm=norm)
        
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

    def apply_embedding_transformation(self, decoder_sequences: torch.Tensor) -> torch.Tensor:
        return self.embedding(decoder_sequences) * math.sqrt(self.args.d_model) # (N, pad_length, d_model)
    
    def apply_positional_embedding(self, decoder_sequences: torch.Tensor) -> torch.Tensor:
        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if hasattr(self, 'tensor_positional_encoding'):
            return decoder_sequences + self.tensor_positional_encoding[:, :decoder_sequences.size(1), :]
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences: torch.Tensor, encoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor, decoder_layer: nn.Module) -> torch.Tensor:
        return decoder_layer(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask)

    def forward(self, decoder_sequences: torch.Tensor, encoder_sequences: torch.Tensor, src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        assert torch.all(encoder_sequences < self.vocab_size), f"Encoder input is out of bounds: {torch.max(encoder_sequences)} >= {self.vocab_size}"
        assert torch.all(decoder_sequences < self.vocab_size), f"Decoder input is out of bounds: {torch.max(decoder_sequences)} >= {self.vocab_size}"

        decoder_sequences = decoder_sequences.to(self.args.decoder_device)
        encoder_sequences = encoder_sequences.to(self.args.decoder_device)
        src_key_padding_mask = src_key_padding_mask.to(self.args.decoder_device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(self.args.decoder_device)

        decoder_sequences = self.apply_embedding_transformation(decoder_sequences) # (N, pad_length, d_model)

        decoder_sequences = self.apply_positional_embedding(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        gating_variances = []
        for decoder_layer in self.decoder_layers:
            decoder_sequences, gating_variance = self.apply_decoder_layer(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask, decoder_layer)
            if gating_variance is not None:
                gating_variances.append(gating_variance)

        decoder_sequences = self.layer_norm(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.classifier(decoder_sequences) # (N, pad_length, vocab_size)

        return decoder_sequences, gating_variances

class Transformer(nn.Module):
    def __init__(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings=False, padding_value=0, norm=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.args = args
        self.maxlen = args.maxlen
        self.padding_value = padding_value

        self.encoder: Encoder = Encoder(args, src_vocab_size, norm=norm)
        self.decoder: Decoder = Decoder(args, tgt_vocab_size, norm=norm)

        init_weights(args, self, tie_embeddings)

    def forward(self, encoder_sequences, decoder_sequences, src_key_padding_mask, tgt_key_padding_mask):
        encoder_sequences = encoder_sequences.to(self.args.encoder_device)
        src_key_padding_mask = src_key_padding_mask.to(self.args.encoder_device)
        self.encoder.embedding = self.encoder.embedding.to(self.args.encoder_device)

        encoder_sequences, encoder_gating_variances = self.encoder(encoder_sequences, src_key_padding_mask) # (N, encoder_sequence_pad_length, d_model)

        encoder_sequences = encoder_sequences.to(self.args.decoder_device)
        decoder_sequences = decoder_sequences.to(self.args.decoder_device)
        src_key_padding_mask = src_key_padding_mask.to(self.args.decoder_device)
        tgt_key_padding_mask = tgt_key_padding_mask.to(self.args.decoder_device)
        self.decoder.embedding = self.decoder.embedding.to(self.args.decoder_device)
        self.decoder.classifier = self.decoder.classifier.to(self.args.decoder_device)
        
        decoder_sequences, decoder_gating_variances = self.decoder(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask) # (N, decoder_sequence_pad_length, vocab_size)

        return decoder_sequences, encoder_gating_variances, decoder_gating_variances
