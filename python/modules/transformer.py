from .multicast_attn import MultiCastAttention
from .multihead_attn import MultiHeadAttention
from .positionwise_fcn import PositionWiseFCNetwork

import math
import torch
import torch.nn as nn
import utils

class Encoder(nn.Module):
    """
    The Encoder.
    """

    def __init__(self, args, vocab_size, d_model, n_heads, d_queries, d_values, qkv_config, 
                 d_inner, use_moe, n_layers, dropout, encoder_param_sharing_type, m_encoder_independent_layers, 
                 positional_encoding_dim, positional_encoding, activation_function, use_admin, device, 
                 learnable_positional_encoding=False):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for the Encoder
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_layers: number of [multi-head attention + position-wise FC] layers in the Encoder
        :param dropout: dropout probability
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.qkv_config = qkv_config
        self.d_inner = d_inner
        self.use_moe = use_moe
        self.n_layers = n_layers
        self.dropout = dropout
        self.positional_encoding_dim = positional_encoding_dim
        self.positional_encoding = positional_encoding
        self.activation_function = activation_function
        self.use_admin = use_admin
        self.device = device

        # disable gradients for buffer/sinusoidal positional encoding if gradients are not configured to be enabled
        if type(self.positional_encoding) == torch.Tensor:
            self.positional_encoding.requires_grad = learnable_positional_encoding
            self.positional_encoding = self.positional_encoding.to(device)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder_layers = self.make_encoder_layers(args, n_layers, encoder_param_sharing_type, m_encoder_independent_layers)

    def to(self, device):
        """
        Override the default to() method to make sure that the device attribute is also moved to the new device.
        """
        self.device = device
        return super(Encoder, self).to(device)

    def make_encoder_layers(self, args, n_layers, param_sharing_type='none', m_independent_layers=0):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(self.make_encoder_layer(args, i))
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(self.make_encoder_layer(args, i))
                else:
                    layers.append(self.make_encoder_layer(args, i, share_params_with=layers[i - 1]))
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer(args, i))
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(args, i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer(args, i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(args, i, share_params_with=layers[res_idx]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(args, i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer(args, ))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(args, i, share_params_with=[None, layers[res_idx][1]]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(args, i, share_params_with=[None, layers[res_idx][1]]))
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer(args, ))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(args, i, share_params_with=[layers[res_idx][0], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(args, i, share_params_with=[layers[res_idx][0], None]))
            elif param_sharing_type == 'all':
                layers.append(self.make_encoder_layer(args, i, share_params_with=layers[0]))
            else:
                layers.append(self.make_encoder_layer(args, i))
        return nn.ModuleList([nn.ModuleList(enc) for enc in layers])

    def make_encoder_layer(self, args, idx, share_params_with=None):
        """
        Creates a single layer in the Encoder by combining a multi-head attention sublayer and a position-wise FC sublayer.
        """

        def multi_head_or_cast_attn():
            return MultiHeadAttention(
                n_layers=args.n_encoder_layers,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_queries=self.d_queries,
                d_values=self.d_values,
                qkv_config=self.qkv_config,
                dropout=self.dropout,
                use_admin=self.use_admin,
                device=self.device,
                positional_encoding_dim=self.positional_encoding_dim,
                d_output=self.d_model,
                positional_encoding=self.positional_encoding,
                in_decoder=False
            ) if args.encoder_layer_self_attn_config is None else MultiCastAttention(
                d_model=self.d_model,
                d_queries=self.d_queries,
                d_values=self.d_values,
                qkv_config=self.qkv_config,
                dropout=self.dropout,
                use_admin=self.use_admin,
                attn_config=args.encoder_layer_self_attn_config,
                device=self.device,
                positional_encoding_dim=self.positional_encoding_dim,
                positional_encoding=self.positional_encoding,
                in_decoder=False,
                sequential=False
            )
        
        attn_layers = share_params_with[0] if share_params_with is not None and share_params_with[0] is not None else multi_head_or_cast_attn()
        ffn = share_params_with[1] if share_params_with is not None and share_params_with[1] is not None else PositionWiseFCNetwork(
            use_moe=self.use_moe,
            n_experts=args.n_experts,
            top_k=args.moe_top_k,
            n_layers=args.n_encoder_layers,
            d_model=self.d_model,
            d_inner=self.d_inner,
            activation_function=self.activation_function, 
            dropout=self.dropout,
            use_admin=self.use_admin,
            device=self.device,
            in_decoder=True
        )

        return [attn_layers, ffn]

    def perform_embedding_transformation(self, encoder_sequences):
        d_model = self.embedding.weight.size(1)
        return self.embedding(encoder_sequences) * math.sqrt(d_model) # (N, pad_length, d_model)

    def apply_positional_embedding(self, encoder_sequences):
        pad_length = encoder_sequences.size(1)  # pad-length of this batch only, varies across batches

        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if type(self.positional_encoding) == torch.Tensor and len(self.positional_encoding.shape) == 3:
            encoder_sequences += self.positional_encoding[:, :pad_length, :]
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences, encoder_sequence_lengths, encoder_layer):
        encoder_sequences, _, _ = encoder_layer[0](query_sequences=encoder_sequences, key_sequences=encoder_sequences, value_sequences=encoder_sequences, key_value_sequence_lengths=encoder_sequence_lengths) # (N, pad_length, d_model)
        encoder_sequences = encoder_layer[1](sequences=encoder_sequences) # (N, pad_length, d_model)
        return encoder_sequences

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        """
        Forward prop.

        :param encoder_sequences: the source language sequences, a tensor of size (N, pad_length)
        :param encoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :return: encoded source language sequences, a tensor of size (N, pad_length, d_model)
        """
        encoder_sequences = self.perform_embedding_transformation(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_positional_embedding(encoder_sequences) # (N, pad_length, d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        for encoder_layer in self.encoder_layers:
            encoder_sequences = self.apply_encoder_layer(encoder_sequences, encoder_sequence_lengths, encoder_layer)

        # post-LN
        encoder_sequences = self.layer_norm(encoder_sequences) # (N, pad_length, d_model)

        return encoder_sequences

class Decoder(nn.Module):
    """
    The Decoder.
    """

    def __init__(self, args, vocab_size, d_model, n_heads, d_queries, d_values, qkv_config, 
                 d_inner, use_moe, n_layers, dropout, decoder_param_sharing_type, m_decoder_independent_layers, 
                 positional_encoding_dim, positional_encoding, activation_function, use_admin, device, 
                 learnable_positional_encoding=False):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for the Encoder
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_layers: number of [multi-head attention + position-wise FC] layers in the Decoder
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.qkv_config = qkv_config
        self.d_inner = d_inner
        self.use_moe = use_moe
        self.n_layers = n_layers
        self.dropout = dropout
        self.positional_encoding_dim = positional_encoding_dim
        self.positional_encoding = positional_encoding
        self.activation_function = activation_function
        self.use_admin = use_admin
        self.device = device

        # disable gradients for buffer/sinusoidal positional encoding if gradients are not configured to be enabled
        if type(self.positional_encoding) == torch.Tensor:
            self.positional_encoding.requires_grad = learnable_positional_encoding
            self.positional_encoding = self.positional_encoding.to(device)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder_layers = self.make_decoder_layers(args, n_layers, decoder_param_sharing_type, m_decoder_independent_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def to(self, device):
        """
        Override the default to() method to make sure that the device attribute is also moved to the new device.
        """
        self.device = device
        return super(Encoder, self).to(device)

    def make_decoder_layers(self, args, n_layers, param_sharing_type='none', m_independent_layers=0):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(self.make_decoder_layer(args, i))
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(self.make_decoder_layer(args, i))
                else:
                    layers.append(self.make_decoder_layer(args, i, share_params_with=layers[i - 1]))
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(args, i))
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(args, i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(args, i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(args, i, share_params_with=layers[res_idx]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(args, i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(args, i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(args, i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(args, i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(args, i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(args, i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(args, i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
            elif param_sharing_type == 'all':
                layers.append(self.make_decoder_layer(args, i, share_params_with=layers[0]))
            else:
                layers.append(self.make_decoder_layer(args, i))
        return nn.ModuleList([nn.ModuleList(dec) for dec in layers])

    def make_decoder_layer(self, args, idx, share_params_with=None):
        """
        Creates a single layer in the Decoder by combining two multi-head attention sublayers and a position-wise FC sublayer.
        """

        def multi_head_or_cast_attn():
            return MultiHeadAttention(
                n_layers=args.n_decoder_layers,
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_queries=self.d_queries,
                d_values=self.d_values,
                qkv_config=self.qkv_config,
                dropout=self.dropout,
                use_admin=self.use_admin,
                device=self.device,
                positional_encoding_dim=self.positional_encoding_dim,
                d_output=self.d_model,
                positional_encoding=self.positional_encoding,
                in_decoder=True
            ) if args.decoder_layer_self_attn_config is None else MultiCastAttention(
                d_model=self.d_model,
                d_queries=self.d_queries,
                d_values=self.d_values,
                qkv_config=self.qkv_config,
                dropout=self.dropout,
                use_admin=self.use_admin,
                attn_config=args.decoder_layer_self_attn_config,
                device=self.device,
                positional_encoding_dim=self.positional_encoding_dim,
                positional_encoding=self.positional_encoding,
                in_decoder=True,
                sequential=False
            )
        
        self_attn = share_params_with[0] if share_params_with is not None and share_params_with[0] is not None else multi_head_or_cast_attn()
        cross_attn = share_params_with[1] if share_params_with is not None and share_params_with[1] is not None else MultiHeadAttention(
            n_layers=args.n_decoder_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            qkv_config=self.qkv_config,
            dropout=self.dropout,
            use_admin=self.use_admin,
            device=self.device,
            positional_encoding_dim=self.positional_encoding_dim,
            d_output=self.d_model,
            positional_encoding=self.positional_encoding,
            in_decoder=True
        )
        ffn = share_params_with[2] if share_params_with is not None and share_params_with[2] is not None else PositionWiseFCNetwork(
            use_moe=self.use_moe,
            n_experts=args.n_experts,
            top_k=args.moe_top_k,
            n_layers=args.n_decoder_layers,
            d_model=self.d_model,
            d_inner=self.d_inner,
            activation_function=self.activation_function, 
            dropout=self.dropout,
            use_admin=self.use_admin,
            device=self.device,
            in_decoder=True
        )

        return [self_attn, cross_attn, ffn]
    
    def apply_embedding_transformation(self, decoder_sequences):
        d_model = self.embedding.weight.size(1)
        return self.embedding(decoder_sequences) * math.sqrt(d_model) # (N, pad_length, d_model)
    
    def apply_positional_embedding(self, decoder_sequences):
        pad_length = decoder_sequences.size(1)  # pad-length of this batch only, varies across batches
        if type(self.positional_encoding) == torch.Tensor and len(self.positional_encoding.shape) == 3:
            decoder_sequences += self.positional_encoding[:, :pad_length, :]
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer):
        decoder_sequences, _, _ = decoder_layer[0](query_sequences=decoder_sequences, key_sequences=decoder_sequences, value_sequences=decoder_sequences, key_value_sequence_lengths=decoder_sequence_lengths) # (N, pad_length, d_model), trash attention_weights
        decoder_sequences, _, _ = decoder_layer[1](query_sequences=decoder_sequences, key_sequences=encoder_sequences, value_sequences=encoder_sequences, key_value_sequence_lengths=encoder_sequence_lengths) # (N, pad_length, d_model)
        decoder_sequences = decoder_layer[2](sequences=decoder_sequences) # (N, pad_length, d_model)
        return decoder_sequences

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        """
        Forward prop.

        :param decoder_sequences: the source language sequences, a tensor of size (N, pad_length)
        :param decoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :param encoder_sequences: encoded source language sequences, a tensor of size (N, encoder_pad_length, d_model)
        :param encoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :return: decoded target language sequences, a tensor of size (N, pad_length, vocab_size)
        """
        decoder_sequences = self.apply_embedding_transformation(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_positional_embedding(decoder_sequences) # (N, pad_length, d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        for decoder_layer in self.decoder_layers:
            decoder_sequences = self.apply_decoder_layer(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer)

        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)
        decoder_sequences = self.classifier(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences

class Transformer(nn.Module):
    """
    The Transformer network.
    """

    def __init__(self, args, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_queries, d_values, 
                 qkv_config, d_inner, use_moe, n_encoder_layers, n_decoder_layers, dropout, 
                 encoder_param_sharing_type, decoder_param_sharing_type, m_encoder_independent_layers, 
                 m_decoder_independent_layers, positional_encoding_dim, positional_encoding, activation_function, 
                 init_weights_from,  init_weights_gain, use_admin, device,  learnable_positional_encoding):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_encoder_layers: number of layers in the Encoder
        :param n_decoder_layers: number of layers in the Decoder
        :param dropout: dropout probability
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.init_weights_from = init_weights_from
        self.init_weights_gain = init_weights_gain
        self.use_admin = use_admin

        self.encoder = Encoder(
            args=args,
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_queries=d_queries,
            d_values=d_values,
            qkv_config=qkv_config,
            d_inner=d_inner,
            use_moe=use_moe,
            n_layers=n_encoder_layers,
            dropout=dropout,
            encoder_param_sharing_type=encoder_param_sharing_type,
            m_encoder_independent_layers=m_encoder_independent_layers,
            positional_encoding_dim=positional_encoding_dim,
            positional_encoding=positional_encoding,
            activation_function=activation_function,
            use_admin=use_admin,
            device=device,
            learnable_positional_encoding=learnable_positional_encoding
        )
        self.decoder = Decoder(
            args=args,
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_queries=d_queries,
            d_values=d_values,
            qkv_config=qkv_config,
            d_inner=d_inner,
            use_moe=use_moe,
            n_layers=n_decoder_layers,
            dropout=dropout,
            decoder_param_sharing_type=decoder_param_sharing_type,
            m_decoder_independent_layers=m_decoder_independent_layers,
            positional_encoding_dim=positional_encoding_dim,
            positional_encoding=positional_encoding,
            activation_function=activation_function,
            use_admin=use_admin,
            device=device,
            learnable_positional_encoding=learnable_positional_encoding
        )

    def init_weights(self, tie_embeddings=True):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of self.init_weights_gain
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                if self.init_weights_from in ['glorot_uniform', 'xavier_uniform']:
                    nn.init.xavier_uniform_(p, gain=self.init_weights_gain)
                elif self.init_weights_from in ['glorot_normal', 'xavier_normal']:
                    nn.init.xavier_normal_(p, gain=self.init_weights_gain)
                elif self.init_weights_from == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(p)
                elif self.init_weights_from == 'kaiming_normal':
                    nn.init.kaiming_normal_(p)
                elif self.init_weights_from == 'orthogonal':
                    nn.init.orthogonal_(p)
                else:
                    raise Exception(f"Unknown weight initialization method: {self.init_weights_from}")

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.classifier.weight = self.decoder.embedding.weight
        print("Model initialized.")

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        """
        Forward propagation.

        :param encoder_sequences: source language sequences, a tensor of size (N, encoder_sequence_pad_length)
        :param decoder_sequences: target language sequences, a tensor of size (N, decoder_sequence_pad_length)
        :param encoder_sequence_lengths: true lengths of source language sequences, a tensor of size (N)
        :param decoder_sequence_lengths: true lengths of target language sequences, a tensor of size (N)
        :return: decoded target language sequences, a tensor of size (N, decoder_sequence_pad_length, vocab_size)
        """
        encoder_sequences = self.encoder(encoder_sequences, encoder_sequence_lengths) # (N, encoder_sequence_pad_length, d_model)
        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths) # (N, decoder_sequence_pad_length, vocab_size)
        return decoder_sequences
