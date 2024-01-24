from .multicast_attn import MultiCastAttention
from .multihead_attn import MultiHeadAttention
from .positionwise_fcn import PositionWiseFCNetwork
from rotary_embedding_torch import RotaryEmbedding

import math
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    The Encoder.
    """

    def __init__(self, args, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers, dropout):
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

        self.args = args

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, d_model)

        # disable gradients for buffer/sinusoidal positional encoding if gradients are not configured to be enabled
        if type(self.positional_encoding) == torch.Tensor:
            self.positional_encoding.requires_grad = args.learnable_positional_encoding

        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder_layers = self.make_encoder_layers(n_layers, args.encoder_param_sharing_type, args.m_encoder_independent_layers)

    def make_encoder_layers(self, n_layers, param_sharing_type='none', m_independent_layers=0):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(self.make_encoder_layer(i))
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(self.make_encoder_layer(i))
                else:
                    layers.append(self.make_encoder_layer(i, share_params_with=layers[i - 1]))
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer(i))
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer(i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(i, share_params_with=layers[res_idx]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(i, share_params_with=[None, layers[res_idx][1]]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(i, share_params_with=[None, layers[res_idx][1]]))
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(i, share_params_with=[layers[res_idx][0], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(i, share_params_with=[layers[res_idx][0], None]))
            elif param_sharing_type == 'all':
                layers.append(self.make_encoder_layer(i, share_params_with=layers[0]))
            else:
                layers.append(self.make_encoder_layer(i))
        return nn.ModuleList([nn.ModuleList(enc) for enc in layers])

    def make_encoder_layer(self, idx, share_params_with=None):
        """
        Creates a single layer in the Encoder by combining a multi-head attention sublayer and a position-wise FC sublayer.
        """

        attn_layers = share_params_with[0] if share_params_with is not None and share_params_with[0] is not None else MultiHeadAttention(
            args=self.args,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            dropout=self.dropout,
            positional_encoding=self.positional_encoding,
            in_decoder=False
        )
        ffn = share_params_with[1] if share_params_with is not None and share_params_with[1] is not None else PositionWiseFCNetwork(
            args=self.args,
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout
        )

        layers = [attn_layers, ffn]

        if self.args.use_admin:
            layers.append(nn.Linear(self.d_model, self.d_model)) # add residual connection with learned parameters

        return layers

    def perform_embedding_transformation(self, encoder_sequences):
        return self.embedding(encoder_sequences) * math.sqrt(self.d_model) # (N, pad_length, d_model)

    def apply_positional_embedding(self, encoder_sequences):
        pad_length = encoder_sequences.size(1)  # pad-length of this batch only, varies across batches

        # 1D buffer/sinusoidal encoding is applied here. 2D buffer/sinusoidal encoding and rotary encoding are applied in the MultiHeadAttention layer(s)
        if type(self.positional_encoding) == torch.Tensor and len(self.positional_encoding.shape) == 3:
            encoder_sequences += self.positional_encoding[:, :pad_length, :].to(self.args.device)
        return encoder_sequences
    
    def apply_encoder_layer(self, encoder_sequences, encoder_sequence_lengths, encoder_layer):
        encoder_sequences, _ = encoder_layer[0](query_sequences=encoder_sequences, key_sequences=encoder_sequences, value_sequences=encoder_sequences, key_value_sequence_lengths=encoder_sequence_lengths) # (N, pad_length, d_model)
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

    def __init__(self, args, vocab_size, positional_encoding, d_model, n_heads, d_queries, d_values, d_inner, n_layers, dropout):
        """
        :param vocab_size: size of the (shared) vocabulary
        :param positional_encoding: positional encodings up to the maximum possible pad-length
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for the Decoder
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors) in the multi-head attention
        :param d_values: size of value vectors in the multi-head attention
        :param d_inner: an intermediate size in the position-wise FC
        :param n_layers: number of [multi-head attention + multi-head attention + position-wise FC] layers in the Decoder
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__()

        self.args = args

        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, d_model)

        # disable gradients for buffer/sinusoidal positional encoding if gradients are not configured to be enabled
        if type(self.positional_encoding) == torch.Tensor:
            self.positional_encoding.requires_grad = args.learnable_positional_encoding

        self.apply_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder_layers = self.make_decoder_layers(n_layers, args.decoder_param_sharing_type, args.m_decoder_independent_layers)

        # classification layer for output language vocabulary
        self.fc = nn.Linear(d_model, vocab_size)

    def make_decoder_layers(self, n_layers, param_sharing_type='none', m_independent_layers=0):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(self.make_decoder_layer(i))
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(self.make_decoder_layer(i))
                else:
                    layers.append(self.make_decoder_layer(i, share_params_with=layers[i - 1]))
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(i))
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(i, share_params_with=layers[res_idx]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(i, share_params_with=layers[res_idx]))
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer(i))
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(i, share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
            elif param_sharing_type == 'all':
                layers.append(self.make_decoder_layer(i, share_params_with=layers[0]))
            else:
                layers.append(self.make_decoder_layer(i))
        return nn.ModuleList([nn.ModuleList(dec) for dec in layers])

    def make_decoder_layer(self, idx, share_params_with=None):
        """
        Creates a single layer in the Decoder by combining two multi-head attention sublayers and a position-wise FC sublayer.
        """

        self_attn = share_params_with[0] if share_params_with is not None and share_params_with[0] is not None else MultiHeadAttention(
            args=self.args,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            dropout=self.dropout,
            positional_encoding=self.positional_encoding,
            in_decoder=True
        )
        cross_attn = share_params_with[1] if share_params_with is not None and share_params_with[1] is not None else MultiHeadAttention(
            args=self.args,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            dropout=self.dropout,
            positional_encoding=self.positional_encoding,
            in_decoder=True
        )
        ffn = share_params_with[2] if share_params_with is not None and share_params_with[2] is not None else PositionWiseFCNetwork(
            args=self.args,
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout
        )

        layers = [self_attn, cross_attn, ffn]

        if self.args.use_admin:
            layers.append(nn.Linear(self.d_model, self.d_model)) # add residual connection with learned parameters

        return layers
    
    def apply_embedding_transformation(self, decoder_sequences):
        return self.embedding(decoder_sequences) * math.sqrt(self.d_model) # (N, pad_length, d_model)
    
    def apply_positional_embedding(self, decoder_sequences):
        pad_length = decoder_sequences.size(1)  # pad-length of this batch only, varies across batches
        if type(self.positional_encoding) == torch.Tensor and len(self.positional_encoding.shape) == 3:
            decoder_sequences += self.positional_encoding[:, :pad_length, :].to(self.args.device)
        return decoder_sequences
    
    def apply_decoder_layer(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths, decoder_layer):
        decoder_sequences, _ = decoder_layer[0](query_sequences=decoder_sequences, key_sequences=decoder_sequences, value_sequences=decoder_sequences, key_value_sequence_lengths=decoder_sequence_lengths) # (N, pad_length, d_model), trash attention_weights
        decoder_sequences, _ = decoder_layer[1](query_sequences=decoder_sequences, key_sequences=encoder_sequences, value_sequences=encoder_sequences, key_value_sequence_lengths=encoder_sequence_lengths) # (N, pad_length, d_model)
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
        decoder_sequences = self.fc(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences

class Transformer(nn.Module):
    """
    The Transformer network.
    """

    def __init__(self, args, src_vocab_size, tgt_vocab_size, positional_encoding, tie_embeddings=True, d_model=512, n_heads=8, d_queries=64, d_values=64, d_inner=2048, n_encoder_layers=6, n_decoder_layers=6, dropout=0.1, use_admin=False):
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

        self.args = args

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.positional_encoding = positional_encoding
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.use_admin = use_admin

        # Encoder
        self.encoder = Encoder(
            args=args,
            vocab_size=src_vocab_size,
            positional_encoding=positional_encoding,
            d_model=d_model,
            n_heads=n_heads,
            d_queries=d_queries,
            d_values=d_values,
            d_inner=d_inner,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # Decoder
        self.decoder = Decoder(
            args=args,
            vocab_size=tgt_vocab_size,
            positional_encoding=positional_encoding,
            d_model=d_model,
            n_heads=n_heads,
            d_queries=d_queries,
            d_values=d_values,
            d_inner=d_inner,
            n_layers=n_decoder_layers,
            dropout=dropout,
        )

        # Initialize weights
        self.init_weights(tie_embeddings=tie_embeddings)

    def init_weights(self, tie_embeddings=True):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of self.args.init_weights_gain
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
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.fc.weight = self.decoder.embedding.weight

        if self.args.use_admin:
            for encoder_layer in self.encoder.encoder_layers:
                nn.init.ones_(encoder_layer[-1].weight)
            for decoder_layer in self.decoder.decoder_layers:
                nn.init.ones_(decoder_layer[-1].weight)

            input_sequence = torch.randn(1, 1, self.d_model)

            input_sequence = self.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
            input_sequence = self.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)
            # input_sequence = self.encoder.apply_dropout(input_sequence) # (N, pad_length, d_model) # don't apply dropout here

            # todo: need to fix so that each successive variance is calculated using the newly calculated variance weights, not the previous values that uses all ones

            variance_so_far = 0
            for encoder_layer in self.encoder.encoder_layers:
                input_sequence = self.encoder.apply_encoder_layer(input_sequence, 1, encoder_layer)
                variance_so_far += torch.var(input_sequence)

                encoder_layer[-1].weight.data = torch.fill_(encoder_layer[-1].weight.data, variance_so_far.item())

            input_sequence = self.encoder.layer_norm(input_sequence) # (N, pad_length, d_model)

            for decoder_layer in self.decoder.decoder_layers:
                input_sequence = self.decoder.apply_decoder_layer(torch.randn(1, 1, self.d_model), 1, input_sequence, 1, decoder_layer)
                variance_so_far += torch.var(input_sequence)

                decoder_layer[-1].weight.data = torch.fill_(decoder_layer[-1].weight.data, variance_so_far.item())

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
        # Encoder
        encoder_sequences = self.encoder(encoder_sequences, encoder_sequence_lengths) # (N, encoder_sequence_pad_length, d_model)

        # Decoder
        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths) # (N, decoder_sequence_pad_length, vocab_size)

        return decoder_sequences
