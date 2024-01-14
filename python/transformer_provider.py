from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from rotary_embedding_torch import RotaryEmbedding

import math
import torch
import torch.nn.functional as F
import utils

class NewTransformerModelProvider:
    def __init__(self):
        super().__init__()

    def provide(self, args, src_vocab_size, tgt_vocab_size, tie_embeddings, positional_encoding):
        return Transformer(
            args=args,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            tie_embeddings=tie_embeddings,
            positional_encoding=positional_encoding,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_queries=args.d_queries,
            d_values=args.d_values,
            d_inner=args.d_inner,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            dropout=args.dropout
        )

class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.
    """

    def __init__(self, args, d_model, n_heads, d_queries, d_values, dropout, positional_encoding=None, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors)
        :param d_values: size of value vectors
        :param dropout: dropout probability
        :param in_decoder: is this Multi-Head Attention sublayer instance in the decoder?
        """
        super(MultiHeadAttention, self).__init__()

        self.args=args

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries # size of key vectors, same as of the query vectors to allow dot-products for similarity

        self.positional_encoding = positional_encoding
        self.in_decoder = in_decoder

        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(d_model, n_heads * self.d_keys)
        self.cast_values = nn.Linear(d_model, n_heads * self.d_values)

        # A linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors (of the same size as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        """
        Forward prop.

        :param query_sequences: the input query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        :param key_value_sequences: the sequences to be queried against, a tensor of size (N, key_value_sequence_pad_length, d_model)
        :param key_value_sequence_lengths: true lengths of the key_value_sequences, to be able to ignore pads, a tensor of size (N)
        :return: attention-weighted output sequences for the query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        """
        batch_size = query_sequences.size(0) # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_value_sequences, query_sequences)

        # Store input for adding later
        input_to_add = query_sequences.clone()

        # Apply layer normalization
        query_sequences = self.layer_norm(query_sequences) # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value sequences (as they are the same as the query sequences)
        # If this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self_attention:
            key_value_sequences = self.layer_norm(key_value_sequences) # (N, key_value_sequence_pad_length, d_model)

        # Project input sequences to queries, keys, values
        queries = self.cast_queries(query_sequences) # (N, query_sequence_pad_length, n_heads * d_queries)
        keys = self.cast_keys(key_sequences) # (N, key_value_sequence_pad_length, n_heads * d_keys)
        values = self.cast_values(value_sequences) # (N, key_value_sequence_pad_length, n_heads * d_values)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.n_heads, self.d_queries) # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads, self.d_keys) # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads, self.d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # RoPE is applied to the queries and keys after the heads are split out but before the dot product for attention and subsequent softmax operations
        if self.positional_encoding is not None and type(self.positional_encoding) == RotaryEmbedding:
            # queries and keys are of shape (N, n_heads, seq len, d_queries/d_keys)
            queries = self.positional_encoding.rotate_queries_or_keys(queries)
            keys = self.positional_encoding.rotate_queries_or_keys(keys)

        # And then, for convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
        queries = queries.contiguous().view(-1, query_sequence_pad_length, self.d_queries) # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = keys.contiguous().view(-1, key_value_sequence_pad_length, self.d_keys) # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = values.contiguous().view(-1, key_value_sequence_pad_length, self.d_values) # (N * n_heads, key_value_sequence_pad_length, d_values)

        # Perform multi-head attention

        # Perform dot-products
        attention_weights = torch.bmm(queries, keys.permute(0, 2, 1)) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (1. / math.sqrt(self.d_keys)) * attention_weights # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(self.args.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # print(f"not_pad_in_keys: {not_pad_in_keys.shape}")
        # print(f"attention_weights: {attention_weights.shape}")
        # print(f"key_value_sequence_lengths: {key_value_sequence_lengths.repeat_interleave(self.n_heads).shape}")
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.n_heads).unsqueeze(1).unsqueeze(2).expand_as(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the decoder, keys chronologically ahead of queries
        if self.in_decoder and self_attention:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(self.args.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Compute softmax along the key dimension
        attention_weights = self.softmax(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Apply dropout
        attention_weights = self.apply_dropout(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values) # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original order of axes
        sequences = sequences.contiguous().view(batch_size, self.n_heads, query_sequence_pad_length, self.d_values).permute(0, 2, 1, 3) # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1) # (N, query_sequence_pad_length, n_heads * d_values)

        # Transform the concatenated subspace-sequences into a single output of size d_model
        sequences = self.cast_output(sequences) # (N, query_sequence_pad_length, d_model)

        # Apply dropout and residual connection
        sequences = self.apply_dropout(sequences) + input_to_add # (N, query_sequence_pad_length, d_model)

        return sequences

class MultiCastAttention(nn.Module):
    def __init__(self, args, attn_config, d_queries, d_values, dropout, positional_encoding=None, in_decoder=False, sequential=False):
        super(MultiCastAttention, self).__init__()

        self.args = args

        self.d_queries = args.d_queries
        self.d_values = args.d_values
        self.dropout = args.dropout

        self.positional_encoding = positional_encoding
        self.in_decoder = in_decoder
        self.sequential = sequential

        self.layers, self.embed_dim_list = nn.ModuleList(self.build_layers_from_attn_config(attn_config))

    def build_layers_from_attn_config(self, attn_config):
        layer_configs = attn_config.split(',')
        layers = []
        for i, layer_config in enumerate(layer_configs):
            layer_config_parts = layer_config.split(':')
            layer_type = layer_config_parts[0]
            layer_output_dim = int(layer_config_parts[1])
            layer_n_heads = int(layer_config_parts[2])
            if layer_type == 'MultiHeadAttention':
                layers.append(MultiHeadAttention(
                    args=self.args,
                    d_model=layer_output_dim,
                    n_heads=layer_n_heads,
                    d_queries=self.d_queries,
                    d_values=self.d_values,
                    dropout=self.dropout,
                    positional_encoding=self.positional_encoding,
                    in_decoder=self.in_decoder
                ))
            else:
                raise Exception(f"Unknown attention layer type: {layer_type}")
        return layers
    
    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        if self.sequential:
            for layer in self.layers:
                query_sequences = layer(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)
            return query_sequences
        else:
            layer_outputs = []
            for layer in self.layers:
                layer_outputs.append(layer(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths))
            return torch.cat(layer_outputs, dim=-1)

class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, args, d_model, d_inner, dropout):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.d_model = d_model
        self.d_inner = d_inner

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # A linear layer to project from the input size to an intermediate size
        self.fc1 = nn.Linear(d_model, d_inner)

        self.activation = utils.create_activation_function(args.activation_function)

        # A linear layer to project from the intermediate size to the output size (same as the input size)
        self.fc2 = nn.Linear(d_inner, d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward prop.

        :param sequences: input sequences, a tensor of size (N, pad_length, d_model)
        :return: transformed output sequences, a tensor of size (N, pad_length, d_model)
        """
        # Store input for adding later
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        # Apply layer-norm
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Transform position-wise
        sequences = self.apply_dropout(self.activation(self.fc1(sequences)))  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)

        # Apply dropout and residual connection
        sequences = self.apply_dropout(sequences) + input_to_add  # (N, pad_length, d_model)

        return sequences

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

        # An embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Set the positional encoding tensor to be un-update-able, i.e. gradients aren't computed
        if self.positional_encoding is not None and type(self.positional_encoding) != RotaryEmbedding:
            self.positional_encoding.requires_grad = False

        # Encoder layers
        self.encoder_layers = self.make_encoder_layers(n_layers, args.encoder_param_sharing_type, args.m_encoder_independent_layers)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layers(self, n_layers, param_sharing_type='none', m_independent_layers=0):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(self.make_encoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(self.make_encoder_layer())
                else:
                    layers.append(self.make_encoder_layer(share_params_with=layers[i - 1]))
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(share_params_with=layers[res_idx]))
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(share_params_with=layers[res_idx]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(share_params_with=layers[res_idx]))
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(share_params_with=[None, layers[res_idx][1]]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(share_params_with=[None, layers[res_idx][1]]))
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_encoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_encoder_layer(share_params_with=[layers[res_idx][0], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_encoder_layer(share_params_with=[layers[res_idx][0], None]))
            elif param_sharing_type == 'all':
                layers.append(self.make_encoder_layer(share_params_with=layers[0]))
            else:
                layers.append(self.make_encoder_layer())
        return nn.ModuleList([nn.ModuleList(enc) for enc in layers])

    def make_encoder_layer(self, share_params_with=None):
        """
        Creates a single layer in the Encoder by combining a multi-head attention sublayer and a position-wise FC sublayer.
        """

        attn_layers = share_params_with[0] if share_params_with is not None and share_params_with[0] is not None else MultiCastAttention(
            self.args,
            self.args.encoder_layer_attn_config,
            self.args.d_queries,
            self.args.d_values,
            self.args.dropout,
            self.positional_encoding,
            in_decoder=False
        )
        ffn = share_params_with[1] if share_params_with is not None and share_params_with[1] is not None else PositionWiseFCNetwork(
            args=self.args,
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout
        )

        return [attn_layers, ffn]

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        """
        Forward prop.

        :param encoder_sequences: the source language sequences, a tensor of size (N, pad_length)
        :param encoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :return: encoded source language sequences, a tensor of size (N, pad_length, d_model)
        """
        pad_length = encoder_sequences.size(1)  # pad-length of this batch only, varies across batches

        # Sum vocab embeddings and position embeddings
        encoder_sequences = self.embedding(encoder_sequences) * math.sqrt(self.d_model) # (N, pad_length, d_model)
        if self.positional_encoding is not None and type(self.positional_encoding) != RotaryEmbedding:
            encoder_sequences += self.positional_encoding[:, :pad_length, :].to(self.args.device)

        # Dropout
        encoder_sequences = self.apply_dropout(encoder_sequences) # (N, pad_length, d_model)

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            # Sublayers
            encoder_sequences = encoder_layer[0](query_sequences=encoder_sequences, key_sequences=encoder_sequences, value_sequences=encoder_sequences, key_value_sequence_lengths=encoder_sequence_lengths) # (N, pad_length, d_model)
            encoder_sequences = encoder_layer[1](sequences=encoder_sequences) # (N, pad_length, d_model)

        # Apply layer-norm
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

        # An embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Set the positional encoding tensor to be un-update-able, i.e. gradients aren't computed
        if self.positional_encoding is not None and type(self.positional_encoding) != RotaryEmbedding:
            self.positional_encoding.requires_grad = False

        # Decoder layers
        self.decoder_layers = self.make_decoder_layers(n_layers, args.decoder_param_sharing_type, args.m_decoder_independent_layers)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Output linear layer that will compute logits for the vocabulary
        self.fc = nn.Linear(d_model, vocab_size)

    def make_decoder_layers(self, n_layers, param_sharing_type='none', m_independent_layers=0):
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(self.make_decoder_layer())
            elif param_sharing_type == 'sequence':
                if (i - 1) % math.floor(n_layers / m_independent_layers) == 0:
                    layers.append(self.make_decoder_layer())
                else:
                    layers.append(self.make_decoder_layer(share_params_with=layers[i - 1]))
            elif param_sharing_type == 'cycle':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer())
                else:
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(share_params_with=layers[res_idx]))
            elif param_sharing_type == 'cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(share_params_with=layers[res_idx]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(share_params_with=layers[res_idx]))
            elif param_sharing_type == 'ffn-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
            elif param_sharing_type == 'heads-cycle-rev':
                if i <= m_independent_layers:
                    layers.append(self.make_decoder_layer())
                elif i <= m_independent_layers * (math.ceil(n_layers / m_independent_layers) - 1):
                    res_idx = ((i - 1) % m_independent_layers) + 1
                    layers.append(self.make_decoder_layer(share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
                else:
                    res_idx = m_independent_layers - ((i - 1) % m_independent_layers)
                    layers.append(self.make_decoder_layer(share_params_with=[layers[res_idx][0], layers[res_idx][1], None]))
            elif param_sharing_type == 'all':
                layers.append(self.make_decoder_layer(share_params_with=layers[0]))
            else:
                layers.append(self.make_decoder_layer())
        return nn.ModuleList([nn.ModuleList(dec) for dec in layers])

    def make_decoder_layer(self, share_params_with=None):
        """
        Creates a single layer in the Decoder by combining two multi-head attention sublayers and a position-wise FC sublayer.
        """

        self_attn = share_params_with[0] if share_params_with is not None and share_params_with[0] is not None else MultiCastAttention(
            args=self.args,
            attn_config=self.args.decoder_layer_self_attn_config,
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

        return [self_attn, cross_attn, ffn]

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        """
        Forward prop.

        :param decoder_sequences: the source language sequences, a tensor of size (N, pad_length)
        :param decoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :param encoder_sequences: encoded source language sequences, a tensor of size (N, encoder_pad_length, d_model)
        :param encoder_sequence_lengths: true lengths of these sequences, a tensor of size (N)
        :return: decoded target language sequences, a tensor of size (N, pad_length, vocab_size)
        """
        pad_length = decoder_sequences.size(1)  # pad-length of this batch only, varies across batches

        # Sum vocab embeddings and position embeddings
        decoder_sequences = self.embedding(decoder_sequences) * math.sqrt(self.d_model) # (N, pad_length, d_model)
        if self.positional_encoding is not None and type(self.positional_encoding) != RotaryEmbedding:
            decoder_sequences += self.positional_encoding[:, :pad_length, :].to(self.args.device)

        # Dropout
        decoder_sequences = self.apply_dropout(decoder_sequences)

        # Decoder layers
        for decoder_layer in self.decoder_layers:
            # Sublayers
            decoder_sequences = decoder_layer[0](query_sequences=decoder_sequences, key_sequences=decoder_sequences, value_sequences=decoder_sequences, key_value_sequence_lengths=decoder_sequence_lengths) # (N, pad_length, d_model)
            decoder_sequences = decoder_layer[1](query_sequences=decoder_sequences, key_sequences=encoder_sequences, value_sequences=encoder_sequences, key_value_sequence_lengths=encoder_sequence_lengths) # (N, pad_length, d_model)
            decoder_sequences = decoder_layer[2](sequences=decoder_sequences) # (N, pad_length, d_model)

        # Apply layer-norm
        decoder_sequences = self.layer_norm(decoder_sequences)  # (N, pad_length, d_model)

        # Find logits over vocabulary
        decoder_sequences = self.fc(decoder_sequences)  # (N, pad_length, vocab_size)

        return decoder_sequences

class Transformer(nn.Module):
    """
    The Transformer network.
    """

    def __init__(self, args, src_vocab_size, tgt_vocab_size, positional_encoding, tie_embeddings=True, d_model=512, n_heads=8, d_queries=64, d_values=64, d_inner=2048, n_encoder_layers=6, n_decoder_layers=6, dropout=0.1):
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
            dropout=dropout
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
            dropout=dropout
        )

        # Initialize weights
        self.init_weights(tie_embeddings=tie_embeddings)

    def init_weights(self, tie_embeddings=True):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.)

        # Share weights between the embedding layers and the logit layer
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.fc.weight = self.decoder.embedding.weight

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

class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567
    """

    def __init__(self, args, eps=0.1):
        """
        :param eps: smoothing co-efficient
        """
        super(LabelSmoothedCE, self).__init__()

        self.args = args
        self.eps = eps

    def forward(self, inputs, targets, lengths):
        """
        Forward prop.

        :param inputs: decoded target language sequences, a tensor of size (N, pad_length, vocab_size)
        :param targets: gold target language sequences, a tensor of size (N, pad_length)
        :param lengths: true lengths of these sequences, to be able to ignore pads, a tensor of size (N)
        :return: mean label-smoothed cross-entropy loss, a scalar
        """
        # Remove pad-positions and flatten
        inputs, _, _, _ = pack_padded_sequence(
            input=inputs,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        ) # (sum(lengths), vocab_size)
        targets, _, _, _ = pack_padded_sequence(
            input=targets,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        ) # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = torch.zeros_like(inputs).scatter(dim=1, index=targets.unsqueeze(1), value=1.).to(self.args.device) # (sum(lengths), n_classes), one-hot
        target_vector = target_vector * (1. - self.eps) + self.eps / target_vector.size(1) # (sum(lengths), n_classes), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(inputs, dim=1)).sum(dim=1) # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss
