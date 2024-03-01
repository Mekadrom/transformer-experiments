from .qkv_attn_mapping import QKVAttentionMapping
from .kv_pos_attn_mapping import KVPosAttentionMapping
from .sum import Sum

import admin_torch
import math
import torch
import torch.nn as nn

class SelfAttnDecoderMask(nn.Module):
    def __init__(self):
        super(SelfAttnDecoderMask, self).__init__()

    def forward(self, attention_weights):
        # Therefore, a position [n, i, j] is valid only if j <= i
        # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
        not_future_mask = torch.ones_like(attention_weights).tril().bool().to(attention_weights.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        return attention_weights.masked_fill(~not_future_mask, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.
    """
    def __init__(self, n_layers, d_model, n_heads, d_queries, d_values, qkv_config, dropout, use_admin, device, positional_encoding_dim, d_output=None, positional_encoding=None, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param n_heads: number of heads in the multi-head attention
        :param d_queries: size of query vectors (and also the size of the key vectors)
        :param d_values: size of value vectors
        :param dropout: dropout probability
        :param in_decoder: is this Multi-Head Attention sublayer instance in the decoder?
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads

        if d_output is None:
            d_output = n_heads * d_values

        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries) # (N, query_sequence_pad_length, n_heads * d_queries)
        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(d_model, n_heads * d_queries) # (N, key_value_sequence_pad_length, n_heads * d_keys)
        self.cast_values = nn.Linear(d_model, n_heads * d_values) # (N, key_value_sequence_pad_length, n_heads * d_values)

        if qkv_config == 'kv+pos':
            self.attention_mapping = KVPosAttentionMapping(positional_encoding_dim, 1, bias=True)
        else:
            self.attention_mapping = QKVAttentionMapping(positional_encoding)

        if in_decoder:
            self.self_attn_decoder_mask = SelfAttnDecoderMask()
        else:
            self.self_attn_decoder_mask = nn.Identity()

        # A linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors (of the same size as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_output)

        if use_admin and not in_decoder:
            self.residual = admin_torch.as_module(n_layers)
        else:
            self.residual = Sum()

        if d_model != d_output:
            self.cast_output = nn.Linear(d_model, d_output)
        else:
            self.cast_output = nn.Identity()

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        """
        Forward prop.

        :param query_sequences: the input query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        :param key_sequences: the sequences to be queried against, a tensor of size (N, key_value_sequence_pad_length, d_model)
        :param value_sequences: the sequences to be queried for, a tensor of size (N, key_value_sequence_pad_length, d_model)
        :param key_value_sequence_lengths: true lengths of the key_value_sequences, to be able to ignore pads, a tensor of size (N)
        :return: attention-weighted output sequences for the query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        """
        batch_size = query_sequences.size(0) # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_sequences.size(1)

        d_queries = self.cast_queries.out_features // self.n_heads
        d_values = self.cast_values.out_features // self.n_heads

        # Is this self-attention?
        self_attention = torch.equal(key_sequences, query_sequences)

        # Store input for adding later
        input_to_add = query_sequences.clone()

        # Apply layer normalization
        query_sequences = self.layer_norm(query_sequences) # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value sequences (as they are the same as the query sequences)
        # If this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self_attention:
            key_sequences = self.layer_norm(key_sequences) # (N, key_value_sequence_pad_length, d_model)
            value_sequences = self.layer_norm(value_sequences) # (N, key_value_sequence_pad_length, d_model)

        # Project input sequences to queries, keys, values
        queries: torch.Tensor = self.cast_queries(query_sequences) # (N, query_sequence_pad_length, n_heads * d_queries)
        keys: torch.Tensor = self.cast_keys(key_sequences) # (N, key_value_sequence_pad_length, n_heads * d_keys)
        values: torch.Tensor = self.cast_values(value_sequences) # (N, key_value_sequence_pad_length, n_heads * d_values)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.n_heads, d_queries) # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads, d_queries) # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.n_heads, d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        queries = queries.permute(0, 2, 1, 3) # (N, n_heads, query_sequence_pad_length, d_queries)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Perform multi-head attention

        # Perform dot-products
        attention_weights, queries, keys, values = self.attention_mapping(self.n_heads, d_queries, d_values, queries, keys, values, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (1. / math.sqrt(d_queries)) * attention_weights # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(attention_weights.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.n_heads).unsqueeze(1).unsqueeze(2).expand_as(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the decoder, keys chronologically ahead of queries
        if self_attention:
            attention_weights = self.self_attn_decoder_mask(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Compute softmax along the key dimension
        attention_weights = self.softmax(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights_for_visualization = attention_weights.clone().detach()

        # Apply dropout
        attention_weights = self.dropout(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values) # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original order of axes
        sequences = sequences.contiguous().view(batch_size, self.n_heads, query_sequence_pad_length, d_values).permute(0, 2, 1, 3) # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1) # (N, query_sequence_pad_length, n_heads * d_values)

        sequences = self.dropout(sequences)

        sequences = self.cast_output(sequences)

        sequences = self.residual(input_to_add, sequences)

        return sequences, attention_weights_for_visualization
