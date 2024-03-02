from .qkv_attn_mapping import QKVAttentionMapping
from .kv_pos_attn_mapping import KVPosAttentionMapping

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
    def __init__(self, args, d_output, positional_encoding, in_decoder=False):
        super(MultiHeadAttention, self).__init__()

        self.args = args

        if d_output is None:
            d_output = args.d_model

        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(args.d_model, args.n_heads * args.d_queries) # (N, query_sequence_pad_length, n_heads * d_queries)
        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(args.d_model, args.n_heads * args.d_queries) # (N, key_value_sequence_pad_length, n_heads * d_keys)
        self.cast_values = nn.Linear(args.d_model, args.n_heads * args.d_values) # (N, key_value_sequence_pad_length, n_heads * d_values)

        if args.qkv_config == 'kv+pos':
            self.attention_mapping = KVPosAttentionMapping(args, 1, bias=True)
        else:
            self.attention_mapping = QKVAttentionMapping(args, positional_encoding=positional_encoding)

        if in_decoder:
            self.self_attn_decoder_mask = SelfAttnDecoderMask()
        else:
            self.self_attn_decoder_mask = nn.Identity()

        # a linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors
        self.cast_output = nn.Linear(args.n_heads * args.d_values, d_output)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        batch_size = query_sequences.size(0) # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_sequences, query_sequences)

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
        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.args.n_heads, self.args.d_queries) # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.args.n_heads, self.args.d_queries) # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.args.n_heads, self.args.d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        queries = queries.permute(0, 2, 1, 3) # (N, n_heads, query_sequence_pad_length, d_queries)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Perform multi-head attention

        attention_weights, queries, keys, values = self.attention_mapping(queries, keys, values, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (1. / math.sqrt(self.args.d_queries)) * attention_weights # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(attention_weights.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.args.n_heads).unsqueeze(1).unsqueeze(2).expand_as(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the decoder, keys chronologically ahead of queries
        if self_attention:
            attention_weights = self.self_attn_decoder_mask(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights = self.softmax(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights_for_visualization = attention_weights.clone().detach()

        attention_weights = self.dropout(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values) # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original order of axes
        sequences = sequences.contiguous().view(batch_size, self.args.n_heads, query_sequence_pad_length, self.args.d_values).permute(0, 2, 1, 3) # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1) # (N, query_sequence_pad_length, n_heads * d_values)

        sequences = self.dropout(sequences)

        sequences = self.cast_output(sequences)

        return sequences, attention_weights_for_visualization
