import math
import torch
import torch.nn as nn

class KVPosAttentionMapping(nn.Module):
    def __init__(self, positional_encoding_dim, bias=True):
        super(KVPosAttentionMapping, self).__init__()

        self.map_pos = nn.Linear(positional_encoding_dim, 1, bias=bias)

    def forward(self, n_heads, d_queries, d_values, queries, keys, values, query_sequence_pad_length, key_value_sequence_pad_length):
        B, H, N, _ = queries.shape
        B, H, M, _ = keys.shape

        tmp = keys if N == M else queries

        attention_weights = torch.matmul(tmp, keys.transpose(-2, -1)) / math.sqrt(n_heads) # (N, query_sequence_pad_length, key_value_sequence_pad_length)

        pos = self.positional_encoding[:, :query_sequence_pad_length, :key_value_sequence_pad_length, :] # (1, query_sequence_pad_length, key_value_sequence_pad_length, d_model)
        attention_weights = attention_weights.unsqueeze(-1) + pos.unsqueeze(0) # (N, query_sequence_pad_length, key_value_sequence_pad_length, d_model)
        attention_weights = self.map_pos(attention_weights).squeeze(-1)

        attention_weights = attention_weights.contiguous().view(-1, query_sequence_pad_length, key_value_sequence_pad_length) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # For convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
        queries = queries.contiguous().view(-1, query_sequence_pad_length, d_queries) # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = keys.contiguous().view(-1, key_value_sequence_pad_length, d_queries) # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = values.contiguous().view(-1, key_value_sequence_pad_length, d_values) # (N * n_heads, key_value_sequence_pad_length, d_values)

        return attention_weights, queries, keys, values