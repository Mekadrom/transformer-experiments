from .rotary_embedding import RotaryEmbeddingModule
from rotary_embedding_torch import RotaryEmbedding

import torch
import torch.nn as nn

class QKVAttentionMapping(nn.Module):
    def __init__(self, args, positional_encoding):
        super(QKVAttentionMapping, self).__init__()

        self.args = args

        if type(positional_encoding) == RotaryEmbedding:
            self.positional_embedding = RotaryEmbeddingModule(positional_encoding)
        else:
            self.positional_embedding = nn.Identity()

    def forward(self, queries, keys, values, query_sequence_pad_length, key_value_sequence_pad_length):
        queries = self.positional_embedding(queries)
        keys = self.positional_embedding(keys)
            
        # For convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
        queries = queries.contiguous().view(-1, query_sequence_pad_length, self.args.d_queries) # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = keys.contiguous().view(-1, key_value_sequence_pad_length, self.args.d_queries) # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = values.contiguous().view(-1, key_value_sequence_pad_length, self.args.d_values) # (N * n_heads, key_value_sequence_pad_length, d_values)
        attention_weights = torch.bmm(queries, keys.transpose(-2, -1)) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        return attention_weights, queries, keys, values
