from rotary_embedding_torch import RotaryEmbedding

import math
import torch
import torch.nn as nn

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
        :param key_sequences: the sequences to be queried against, a tensor of size (N, key_value_sequence_pad_length, d_model)
        :param key_sequences: the sequences to be queried for, a tensor of size (N, key_value_sequence_pad_length, d_model)
        :param key_value_sequence_lengths: true lengths of the key_value_sequences, to be able to ignore pads, a tensor of size (N)
        :return: attention-weighted output sequences for the query sequences, a tensor of size (N, query_sequence_pad_length, d_model)
        """
        batch_size = query_sequences.size(0) # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_sequences.size(1)

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

        if self.args.qkv_config == 'kv+pos' and type(self.positional_encoding) != RotaryEmbedding:
            pos = self.positional_encoding[:, :key_value_sequence_pad_length, :] # (1, key_value_sequence_pad_length, d_model)
            print(f"sequences: {key_sequences.shape}")
            print(f"pos: {pos.shape}")
            key_sequences = key_sequences + pos # (N, n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
            print(f"key_sequences: {key_sequences.shape}")

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
        if self.args.qkv_config == 'kv+pos' and type(self.positional_encoding) != RotaryEmbedding:
            print(f"queries: {queries.shape}")
            print(f"keys: {keys.shape}")
            print(f"values: {values.shape}")
            print(f"keys.permute(0, 2, 1): {keys.permute(0, 2, 1).shape}")
            attention_weights = torch.bmm(keys if query_sequence_pad_length == key_value_sequence_pad_length else queries, keys.permute(0, 2, 1)) / math.sqrt(int(self.args.d_model / self.args.n_heads)) # (N, key_value_sequence_pad_length, key_value_sequence_pad_length)
            print(f"attention_weights: {attention_weights.shape}")
        else:
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
