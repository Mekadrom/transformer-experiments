import math
import torch
import torch.nn as nn
import utils

class MultiHeadAttention(nn.Module):
    def __init__(self, args, self_attn, in_decoder=False, incl_conv=None):
        super(MultiHeadAttention, self).__init__()

        self.args = args
        self.self_attn = self_attn
        self.in_decoder = in_decoder

        if args.qkv_config == 'kv+pos':
            self.map_pos = nn.Linear(args.positional_encoding_dim, 1, bias=True)

        if args.positional_encoding_type == 'rotary':
            self.rotary_embedding = utils.get_positional_encoding(args)

        if incl_conv is None:
            incl_conv = args.use_lite_conv

        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(args.d_model, args.n_heads * args.d_queries) # (N, query_sequence_pad_length, n_heads * d_queries)
        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(args.d_model, args.n_heads * args.d_queries) # (N, key_value_sequence_pad_length, n_heads * d_keys)
        self.cast_values = nn.Linear(args.d_model, args.n_heads * args.d_values) # (N, key_value_sequence_pad_length, n_heads * d_values)

        # don't use lite conv for cross-attention
        if incl_conv:
            self.lite_conv = nn.Sequential(
                nn.Conv1d(args.d_model // 2, args.d_model // 2, kernel_size=3, padding=1, groups=args.d_model // 2),
                nn.Conv1d(args.d_model // 2, args.d_model // 2, kernel_size=1),
            )

            # a linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors
            self.mha_cast_output = nn.Linear(args.n_heads * args.d_values // 2, args.d_model // 2)
        else:
            # a linear projection to cast (n_heads sets of) computed attention-weighted vectors to output vectors
            self.mha_cast_output = nn.Linear(args.n_heads * args.d_values, args.d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

    def multihead_attn(self, queries, keys, values, key_value_sequence_lengths):
        batch_size = queries.size(0) # batch size (N) in number of sequences
        query_sequence_pad_length = queries.size(1)
        key_value_sequence_pad_length = keys.size(1)

        d_queries = queries.size(-1) // self.args.n_heads # dimensionality of the queries
        d_keys = keys.size(-1) // self.args.n_heads # dimensionality of the keys
        d_values = values.size(-1) // self.args.n_heads # dimensionality of the values

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(batch_size, query_sequence_pad_length, self.args.n_heads, d_queries) # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(batch_size, key_value_sequence_pad_length, self.args.n_heads, d_keys) # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(batch_size, key_value_sequence_pad_length, self.args.n_heads, d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the sequence lengths and the queries/keys/values
        queries = queries.permute(0, 2, 1, 3) # (N, n_heads, query_sequence_pad_length, d_queries)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Perform multi-head attention
        if hasattr(self, 'map_pos'):
            # for kv+pos
            B, H, N, _ = queries.shape
            B, H, M, _ = keys.shape

            tmp = keys if N == M else queries

            attention_weights = torch.matmul(tmp, keys.transpose(-2, -1)) / math.sqrt(self.args.n_heads) # (N, query_sequence_pad_length, key_value_sequence_pad_length)

            # assumes positional encoding is tensor
            pos = self.positional_encoding[:, :query_sequence_pad_length, :key_value_sequence_pad_length, :] # (1, query_sequence_pad_length, key_value_sequence_pad_length, d_model)
            attention_weights = attention_weights.unsqueeze(-1) + pos.unsqueeze(0) # (N, query_sequence_pad_length, key_value_sequence_pad_length, d_model)
            attention_weights = self.map_pos(attention_weights).squeeze(-1)

            attention_weights = attention_weights.contiguous().view(-1, query_sequence_pad_length, key_value_sequence_pad_length) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # For convenience, convert to 3D tensors by merging the batch and n_heads dimensions
            # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
            queries = queries.contiguous().view(-1, query_sequence_pad_length, d_queries) # (N * n_heads, query_sequence_pad_length, d_queries)
            keys = keys.contiguous().view(-1, key_value_sequence_pad_length, d_keys) # (N * n_heads, key_value_sequence_pad_length, d_keys)
            values = values.contiguous().view(-1, key_value_sequence_pad_length, d_values) # (N * n_heads, key_value_sequence_pad_length, d_values)
        else:
            if hasattr(self, 'rotary_embedding') and self.rotary_embedding is not None:
                queries = self.rotary_embedding.rotate_queries_or_keys(queries)
                keys = self.rotary_embedding.rotate_queries_or_keys(keys)
                
            # For convenience, convert to 3D tensors by merging the batch and n_heads dimensions
            # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
            queries = queries.contiguous().view(-1, query_sequence_pad_length, d_queries) # (N * n_heads, query_sequence_pad_length, d_queries)
            keys = keys.contiguous().view(-1, key_value_sequence_pad_length, d_queries) # (N * n_heads, key_value_sequence_pad_length, d_keys)
            values = values.contiguous().view(-1, key_value_sequence_pad_length, d_values) # (N * n_heads, key_value_sequence_pad_length, d_values)
            attention_weights = torch.bmm(queries, keys.transpose(-2, -1)) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (1. / math.sqrt(d_queries)) * attention_weights # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = torch.LongTensor(range(key_value_sequence_pad_length)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(attention_weights.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.args.n_heads).unsqueeze(1).unsqueeze(2).expand_as(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the decoder, keys chronologically ahead of queries
        if self.self_attn and self.in_decoder:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(attention_weights.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights = self.softmax(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights_for_visualization = attention_weights.clone().detach()

        attention_weights = self.dropout(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values) # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original order of axes
        sequences = sequences.contiguous().view(batch_size, self.args.n_heads, query_sequence_pad_length, d_values).permute(0, 2, 1, 3) # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1) # (N, query_sequence_pad_length, n_heads * d_values)

        sequences = self.dropout(sequences)

        sequences = self.mha_cast_output(sequences)

        return sequences, attention_weights_for_visualization

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        query_sequences = self.layer_norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.layer_norm(key_sequences)
            value_sequences = self.layer_norm(value_sequences)

        queries: torch.Tensor = self.cast_queries(query_sequences)
        keys: torch.Tensor = self.cast_keys(key_sequences)
        values: torch.Tensor = self.cast_values(value_sequences)

        if hasattr(self, 'lite_conv'):
            # half of embedding dim goes to MHA, half goes to conv
            mha_queries, lite_conv_queries = torch.split(queries, queries.size(-1) // 2, dim=-1)
            mha_keys, lite_conv_keys = torch.split(keys, keys.size(-1) // 2, dim=-1)
            mha_values, lite_conv_values = torch.split(values, values.size(-1) // 2, dim=-1)
        else:
            mha_queries = queries
            mha_keys = keys
            mha_values = values

            lite_conv_queries = None
            lite_conv_keys = None
            lite_conv_values = None

        mha_output, attention_weights = self.multihead_attn(mha_queries, mha_keys, mha_values, key_value_sequence_lengths)

        if hasattr(self, 'lite_conv') and lite_conv_queries is not None and lite_conv_keys is not None and lite_conv_values is not None:
            lite_conv_output = self.lite_conv(lite_conv_queries.transpose(1, 2)).transpose(1, 2) # (N, query_sequence_pad_length, d_model // 2)

            output = torch.cat([mha_output, lite_conv_output], dim=-1)
        else:
            lite_conv_output = None
            output = mha_output

        return output, attention_weights, lite_conv_output
