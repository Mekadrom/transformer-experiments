import math
import torch
import torch.nn as nn
import utils

class MultiHeadAttention(nn.Module):
    def __init__(self, args, self_attn, in_decoder=False):
        super(MultiHeadAttention, self).__init__()

        self.args = args
        self.self_attn = self_attn
        self.in_decoder = in_decoder

        if args.positional_encoding_type == 'rotary':
            self.rotary_embedding = utils.get_positional_encoding(args)

        # A linear projection to cast (n_kv_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(args.d_model, args.n_q_heads * args.d_queries) # (N, query_sequence_pad_length, n_kv_heads * d_queries)
        # A linear projection to cast (n_kv_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(args.d_model, args.n_kv_heads * args.d_queries) # (N, key_value_sequence_pad_length, n_kv_heads * d_keys)
        self.cast_values = nn.Linear(args.d_model, args.n_kv_heads * args.d_values) # (N, key_value_sequence_pad_length, n_kv_heads * d_values)

        # a linear projection to cast (n_q_heads sets of) computed attention-weighted vectors to output vectors
        self.mha_cast_output = nn.Linear(args.n_q_heads * args.d_values, args.d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

    def multihead_attn(self, q_heads, k_heads, v_heads, key_value_sequence_lengths):
        N = q_heads.size(0) # batch size (N) in number of sequences
        t = q_heads.size(1) # query sequence padded lengths
        T = k_heads.size(1) # key-value sequence padded lengths

        d_queries = q_heads.size(-1) // self.args.n_q_heads # dimensionality of the queries
        d_keys = k_heads.size(-1) // self.args.n_kv_heads # dimensionality of the keys
        d_values = v_heads.size(-1) // self.args.n_kv_heads # dimensionality of the values

        # Split the last dimension by the n_kv_heads subspaces
        q_heads = q_heads.contiguous().view(N, t, self.args.n_kv_heads, self.args.n_q_heads // self.args.n_kv_heads, d_queries) # (N, query_sequence_pad_length, n_kv_heads, q_heads_per_kv_head, d_queries)
        k_heads = k_heads.contiguous().view(N, T, self.args.n_kv_heads, d_keys) # (N, key_value_sequence_pad_length, n_kv_heads, d_keys)
        v_heads = v_heads.contiguous().view(N, T, self.args.n_kv_heads, d_values) # (N, key_value_sequence_pad_length, n_kv_heads, d_values)

        if hasattr(self, 'rotary_embedding') and self.rotary_embedding is not None:
            q_heads = self.rotary_embedding.rotate_queries_or_keys(q_heads)
            k_heads = self.rotary_embedding.rotate_queries_or_keys(k_heads)

        # compute attention weights    
        attention_weights = torch.einsum('...thHd,...Thd->...hHtT', q_heads, k_heads) # (N, n_kv_heads, q_heads_per_kv_heads, query_sequence_pad_length, key_value_sequence_pad_length) OR (NhHtT)

        # Scale dot-products
        attention_weights = (1. / math.sqrt(d_queries)) * attention_weights

        # Before computing softmax weights, prevent queries from attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = torch.LongTensor(range(T)).unsqueeze(0).unsqueeze(0).expand_as(attention_weights).to(attention_weights.device)  # (N, n_kv_heads, q_heads_per_kv_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # print(f"not_pad_in_keys: {not_pad_in_keys.shape}")
        # print(f"key_value_sequence_lengths: {key_value_sequence_lengths.shape}")
        # print(f"key_value_sequence_lengths.repeat_interleave: {key_value_sequence_lengths.repeat_interleave(self.args.n_kv_heads, dim=-1).shape}")
        not_pad_in_keys = not_pad_in_keys < key_value_sequence_lengths.repeat_interleave(self.args.n_kv_heads, dim=-1).unsqueeze(-1).repeat_interleave(self.args.n_q_heads // self.args.n_kv_heads, dim=-1).unsqueeze(-1).unsqueeze(-1).expand_as(attention_weights)  # (N, n_kv_heads, q_heads_per_kv_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(~not_pad_in_keys, -float('inf'))

        # MASK 2: if this is self-attention in the decoder, keys chronologically ahead of queries
        if self.self_attn and self.in_decoder:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(attention_weights.device)

            # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(~not_future_mask, -float('inf'))

        attention_weights = self.softmax(attention_weights)

        # for visualization, switch the kv_heads and q_per_kv_heads dimensions
        attention_weights_for_visualization = attention_weights.permute(0, 2, 1, 3, 4).clone().detach() # (N, q_heads_per_kv_heads, n_kv_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights = self.dropout(attention_weights)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.einsum('...hHtT,...Thd->...thHd', attention_weights, v_heads) # (N, query_sequence_pad_length, n_kv_heads, n_q_heads // n_kv_heads, d_values)

        # Concatenate the n_kv_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(N, t, -1) # (N, query_sequence_pad_length, n_q_heads * d_values)

        sequences = self.dropout(sequences)

        sequences = self.mha_cast_output(sequences)

        return sequences, attention_weights_for_visualization

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        query_sequences = self.layer_norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.layer_norm(key_sequences)
            value_sequences = self.layer_norm(value_sequences)

        query_heads: torch.Tensor = self.cast_queries(query_sequences)
        key_heads: torch.Tensor = self.cast_keys(key_sequences)
        value_heads: torch.Tensor = self.cast_values(value_sequences)

        return self.multihead_attn(query_heads, key_heads, value_heads, key_value_sequence_lengths)
