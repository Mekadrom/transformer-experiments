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

        # A linear projection to cast (n_heads sets of) queries from the input query sequences
        self.cast_queries = nn.Linear(args.d_model, args.n_heads * args.d_queries) # (N, query_sequence_pad_length, n_heads * d_queries)
        # A linear projection to cast (n_heads sets of) keys and values from the input reference sequences
        self.cast_keys = nn.Linear(args.d_model, args.n_heads * args.d_queries) # (N, key_value_sequence_pad_length, n_heads * d_keys)
        self.cast_values = nn.Linear(args.d_model, args.n_heads * args.d_values) # (N, key_value_sequence_pad_length, n_heads * d_values)

        # a linear projection to cast (n_q_heads sets of) computed attention-weighted vectors to output vectors
        self.mha_cast_output = nn.Linear(args.n_heads * args.d_values, args.d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths, key_padding_mask=None, attn_mask=None):
        query_sequences = self.layer_norm(query_sequences)

        # if this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
            key_sequences = self.layer_norm(key_sequences)
            value_sequences = self.layer_norm(value_sequences)

        q_heads = self.cast_queries(query_sequences)
        k_heads = self.cast_keys(key_sequences)
        v_heads = self.cast_values(value_sequences)

        N = q_heads.size(0) # batch size (N) in number of sequences
        t = q_heads.size(1) # query sequence padded lengths
        T = k_heads.size(1) # key-value sequence padded lengths

        d_queries = q_heads.size(-1) // self.args.n_heads # dimensionality of the queries
        d_keys = k_heads.size(-1) // self.args.n_heads # dimensionality of the keys
        d_values = v_heads.size(-1) // self.args.n_heads # dimensionality of the values

        # Split the last dimension by the n_heads subspaces
        q_heads = q_heads.contiguous().view(N, t, self.args.n_heads, d_queries) # (N, query_sequence_pad_length, n_heads, q_heads_per_kv_head, d_queries)
        k_heads = k_heads.contiguous().view(N, T, self.args.n_heads, d_keys) # (N, key_value_sequence_pad_length, n_heads, d_keys)
        v_heads = v_heads.contiguous().view(N, T, self.args.n_heads, d_values) # (N, key_value_sequence_pad_length, n_heads, d_values)

        q_heads = q_heads.permute(0, 2, 1, 3)
        k_heads = k_heads.permute(0, 2, 1, 3)
        v_heads = v_heads.permute(0, 2, 1, 3)

        if hasattr(self, 'rotary_embedding') and self.rotary_embedding is not None:
            q_heads = self.rotary_embedding.rotate_queries_or_keys(q_heads)
            k_heads = self.rotary_embedding.rotate_queries_or_keys(k_heads)

        # generate attention weights by taking the dot product of queries and keys
        attention_weights = torch.einsum('...td,...Td->...tT', q_heads, k_heads) # (N, heads, query_sequence_pad_length, key_value_sequence_pad_length) OR (NhtT)
        attention_weights = (1.0 / math.sqrt(d_queries)) * attention_weights
        attention_weights = 30.0 * torch.tanh(attention_weights / 30.0)

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == attention_weights.shape[0], f"batch dimension for padding is wrong: {key_padding_mask.shape[0]} != {attention_weights.shape[0]}. overall shape: {key_padding_mask.shape} != {attention_weights.shape}"
            assert key_padding_mask.shape[1] == attention_weights.shape[3], f"padding mask length is wrong: {key_padding_mask.shape[1]} != {attention_weights.shape[3]}. overall shape: {key_padding_mask.shape} != {attention_weights.shape}"

            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

            # print(f"key_padding_mask: {key_padding_mask.shape}, attention_weights: {attention_weights.shape}")

            # mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            # print(f"key_padding_mask: {key_padding_mask}")
            attention_weights = attention_weights.masked_fill(key_padding_mask, -float('inf'))
            # print(f"attention_weights: {attention_weights == -float('inf')}")

        if self.self_attn and self.in_decoder:
            assert attn_mask is not None, "attn_mask must be provided for decoder self-attention"

        if attn_mask is not None:
            assert attn_mask.shape[0] == attention_weights.shape[2], f"attn_mask length is wrong: {attn_mask.shape[0]} != {attention_weights.shape[2]}. overall shape: {attn_mask.shape} != {attention_weights.shape}"
            assert attn_mask.shape[1] == attention_weights.shape[3], f"attn_mask length is wrong: {attn_mask.shape[1]} != {attention_weights.shape[3]}. overall shape: {attn_mask.shape} != {attention_weights.shape}"

            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            # mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(attn_mask, -float('inf'))

        attention_weights = self.softmax(attention_weights)

        # for visualization, switch the kv_heads and q_per_kv_heads dimensions
        attention_weights_for_visualization = attention_weights.clone().detach() # (N, n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights = self.dropout(attention_weights)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.einsum('...tT,...Td->...td', attention_weights, v_heads) # (N, n_heads, query_sequence_pad_length, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.permute(0, 2, 1, 3).contiguous().view(N, t, -1) # (N, query_sequence_pad_length, n_q_heads * d_values)

        sequences = self.dropout(sequences)

        sequences = self.mha_cast_output(sequences)

        return sequences, attention_weights_for_visualization
