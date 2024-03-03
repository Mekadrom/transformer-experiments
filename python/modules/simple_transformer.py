from rotary_embedding_torch import RotaryEmbedding

import math
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, args):
        super(FCN, self).__init__()

        self.expand = nn.Linear(args.d_model, args.d_inner)
        self.condense = nn.Linear(args.d_inner, args.d_model)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.condense(x)
        x = self.dropout(x)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args, self_attn, in_decoder=False):
        super(MultiHeadAttention, self).__init__()

        self.args = args
        self.self_attn = self_attn
        self.in_decoder = in_decoder

        self.cast_queries = nn.Linear(args.d_model, args.n_heads * args.d_queries)
        self.cast_keys = nn.Linear(args.d_model, args.n_heads * args.d_queries)
        self.cast_values = nn.Linear(args.d_model, args.n_heads * args.d_values)

        self.rotary_positional_encoding = RotaryEmbedding(args.positional_encoding_dim)

        self.cast_output = nn.Linear(args.n_heads * args.d_values, args.mha_d_output)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        batch_size = query_sequences.size(0) # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_sequences, query_sequences)

        # Apply layer normalization
        query_sequences = self.layer_norm(query_sequences) # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value sequences (as they are the same as the query sequences)
        # If this isn't self-attention, they will already have been normed in the last layer of the Encoder (from whence they came)
        if self.self_attn:
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

        # rotary positional encoding applied here
        queries =  self.rotary_positional_encoding.rotate_queries_or_keys(queries)
        keys = self.rotary_positional_encoding.rotate_queries_or_keys(keys)
            
        # For convenience, convert to 3D tensors by merging the batch and n_heads dimensions
        # This is to prepare it for the batch matrix multiplication (i.e. the dot product)
        queries = queries.contiguous().view(-1, query_sequence_pad_length, self.args.d_queries) # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = keys.contiguous().view(-1, key_value_sequence_pad_length, self.args.d_queries) # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = values.contiguous().view(-1, key_value_sequence_pad_length, self.args.d_values) # (N * n_heads, key_value_sequence_pad_length, d_values)
        attention_weights = torch.bmm(queries, keys.transpose(-2, -1)) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

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
        if self_attention and self.in_decoder:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j > i to 0
            not_future_mask = torch.ones_like(attention_weights).tril().bool().to(attention_weights.device) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative number, so that they evaluate to 0 under the softmax
            attention_weights.masked_fill(~not_future_mask, -float('inf')) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights = self.softmax(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        attention_weights = self.dropout(attention_weights) # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on these softmax weights
        sequences = torch.bmm(attention_weights, values) # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original order of axes
        sequences = sequences.contiguous().view(batch_size, self.args.n_heads, query_sequence_pad_length, self.args.d_values).permute(0, 2, 1, 3) # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size d_values)
        sequences = sequences.contiguous().view(batch_size, query_sequence_pad_length, -1) # (N, query_sequence_pad_length, n_heads * d_values)

        sequences = self.dropout(sequences)

        sequences = self.cast_output(sequences)

        return sequences
    
class Encoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, args.d_model)

        self.self_attn = nn.ModuleList([MultiHeadAttention(args, True, False) for _ in range(args.n_encoder_layers)])
        self.fcn = nn.ModuleList([FCN(args) for _ in range(args.n_encoder_layers)])
        self.layer_norm = nn.LayerNorm(args.d_model)

        self.activation = nn.ReLU()
        self.apply_dropout = nn.Dropout(args.dropout)

    def forward(self, encoder_sequences, encoder_sequence_lengths):
        d_model = self.embedding.weight.size(1)

        encoder_sequences = self.embedding(encoder_sequences) * math.sqrt(d_model)
        encoder_sequences = self.apply_dropout(encoder_sequences)

        for self_attn, fcn in zip(self.self_attn, self.fcn):
            # multi head attention (self-attention) with residual
            residual = encoder_sequences.clone()
            encoder_sequences = self_attn(encoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths)
            encoder_sequences += residual

            # ffn/fcn with residual
            residual = encoder_sequences.clone()
            encoder_sequences = fcn(encoder_sequences)
            encoder_sequences += residual

        encoder_sequences = self.layer_norm(encoder_sequences)

        return encoder_sequences
    
class Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Decoder, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(vocab_size, args.d_model)

        self.self_attn = nn.ModuleList([MultiHeadAttention(args, True, True) for _ in range(args.n_decoder_layers)])
        self.cross_attn = nn.ModuleList([MultiHeadAttention(args, False, True) for _ in range(args.n_decoder_layers)])
        self.fcn = nn.ModuleList([FCN(args) for _ in range(args.n_decoder_layers)])
        self.layer_norm = nn.LayerNorm(args.d_model)

        self.generator = nn.Linear(args.d_model, vocab_size)

        self.activation = nn.ReLU()
        self.apply_dropout = nn.Dropout(args.dropout)

    def forward(self, decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths):
        d_model = self.embedding.weight.size(1)
        decoder_sequences = self.embedding(decoder_sequences) * math.sqrt(d_model)
        decoder_sequences = self.apply_dropout(decoder_sequences)

        for self_attn, cross_attn, fcn in zip(self.self_attn, self.cross_attn, self.fcn):
            # multi head attention (self-attention) with residual
            residual = decoder_sequences.clone()
            decoder_sequences = self_attn(decoder_sequences, decoder_sequences, decoder_sequences, decoder_sequence_lengths)
            decoder_sequences += residual

            # multi head attention (cross-attention) with residual
            residual = decoder_sequences.clone()
            decoder_sequences = cross_attn(decoder_sequences, encoder_sequences, encoder_sequences, encoder_sequence_lengths)
            decoder_sequences += residual

            # ffn/fcn with residual
            residual = decoder_sequences.clone()
            decoder_sequences = fcn(decoder_sequences)
            decoder_sequences += residual

        decoder_sequences = self.layer_norm(decoder_sequences)
        decoder_sequences = self.generator(decoder_sequences)

        return decoder_sequences

class SimpleTransformer(nn.Module):
    def __init__(self, args, src_vocab_size, tgt_vocab_size):
        super(SimpleTransformer, self).__init__()

        self.args = args

        self.encoder = Encoder(args, src_vocab_size)
        self.decoder = Decoder(args, tgt_vocab_size)

    def init_weights(self, tie_embeddings=False):
        # Glorot uniform initialization with a gain of self.init_weights_gain
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
        nn.init.normal_(self.encoder.embedding.weight, mean=0., std=math.pow(self.args.d_model, -0.5))
        self.decoder.embedding.weight = self.encoder.embedding.weight

        if tie_embeddings:
            self.decoder.generator.weight = self.decoder.embedding.weight

    def forward(self, encoder_sequences, decoder_sequences, encoder_sequence_lengths, decoder_sequence_lengths):
        encoder_sequences = self.encoder(encoder_sequences, encoder_sequence_lengths)
        decoder_sequences = self.decoder(decoder_sequences, decoder_sequence_lengths, encoder_sequences, encoder_sequence_lengths)
        return decoder_sequences
