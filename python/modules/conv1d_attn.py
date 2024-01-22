from rotary_embedding_torch import RotaryEmbedding

import math
import torch
import torch.nn as nn

class Conv1dAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.
    """

    def __init__(self, args, d_model, features, dropout, kernel_size, positional_encoding=None, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param features: number of features to extract
        :param d_queries: size of query vectors (and also the size of the key vectors)
        :param d_values: size of value vectors
        :param dropout: dropout probability
        :param in_decoder: is this Multi-Head Attention sublayer instance in the decoder?
        """
        super(Conv1dAttention, self).__init__()

        self.args=args

        self.d_model = d_model
        self.features = features

        self.positional_encoding = positional_encoding
        self.in_decoder = in_decoder

        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=features, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(features)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        input = query_sequences.transpose(1, 2).contiguous() # (batch_size, d_model, seq_len)

        output = self.conv1d(input) # (batch_size, features, seq_len)

        output = output.transpose(1, 2).contiguous() # (batch_size, seq_len, features)

        output = self.layer_norm(output) # (batch_size, seq_len, features)

        # output = self.apply_dropout(output) # (batch_size, seq_len, features)

        return output # (batch_size, seq_len, features)
