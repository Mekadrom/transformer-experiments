import torch.nn as nn
import torch.nn.functional as F

class LinearConvAttention(nn.Module):
    # performs depth-wise separable 1 dimensional convolution on the input sequences
    def __init__(self, args, in_decoder):
        super(LinearConvAttention, self).__init__()

        self.args = args
        self.in_decoder = in_decoder

        self.depthwise = nn.Conv1d(args.d_model, args.d_model, kernel_size=args.kernel_size, groups=args.d_model, padding=args.kernel_size // 2)
        self.pointwise = nn.Conv1d(args.d_model, args.d_model, kernel_size=1)

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        """
        param query_sequences: tensor of shape (batch_size, query_sequence_pad_length, d_model/in_channels)
        param key_sequences: tensor of shape (batch_size, key_value_sequence_pad_length, d_model/in_channels)
        """

        sequences = query_sequences

        sequences = sequences.transpose(1, 2) # (batch_size, d_model, query_sequence_pad_length)

        sequences = self.depthwise(sequences)
        sequences = self.pointwise(sequences)

        sequences = sequences.transpose(1, 2)

        return sequences
