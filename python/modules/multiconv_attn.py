import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiConvAttn(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size, device, positional_encoding=None):
        super(MultiConvAttn, self).__init__()

        self.positional_encoding = positional_encoding

        self.cast_queries = nn.Linear(d_model, n_heads * kernel_size)
        self.conv_bias = torch.zeros(n_heads).to(device)
        self.conv_bias.requires_grad = True

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        """
        query_sequences: (batch_size, sequence_length, d_model)
        key_sequences: (batch_size, sequence_length, d_model)
        """
        d_model = self.cast_queries.in_features
        n_heads = self.conv_bias.size(0)
        kernel_size = self.cast_queries.out_features // n_heads

        batch_size = query_sequences.size(0)
        query_sequence_pad_length = query_sequences.size(1)

        weights = self.cast_queries(query_sequences) # (N, sequence_length, n_heads * kernel_size)

        # switch d_model and sequence_length dimensions for conv1d
        input_keys = key_sequences.permute(0, 2, 1) # (batch_size, d_model, sequence_length)

        # merge batch and d_model dimensions for conv1d grouping
        input_reshaped = input_keys.contiguous().view(1, batch_size * d_model, query_sequence_pad_length) # (1, batch_size * d_model, sequence_length)
        weights_reshaped = weights.view(batch_size * n_heads, 1, query_sequence_pad_length * kernel_size) # (batch_size * n_heads, 1, sequence_length * kernel_size)

        """
        minibatch = batch_size
        in_channels = d_model
        iW = query_sequence_pad_length
        out_channels = n_heads
        groups = batch_size
        kW = kernel_size
        """
        print(f"input_reshaped: {input_reshaped.size()}")
        print(f"weights_reshaped: {weights_reshaped.size()}")
        
        conv_output = F.conv1d(input_reshaped, weights_reshaped, bias=self.conv_bias, padding=int((kernel_size - 1) / 2), groups=batch_size)

        # switch batch and d_model dimensions back
        conv_output = conv_output.view(batch_size, n_heads, -1) # (batch_size, n_heads, sequence_length)

        return conv_output
