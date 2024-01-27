import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiConvAttn(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size, stride, padding, device, positional_encoding=None, in_decoder=False):
        super(MultiConvAttn, self).__init__()

        self.d_model = torch.tensor(d_model, dtype=torch.long).to(device)
        self.d_model.requires_grad = False

        self.n_heads = torch.tensor(n_heads, dtype=torch.long).to(device)
        self.n_heads.requires_grad = False

        self.kernel_size = torch.tensor(kernel_size, dtype=torch.long).to(device)
        self.kernel_size.requires_grad = False

        self.stride = torch.tensor(stride, dtype=torch.long).to(device)
        self.stride.requires_grad = False

        self.padding = torch.tensor(padding, dtype=torch.long).to(device)
        self.padding.requires_grad = False

        self.positional_encoding = positional_encoding

        self.in_decoder = torch.tensor(in_decoder, dtype=torch.bool).to(device)
        self.in_decoder.requires_grad = False

        self.cast_queries = nn.Linear(d_model, n_heads * kernel_size)

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        """
        query_sequences: (batch_size, sequence_length, d_model)
        key_sequences: (batch_size, sequence_length, d_model)
        """
        batch_size = query_sequences.size(0)
        query_sequence_pad_length = query_sequences.size(1)

        query_kernels = self.cast_queries(query_sequences) # (batch_size, sequence_length, n_heads * kernel_size)

        keys = key_sequences.permute(0, 2, 1) # (batch_size, d_model, sequence_length)

        keys = keys.view(1, batch_size * self.d_model.item(), query_sequence_pad_length)
        query_kernels = query_kernels.view(batch_size * self.n_heads.item(), self.d_model.item(), self.kernel_size.item())

        """
        minibatch = batch_size
        in_channels = d_model
        iW = query_sequence_pad_length
        out_channels = n_heads
        groups = batch_size
        kW = kernel_size
        """
        conv_output = F.conv1d(keys, query_kernels, stride=self.stride.item(), padding=self.padding.item(), groups=batch_size)

        conv_output = conv_output.view(batch_size, self.n_heads.item(), query_sequence_pad_length) # (batch_size, n_heads, sequence_length)

        return conv_output
