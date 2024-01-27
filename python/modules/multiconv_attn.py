import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiConvAttn(nn.Module):
    def __init__(self, args, n_heads, kernel_size, stride, padding, positional_encoding=None, in_decoder=False):
        super(MultiConvAttn, self).__init__()

        self.args = args

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.positional_encoding = positional_encoding
        self.in_decoder = in_decoder

        self.cast_queries = nn.Linear(args.d_model, n_heads * kernel_size)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(MultiConvAttn, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict[prefix + 'kernel_size'] = self.kernel_size
        state_dict[prefix + 'stride'] = self.stride
        state_dict[prefix + 'padding'] = self.padding
        state_dict[prefix + 'in_decoder'] = self.in_decoder
        return state_dict
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'kernel_size' in state_dict:
            self.kernel_size = state_dict[prefix + 'kernel_size']
        else:
            missing_keys.append(prefix + 'kernel_size')
        if prefix + 'stride' in state_dict:
            self.stride = state_dict[prefix + 'stride']
        else:
            missing_keys.append(prefix + 'stride')
        if prefix + 'padding' in state_dict:
            self.padding = state_dict[prefix + 'padding']
        else:
            missing_keys.append(prefix + 'padding')
        if prefix + 'in_decoder' in state_dict:
            self.in_decoder = state_dict[prefix + 'in_decoder']
        else:
            missing_keys.append(prefix + 'in_decoder')
        super(MultiConvAttn, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, query_sequences: torch.Tensor, key_sequences: torch.Tensor, value_sequences: torch.Tensor, key_value_sequence_lengths: torch.Tensor):
        """
        query_sequences: (batch_size, sequence_length, d_model)
        key_sequences: (batch_size, sequence_length, d_model)
        """
        batch_size = query_sequences.size(0)

        query_kernels = self.cast_queries(query_sequences) # (batch_size, sequence_length, kernel_size)

        keys = key_sequences.permute(0, 2, 1) # (batch_size, d_model, sequence_length)

        F.conv1d(keys, query_kernels, stride=self.stride, padding=self.padding)