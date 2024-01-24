from modules.conv1d_attn import Conv1dAttention
from modules.multihead_attn import MultiHeadAttention

import torch
import torch.nn as nn

class MultiCastAttention(nn.Module):
    """
    Multicaster that either concats the results of multiple attention modules or runs them sequentially.
    """
    def __init__(self, args, encoder_decoder_layer_index, attn_config, d_queries, d_values, dropout, positional_encoding=None, in_decoder=False, sequential=False):
        super(MultiCastAttention, self).__init__()

        self.args = args

        self.d_queries = d_queries
        self.d_values = d_values
        self.dropout = dropout

        self.positional_encoding = positional_encoding
        self.in_decoder = in_decoder
        self.sequential = sequential

        self.layers, self.embed_dim_list = self.build_layers_from_attn_config(encoder_decoder_layer_index, attn_config)
        self.layers = nn.ModuleList(self.layers)

    def build_layers_from_attn_config(self, encoder_decoder_layer_index, attn_config):
        layer_configs = attn_config.split(',')
        layers = []
        embed_dim_list = []
        kernel_sizes = [int(x) for x in self.args.kernel_sizes.split(',') if x != '']
        for i, layer_config in enumerate(layer_configs):
            layer_config_parts = layer_config.split(':')
            layer_type = layer_config_parts[0]
            layer_output_dim = int(layer_config_parts[1])
            layer_n_heads = int(layer_config_parts[2])
            if layer_type == 'MultiHeadAttention':
                layers.append(MultiHeadAttention(
                    args=self.args,
                    d_model=layer_output_dim,
                    n_heads=layer_n_heads,
                    d_queries=self.d_queries,
                    d_values=self.d_values,
                    dropout=self.dropout,
                    positional_encoding=self.positional_encoding,
                    in_decoder=self.in_decoder
                ))
                embed_dim_list.append(layer_output_dim)
            elif layer_type == 'Conv1dAttention':
                kernel_size = kernel_sizes[encoder_decoder_layer_index] if encoder_decoder_layer_index < len(kernel_sizes) else kernel_sizes[-1]
                layers.append(Conv1dAttention(
                    args=self.args,
                    d_model=layer_output_dim,
                    features=layer_n_heads,
                    dropout=self.dropout,
                    kernel_size=kernel_size,
                    positional_encoding=self.positional_encoding,
                    in_decoder=self.in_decoder
                ))
                embed_dim_list.append(layer_output_dim)
            else:
                raise Exception(f"Unknown attention layer type: {layer_type}")
        return layers, embed_dim_list
    
    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        if self.sequential:
            for layer in self.layers:
                query_sequences = layer(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)
            return query_sequences
        else:
            start = 0
            layer_outputs = []
            for layer in self.layers:
                q = query_sequences[..., start:start + layer.d_model]
                k = key_sequences[..., start:start + layer.d_model]
                v = value_sequences[..., start:start + layer.d_model]
                layer_outputs.append(layer(q, k, v, key_value_sequence_lengths))
                start += layer.d_model

            return torch.cat([layer_output[0] if type(layer_output) in [tuple, list] else layer_output for layer_output in layer_outputs], dim=-1)