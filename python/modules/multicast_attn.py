from modules.multiconv_attn import MultiConvAttn
from modules.multihead_attn import MultiHeadAttention

import torch
import torch.nn as nn

class MultiCastAttention(nn.Module):
    """
    Multicaster that either concats the results of multiple attention modules or runs them sequentially.
    """
    def __init__(self, d_model, d_queries, d_values, qkv_config, dropout, use_admin, attn_config, device, positional_encoding_dim, positional_encoding=None, in_decoder=False, sequential=False):
        super(MultiCastAttention, self).__init__()

        self.layers, self.embed_dim_list = self.build_layers_from_attn_config(d_model, d_queries, d_values, qkv_config, dropout, use_admin, attn_config, positional_encoding_dim, positional_encoding, in_decoder, sequential, device)
        self.layers = nn.ModuleList(self.layers)
        
        self.embed_dim_list = torch.LongTensor(self.embed_dim_list).to(device)
        self.embed_dim_list.requires_grad = False

        self.sequential = torch.BoolTensor([sequential]).to(device)
        self.sequential.requires_grad = False

    def build_layers_from_attn_config(self, d_model, d_queries, d_values, qkv_config, dropout, use_admin, attn_config, positional_encoding_dim, positional_encoding, in_decoder, sequential, device):
        layer_configs = attn_config.split(',')
        layers = []
        embed_dim_list = []
        for i, layer_config in enumerate(layer_configs):
            layer_config_parts = layer_config.split(':')
            layer_type = layer_config_parts[0]
            layer_output_dim = int(layer_config_parts[1])
            layer_n_heads = int(layer_config_parts[2])
            if layer_type == 'MultiHeadAttention':
                layers.append(MultiHeadAttention(
                    n_layers=len(layer_configs),
                    d_model=d_model,
                    n_heads=layer_n_heads,
                    d_queries=d_queries,
                    d_values=d_values,
                    qkv_config=qkv_config,
                    dropout=dropout,
                    use_admin=use_admin,
                    device=device,
                    positional_encoding_dim=positional_encoding_dim,
                    d_output=layer_output_dim,
                    positional_encoding=positional_encoding,
                    in_decoder=in_decoder
                ))
                embed_dim_list.append(layer_output_dim)
            elif layer_type == 'MultiConvAttention':
                layers.append(MultiConvAttn(
                    d_model=d_model,
                    n_heads=layer_n_heads,
                    kernel_size=layer_n_heads,
                    device=device,
                    positional_encoding=positional_encoding,
                    in_decoder=in_decoder
                ))
                embed_dim_list.append(layer_output_dim)
            else:
                raise Exception(f"Unknown attention layer type: {layer_type}")
        return layers, embed_dim_list
    
    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        if self.sequential.item():
            for layer in self.layers:
                query_sequences = layer(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)
            return query_sequences
        else:
            start = 0
            layer_outputs = []
            for layer in self.layers:
                # q = query_sequences[..., start:start + layer.d_model.item()]
                # k = key_sequences[..., start:start + layer.args.d_model]
                # v = value_sequences[..., start:start + layer.args.d_model]
                # layer_outputs.append(layer(q, k, v, key_value_sequence_lengths))
                layer_outputs.append(layer(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths))
                # start += layer.args.d_model

            return torch.cat([layer_output[0] if type(layer_output) in [tuple, list] else layer_output for layer_output in layer_outputs], dim=-1)
