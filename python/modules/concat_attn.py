import torch
import torch.nn as nn
import utils

class ConcatenateAttention(nn.Module):
    def __init__(self, args, multihead_attn, linearconv_attn, apply_nonlinearity=False):
        super(ConcatenateAttention, self).__init__()

        self.multihead_attn = multihead_attn
        self.linearconv_attn = linearconv_attn

        if apply_nonlinearity:
            self.cast_output = nn.Sequential(
                nn.Linear(args.d_model * 2, args.d_model),
                utils.create_activation_function(args.activation_function),
            )
        else:
            self.cast_output = nn.Sequential(
                nn.Linear(args.d_model * 2, args.d_model)
            )

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        mha, attention_weights = self.multihead_attn(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)
        lca = self.linearconv_attn(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)
        return self.cast_output(torch.cat((mha, lca), dim=-1)), attention_weights
