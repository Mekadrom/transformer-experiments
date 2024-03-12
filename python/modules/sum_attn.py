import torch.nn as nn
import utils

class SumAttention(nn.Module):
    def __init__(self, args, multihead_attn, linearconv_attn, apply_nonlinearity=False):
        super(SumAttention, self).__init__()

        self.multihead_attn = multihead_attn
        self.linearconv_attn = linearconv_attn

        if apply_nonlinearity:
            self.cast_output = nn.Sequential(
                nn.Linear(args.d_model, args.d_model),
                utils.create_activation_function(args.activation_function),
            )

    def forward(self, query_sequences, key_sequences, value_sequences, key_value_sequence_lengths):
        mha, attention_weights = self.multihead_attn(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)
        lca = self.linearconv_attn(query_sequences, key_sequences, value_sequences, key_value_sequence_lengths)

        if hasattr(self, 'cast_output'):
            return self.cast_output(mha + lca), attention_weights
        return mha + lca, attention_weights
