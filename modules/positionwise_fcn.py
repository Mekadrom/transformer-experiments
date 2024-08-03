from modules import sparse_moe
from torch import nn

import utils

class PositionWiseFCNetwork(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.layer_norm = norm(args.d_model, args.norm_eps)
        self.activation = utils.create_activation_function(args.d_inner, args.activation_function)
        self.dropout = nn.Dropout(args.dropout)
        
        if args.moe_type == 'simple':
            self.expand = sparse_moe.SparseMoE(args)
        else:
            self.expand = nn.Linear(args.d_model, args.d_inner)

        self.condense = nn.Linear(args.d_inner, args.d_model)

    def forward(self, sequences):
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        if type(self.expand) == nn.Linear:
            sequences = self.expand(sequences) # (N, pad_length, d_inner)
            gating_variances = None
        else:
            sequences, gating_variances = self.expand(sequences)

        sequences = self.activation(sequences)
        sequences = self.dropout(sequences)  # (N, pad_length, d_inner)

        sequences = self.condense(sequences)  # (N, pad_length, d_model)

        sequences = self.dropout(sequences) # (N, pad_length, d_model)

        return sequences, gating_variances
