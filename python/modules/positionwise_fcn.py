import torch.nn as nn
import utils

class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, args):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.layer_norm = nn.LayerNorm(self.args.d_model)
        self.expand = nn.Linear(self.args.d_model, self.args.d_inner)
        self.activation = utils.create_activation_function(args.activation_function)
        self.condense = nn.Linear(self.args.d_inner, self.args.d_model)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, sequences):
        """
        Forward prop.

        :param sequences: input sequences, a tensor of size (N, pad_length, d_model)
        :return: transformed output sequences, a tensor of size (N, pad_length, d_model)
        """
        # residual connection
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        sequences = self.dropout(self.activation(self.expand(sequences)))  # (N, pad_length, d_inner)
        sequences = self.condense(sequences)  # (N, pad_length, d_model)

        # Apply dropout and residual connection
        sequences = self.dropout(sequences) + input_to_add  # (N, pad_length, d_model)

        return sequences
