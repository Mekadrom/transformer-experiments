import admin_torch
import torch
import torch.nn as nn
import utils

class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, n_layers, d_model, d_inner, activation_function, dropout, use_admin, device, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = torch.tensor(d_model, dtype=torch.long).to(device)
        self.d_model.requires_grad = False

        self.d_inner = torch.tensor(d_inner, dtype=torch.long).to(device)
        self.d_inner.requires_grad = False

        self.layer_norm = nn.LayerNorm(d_model)
        self.expand = nn.Linear(d_model, d_inner)
        self.activation = utils.create_activation_function(activation_function)
        self.condense = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_admin and not in_decoder:
            self.residual = admin_torch.as_module(n_layers)

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

        sequences = self.dropout(sequences) # (N, pad_length, d_model)
        if hasattr(self, 'residual'):
            sequences = self.residual(sequences, input_to_add) 
        else:
            sequences = sequences + input_to_add

        return sequences
