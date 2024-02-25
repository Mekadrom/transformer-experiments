from .positionwise_fcn import PositionWiseFCNetwork

import torch
import torch.nn as nn

class PositionWiseMixtureOfExpertsNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, n_experts, n_layers, d_model, d_inner, activation_function, dropout, use_admin, device, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseMixtureOfExpertsNetwork, self).__init__()

        self.experts = nn.ModuleList([PositionWiseFCNetwork(n_layers, d_model, d_inner, activation_function, dropout, use_admin, device, in_decoder) for _ in range(n_experts)])
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, sequences):
        # only run through one expert based on the router's decision
        expert_indices = torch.argmax(self.router(sequences), dim=-1)

        print(f"expert_indices: {expert_indices}")

        return self.experts[expert_indices](sequences)
