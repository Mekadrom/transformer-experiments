from .sum import Sum

import admin_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class UnifiedExpertMoE(nn.Module):
    def __init__(self, args):
        super(UnifiedExpertMoE, self).__init__()

        self.args = args

        self.expert_weights = nn.Parameter(torch.Tensor(args.n_experts, args.d_model, args.d_inner))
        self.expert_biases = nn.Parameter(torch.Tensor(args.n_experts, args.d_inner))
        self.gating = nn.Linear(args.d_model, args.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.expert_weights)
        nn.init.zeros_(self.expert_biases)
        
    def forward(self, sequences):
        N, P, _ = sequences.size()
        
        d_inner = self.expert_biases.size(-1)

        # merge batch and sequence dimensions
        sequences = sequences.view(-1, sequences.size(-1)) # (N * pad_length, d_model)

        gating_scores = self.softmax(self.gating(sequences))
        top_k_values, top_k_indices = gating_scores.topk(self.args.moe_top_k, dim=-1)

        output = torch.zeros(N*P, d_inner, device=sequences.device)

        for i in range(self.args.moe_top_k):
            # select the i-th expert weights and biases based on top_k_indices
            selected_weights = self.expert_weights[top_k_indices[:, i]]
            selected_biases = self.expert_biases[top_k_indices[:, i]]
            expert_output = torch.bmm(sequences.unsqueeze(1), selected_weights).squeeze(1) + selected_biases
            output += expert_output * top_k_values[:, i].unsqueeze(-1)
        
        # normalize
        output /= self.args.moe_top_k

        return output.view(N, P, -1)

class SparseMoE(nn.Module):
    def __init__(self, args):
        super(SparseMoE, self).__init__()

        self.args = args

        self.expert_weights = nn.ModuleList([nn.Linear(args.d_model, args.d_inner) for _ in range(args.n_experts)])
        self.gating = nn.Linear(args.d_model, args.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        
        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.expert_weights:
            nn.init.xavier_uniform_(expert.weight)
            nn.init.zeros_(expert.bias)

    def forward(self, sequences):
        N, P, D = sequences.shape

        # merge batch and sequence dimensions
        flat_sequences = sequences.view(-1, D) # (N * pad_length, d_model)
        gating_scores = self.softmax(self.gating(flat_sequences))

        top_k_indices = torch.topk(gating_scores, self.args.moe_top_k, dim=1).indices

        output = torch.zeros(N*P, self.expert_weights[0].out_features, device=sequences.device)

        for i in range(len(self.expert_weights)):
            expert_mask = top_k_indices == i
            expert_input = flat_sequences[expert_mask.any(dim=1)]
            expert_output = self.expert_weights[i](expert_input)

            output[expert_mask.any(dim=1)] += expert_output

        # normalize
        output /= self.args.moe_top_k

        return output.view(N, P, -1)

class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.
    """

    def __init__(self, args, in_decoder=False):
        """
        :param d_model: size of vectors throughout the transformer model, i.e. input and output sizes for this sublayer
        :param d_inner: an intermediate size
        :param dropout: dropout probability
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.layer_norm = nn.LayerNorm(args.d_model)
        self.activation = utils.create_activation_function(args.activation_function)
        self.dropout = nn.Dropout(args.dropout)
        
        if args.use_admin and not in_decoder:
            self.residual = admin_torch.as_module(args.n_layers)
        else:
            self.residual = Sum()

        if args.use_moe:
            self.expand = SparseMoE(args)
        else:
            self.expand = nn.Linear(args.d_model, args.d_inner)

        self.condense = nn.Linear(args.d_inner, args.d_model)

    def forward(self, sequences):
        """
        Forward prop.

        :param sequences: input sequences, a tensor of size (N, pad_length, d_model)
        :return: transformed output sequences, a tensor of size (N, pad_length, d_model)
        """
        # residual connection
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        sequences = self.expand(sequences) # (N, pad_length, d_inner)

        sequences = self.activation(sequences)
        sequences = self.dropout(sequences)  # (N, pad_length, d_inner)

        sequences = self.condense(sequences)  # (N, pad_length, d_model)

        sequences = self.dropout(sequences) # (N, pad_length, d_model)

        sequences = self.residual(input_to_add, sequences) 

        return sequences
