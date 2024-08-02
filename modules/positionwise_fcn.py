import torch
import torch.nn as nn
import utils

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

        # record export choices to self.gating_variances for loss calculation to encourage diversity
        if self.training:
            gating_variances = torch.var(gating_scores, dim=0)
        else:
            gating_variances = None

        # normalize
        output /= self.args.moe_top_k

        return output.view(N, P, -1), gating_variances
    
class PositionWiseFCNetwork(nn.Module):
    def __init__(self, args, norm=nn.LayerNorm):
        super(PositionWiseFCNetwork, self).__init__()

        self.args = args

        self.layer_norm = norm(args.d_model, args.norm_eps)
        self.activation = utils.create_activation_function(args.d_inner, args.activation_function)
        self.dropout = nn.Dropout(args.dropout)
        
        if args.moe_type == 'simple':
            self.expand = SparseMoE(args)
        elif args.moe_type == 'millions':
            self.expand = MillionsOfExperts(args)
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
