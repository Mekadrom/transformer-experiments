import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x):

        x_sqr = x**2
        RMS = torch.rsqrt(x_sqr.mean(dim = -1, keepdim = True) + self.eps)
        new_x = x * RMS
        new_x = new_x * self.weight

        return new_x