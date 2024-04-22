import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim=-1, bias=True):
        super(SwiGLU, self).__init__()
        
        self.dim = dim
        self.dense = nn.Linear(1, 2, bias=bias)

    def forward(self, x):
        x = self.dense(x)
        out, gate = torch.chunk(x, 2, dim=self.dim)
        gate = F.silu(gate)  # SiLU is the PyTorch equivalent of Swish
        x = out * gate
        return x
