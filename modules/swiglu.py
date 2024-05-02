import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_in):
        super(SwiGLU, self).__init__()

        self.cast = nn.Linear(d_in // 2, d_in)

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.cast(x)
        return x
