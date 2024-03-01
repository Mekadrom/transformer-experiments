import torch.nn as nn

class TupleIdentity(nn.Module):
    def __init__(self):
        super(TupleIdentity, self).__init__()

    def forward(self, x):
        return x
