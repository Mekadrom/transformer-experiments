import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, positional_encoding):
        super(PositionalEncoding, self).__init__()

        self.positional_encoding = positional_encoding

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :]
