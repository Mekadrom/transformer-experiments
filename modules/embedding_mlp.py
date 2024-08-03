from torch import nn

import torch

class EmbeddingMLP(nn.Module):
    def __init__(self, vocab_size, compress_dim, emb_dim, activate=nn.Identity):
        super(EmbeddingMLP, self).__init__()

        self.embedding = nn.Embedding(vocab_size, compress_dim)
        self.activate = activate()
        self.compress = nn.Linear(compress_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.activate(x)
        x = x.to(self.compress.weight.device)
        x = self.compress(x)
        return x
