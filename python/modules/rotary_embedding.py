from rotary_embedding_torch import RotaryEmbedding

import torch.nn as nn

class RotaryEmbeddingModule(nn.Module):
    def __init__(self, rotary_positional_embedding: RotaryEmbedding):
        super(RotaryEmbeddingModule, self).__init__()

        self.rotary_positional_embedding = rotary_positional_embedding

    def forward(self, x):
        # RoPE is applied to the queries and keys after the heads are split out but before the dot product for attention and subsequent softmax operations
        # queries and keys are of shape (N, n_heads, seq len, d_queries/d_keys)
        return self.rotary_positional_embedding.rotate_queries_or_keys(x)
