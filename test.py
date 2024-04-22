import torch
import time

# Create a sample tensor
tensor = torch.randn(5000, 1000, 1000)

# Measure time for tensor.transpose(-2, -1)
start_time = time.time()
transposed = tensor.transpose(-2, -1)
transpose_time = time.time() - start_time

# Measure time for tensor.permute(0, 2, 1)
start_time = time.time()
permuted = tensor.permute(0, 2, 1)
permute_time = time.time() - start_time

print(str(transpose_time), str(permute_time))
