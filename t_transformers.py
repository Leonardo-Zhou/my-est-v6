import torch
import math
import torch.nn.functional as F
import warnings
import collections.abc as container_abcs
import torch.nn as nn
from einops import rearrange
import networks

# Test with print for debug
A_seq = torch.randn(1, 5, 3, 256, 320)  # Small size to avoid tool memory issue
M_seq = torch.randn(1, 5, 1, 256, 320)
model = networks.MaskedSpatioTemporalReflectionModule(patch_size=16, embed_dim=128, depth=12)  # Adjust patch for small H/W
print("Input A_seq shape:", A_seq.shape)
out = model(A_seq, M_seq)
print("Output shape:", out.shape)