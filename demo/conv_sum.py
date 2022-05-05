# to varify if conv(A) + conv(B) = conv(A+B)
# The answer is yes (ignoring bias)
import torch
import torch.nn as nn


# dim=1
# W,H = 6,6
# feature = torch.rand(1, dim, W, H)
# gap = torch.rand(1, dim, 1, 1).expand_as(feature)
# conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
# conv.eval()
# out1 = conv(feature)+conv(gap)
# out2 = conv(feature+gap)
# print(out1-out2)
# print(gap)
# print(conv.bias)

# check DW conv parameter shape
conv = nn.Conv2d(64,64,3,groups=64, bias=False)
print(conv.weight.shape)

