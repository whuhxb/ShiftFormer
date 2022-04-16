import models
import torch.nn as nn
import torch
import fvcore.nn
import fvcore.transforms
import fvcore.common
from fvcore.nn import FlopCountAnalysis
from timm.models import deit_base_patch16_224, deit_small_patch16_224, deit_tiny_patch16_224


model = models.fct_s24_64_7118_TTT_8844()
# model = deit_tiny_patch16_224()

inputs = (torch.randn((1,3,224,224)),)
k = 1000.0
flops = FlopCountAnalysis(model, inputs).total()
print(f"Flops : {flops}")
flops = flops/(k**3)
print(f"Flops : {flops:.1f}G")
params = fvcore.nn.parameter_count(model)[""]
print(f"Params : {params}")
params = params/(k**2)
# print(flops.total())
print(f"Params : {params:.1f}M")
