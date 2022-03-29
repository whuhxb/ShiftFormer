import models
import torch.nn as nn
import torch
import fvcore.nn
import fvcore.transforms
import fvcore.common
from fvcore.nn import FlopCountAnalysis


model = models.model1_static_shiftformer_s12()

inputs = (torch.randn((1,3,224,224)),)
k = 1000.0
flops = FlopCountAnalysis(model, inputs).total()

flops = flops/(k**3)
print(f"Flops : {flops:.1f}G")
params = fvcore.nn.parameter_count(model)[""]
params = params/(k**2)
# print(flops.total())
print(f"Params : {params:.1f}M")
