import torch
import torch.nn.functional as F


input = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
print(input)
x = F.interpolate(input, scale_factor=1.5, mode='nearest')
print(x)
x = F.interpolate(input, scale_factor=2, mode='nearest')
print(x)
x = F.interpolate(input, scale_factor=2, mode='bilinear')
print(x)
x = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
print(x)
