from thop import profile
from torchvision.models import resnet50
import torch

if __name__ == '__main__':
    input_size = 96

    net = resnet50()

    input = torch.randn(1, 3, input_size, input_size)
    flops, params = profile(net, inputs=(input,))


"""
                     flops      params
mobilenetv1-阉割版: 6210050.0,  159442.0
mobilenetv2:       11708320.0, 398690.0
"""

