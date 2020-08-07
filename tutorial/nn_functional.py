import torch
import torch.nn.functional as F


if __name__ == '__main__1':  # F.interpolate 实验
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


if __name__ == '__main__2':  # F.pad 实验
    input = torch.arange(1, 10, dtype=torch.float32).view(1, 1, 3, 3)
    print(input, input.shape)

    x = F.pad(input, [1, 1])
    print(x, x.shape)
    x = F.pad(input, [1, 0, 1, 0])
    print(x, x.shape)
    x = F.pad(input, [1, 1, 1, 1, 1, 1])
    print(x, x.shape)
    x = F.pad(input, [1, 1, 1, 1, 1, 1, 1, 1])
    print(x, x.shape)
    # x = F.pad(input, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # print(x, x.shape)

    x = F.pad(input, [1, 1, 1])
    print(x, x.shape)

