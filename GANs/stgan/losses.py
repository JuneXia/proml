import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridRegular(nn.Module):
    def __init__(self, grad_lambda=1):
        super(GridRegular, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.grad_lambda = grad_lambda

    def forward(self, x):
        xp = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]  # left right up bottom
        yp = F.pad(x, [0, 0, 1, 0])[:, :, 0:-1, :]

        loss_dx = torch.mean(self.criterion(xp, x)[:, :, :, 0:-1])
        loss_dy = torch.mean(self.criterion(yp, x)[:, :, 1:, :])
        return self.grad_lambda * (loss_dx + loss_dy)