from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, in_channel=3, inshape=26):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # N = (W - F + 2p) / S  + 1
        loc_shape = int(((inshape - 7 + 1)/2 - 5 + 1)/2)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * loc_shape * loc_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        b, c, h, w = xs.shape
        # xs = xs.view(-1, 10 * 3 * 3)
        xs = xs.view(-1, c * h * w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        self.theta_tmp = theta.data.cpu().numpy()
        # print(theta)
        grid = F.affine_grid(theta, x.size()) # affine with theta
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        return x