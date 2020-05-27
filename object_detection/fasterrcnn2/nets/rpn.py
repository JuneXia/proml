import torch
from torch import nn
from torch.nn import functional as F


# def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
#     # 计算网格中心点
#     shift_x = np.arange(0, width * feat_stride, feat_stride)
#     shift_y = np.arange(0, height * feat_stride, feat_stride)
#     shift_x, shift_y = np.meshgrid(shift_x, shift_y)
#     shift = np.stack((shift_x.ravel(),shift_y.ravel(),
#                       shift_x.ravel(),shift_y.ravel(),), axis=1)
#
#     # 每个网格点上的9个先验框
#     A = anchor_base.shape[0]
#     K = shift.shape[0]
#     anchor = anchor_base.reshape((1, A, 4)) + \
#              shift.reshape((K, 1, 4))
#     # 所有的先验框
#     anchor = anchor.reshape((K * A, 4)).astype(np.float32)
#     return anchor


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, n_anchors):
        super(RegionProposalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.score = nn.Conv2d(out_channels, n_anchors * 2, kernel_size=1, stride=1, padding=0)
        self.locate = nn.Conv2d(out_channels, n_anchors * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, _, hh, ww = x.shape
        shared_map = F.relu(self.conv1(x))

        rpn_bbox = F.relu(self.locate(shared_map))
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # 第一个n对应的是batch_size, 第二个-1对应的是anchors数量， 第3个4对应的bbox的4个坐标值

        rpn_class = F.relu(self.score(shared_map))
        rpn_class = rpn_class.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # 第一个n对应的是batch_size, 第二个-1对应的是anchors数量， 第3个2对应的是两个类别(前景或后景)
        rpn_prob = F.softmax(rpn_class, dim=-1)

        # TODO: rpn_class 实际上可以不要的，但如果要的话实际上还应该要经过argmax进行处理
        return rpn_class, rpn_prob, rpn_bbox


if __name__ == '__main__':
    shared_map = torch.zeros((3, 1024, 8, 8))
    rpn_net = RegionProposalNetwork(1024, 512, 9)
    rpn_net(shared_map)