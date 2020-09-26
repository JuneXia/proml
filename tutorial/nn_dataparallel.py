"""
References: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np


# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_ids = [7, 9]

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)

        # ************************************************
        # NOTE: 多卡训练时，forward中创建tensor该to到哪块卡上：
        # ************************************************
        # noise = Variable(torch.ones(output.shape))  # 默认在cpu上
        # noise = Variable(torch.ones(output.shape)).type(torch.cuda.FloatTensor)  # 默认to到cuda0
        # noise = Variable(torch.ones(output.shape)).cuda()  # 默认to到cuda0
        # noise = Variable(torch.ones(output.shape)).cuda(gpu_ids[0])  # 当只有gpu_ids中只有一块卡时是ok的，当gpu_ids有多块卡时失败，因为多卡是并行计算的，都to到一块卡上显然不行
        noise = Variable(torch.ones(output.shape)).cuda(output.device)  # ok
        # ************************************************

        output += noise
        print("\tIn Model: input size", input.size(),
              "output size", output.size(),
              "\tinput.device", input.device,
              "output.device", output.device,
              "\tnoise.device", noise.device)

        # NOTE: 返回值
        # return output  # 返回一个tensor OK
        # return output, noise  # 返回多个tensor OK
        # return output, {'output': output, 'noise': noise}  # 返回值有tensor，也有dict，OK
        # return output, {'image': {'output': output, 'noise': noise}}  # 返回的dict具有多层包装，OK
        return output, {'image': {'output': output, 'noise': noise}, 'scalar': noise.sum()}  # 返回的dict具有多层包装，OK


if __name__ == '__main__1':
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = False

    model = Model(input_size, output_size)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.to(gpu_ids[0])

    for data in rand_loader:
        # input = data.to(gpu_ids[0])  # ok
        input = data  # ok
        output, noise = model(input)
        print("Outside: input size", input.size(),
              "output_size", output.size(),
              "output.device", output.device)


if __name__ == '__main__':
    loss1 = np.random.uniform(1, 2, 8)
    loss2 = np.random.uniform(1, 2, 8)

    total_loss = loss1.mean() + loss2.mean()
    print('method 1: ', total_loss)

    loss1_1 = loss1[0:3].mean()
    loss1_2 = loss1[3:6].mean()
    loss1_3 = loss1[6:].mean()

    loss2_1 = loss2[0:3].mean()
    loss2_2 = loss2[3:6].mean()
    loss2_3 = loss2[6:].mean()

    total_loss1 = loss1_1 + loss2_1
    total_loss2 = loss1_2 + loss2_2
    total_loss3 = loss1_3 + loss2_3
    total_loss = (total_loss1 + total_loss2 + total_loss3)/3
    print('method 2: ', total_loss)

    print('debug')

