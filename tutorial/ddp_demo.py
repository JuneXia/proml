# -*- coding: utf-8 -*-
# @Time    : 2020/6/16
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : ddp_demo.py

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel


input_size = 5
output_size = 5
batch_size = 40
data_size = 900

gpu_ids = [0,1,2,3]
# nodes = 1
world_size = len(gpu_ids) #* nodes

flag_ddp = True
device = torch.device(gpu_ids[0])

if flag_ddp:
    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")#, world_size=world_size)

    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(device)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

dataset = RandomDataset(input_size, data_size)
# **** DDP ****
if flag_ddp:
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    sampler = DistributedSampler(dataset, num_replicas=world_size)
else:
    sampler = None
# 3）使用DistributedSampler
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size * world_size,
                         num_workers=48,
                         # pin_memory=True,
                         sampler=sampler)
print(" * shape ", next(iter(rand_loader)).shape)

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        model = [nn.Linear(input_size, 128),
                 nn.Linear(128, 256),
                 # nn.Linear(256, 512),
                 # nn.Linear(512, 1024),
                 # nn.Linear(1024, 2048),
                 nn.Linear(256, output_size)
                 ]

        self.fc = nn.Sequential(*model)

    def forward(self, input):
        print(' * inputdeivce, ', input.device)
        output = self.fc(input)
        print(' * output, ', output.device)

        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.net = Net(input_size, output_size)
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        out = self.net(x)

        loss = self.l1(out, y)
        print(' * loss: ', loss.device)
        return loss

class OptimManger():
    def __init__(self, ddpmodule):
        if type(ddpmodule) == DistributedDataParallel or type(ddpmodule) == nn.DataParallel:
            module = ddpmodule.module
        else:
            module = ddpmodule
        self.optim = torch.optim.Adam(module.net.parameters(), lr=0.01)

    def optimize_parameters(self, loss):
        self.optim.zero_grad()
        loss.mean().backward()
        self.optim.step()


model = Model(input_size, output_size)
# self.optim = torch.optim.Adam(self.net.parameters(), lr=0.01)

# 4) 封装之前要把模型移到对应的gpu
model.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    if flag_ddp:
        model = DistributedDataParallel(model, device_ids=gpu_ids)
    else:
        model = nn.DataParallel(model, device_ids=gpu_ids)
optim = OptimManger(model)

for data in rand_loader:
    input_var = data
    print(" input shape", input_var.shape)
    target = torch.ones_like(input_var)
    output = model(input_var, target)
    optim.optimize_parameters(output)
    print(" * Outside: input size", input_var.size(), "output_size", output.size)

'''
Let's use 4 GPUs!
 * shape,  torch.Size([23, 5])
 * inputdeivce,  cuda:0
 * inputdeivce,  cuda:1
 * inputdeivce,  cuda:2
 * inputdeivce,  cuda:3
 * output,  cuda:0
  In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
 * output,  cuda:1
  In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
 * output,  cuda:2
  In Model: input size torch.Size([6, 5]) output size torch.Size([6, 2])
 * output,  cuda:3
  In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
 * input_var cuda:0
 * device cuda:0
 
 python -m torch.distributed.launch
'''