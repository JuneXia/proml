import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tools.my_dataset import RMBDataset
import random
from libml.utils.config import SysConfig


# flag = True
flag = False
if flag:
    max_epoches = 3

    index_list = [1, 5, 78, 9, 68]
    # sampler = torch.utils.data.SequentialSampler(index_list)

    sampler = torch.utils.data.RandomSampler(index_list, replacement=False, num_samples=None)

    for epoch in range(max_epoches):
        print('epoch', epoch, end=':\t')
        for idx in sampler:
            print(idx, end=' ')
        print('')


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# flag = True
flag = False
if flag:
    max_epoches = 3
    batch_size = 10
    split_dir = os.path.join(SysConfig['home_path'], 'res', 'RMB_data/rmb_split')
    train_dir = os.path.join(split_dir, "train")
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
    n_train = len(train_data)
    split = n_train // 3
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])

    # 直接迭代train_sampler采样器，得到的是一个个索引值
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for data in train_sampler:
    #     print(data)
    # print('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    for epoch in range(max_epoches):
        print('epoch', epoch, end=':\t')
        for data in train_sampler:
            print(data)
        print('')


# flag = True
flag = False
if flag:
    max_epoches = 3
    batch_size = 10
    split_dir = os.path.join(SysConfig['home_path'], 'res', 'RMB_data/rmb_split')
    train_dir = os.path.join(split_dir, "train")
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)

    # ============================ 为每张图片赋予一个权重 ============================
    # 这里的权重与实际大小无关，只与相互之间的比值有关
    weights = []
    for _, label in train_data:
        if label == 0:
            weights.append(1)
        else:
            weights.append(2)

    n_train = len(train_data)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, n_train + 2, replacement=True)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    # 直接迭代train_sampler采样器，得到的是一个个索引值
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # for data in train_sampler:
    #     print(data)
    # print('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    actual_labels = []
    for epoch in range(max_epoches):
        print('epoch', epoch)
        for i, data in enumerate(train_sampler):
            # print(i, data[1])
            print(i, data)
            # actual_labels.extend(data[1].numpy().tolist())
        print('')

    print('0/1: {}'.format(actual_labels.count(0) / actual_labels.count(1)))


flag = True
# flag = False
if flag:
    max_epoches = 3
    batch_size = 10
    split_dir = os.path.join(SysConfig['home_path'], 'res', 'RMB_data/rmb_split')
    train_dir = os.path.join(split_dir, "train")
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)

    # ============================ 为每张图片赋予一个权重 ============================
    # 这里的权重与实际大小无关，只与相互之间的比值有关
    weights = []
    for _, label in train_data:
        if label == 0:
            weights.append(1)
        else:
            weights.append(2)

    n_train = len(train_data)

    index_list = [1, 5, 78, 9, 68]
    # sampler = torch.utils.data.SequentialSampler(list(range(len(train_data))))
    sampler = torch.utils.data.RandomSampler(list(range(len(train_data))), replacement=False, num_samples=None)
    train_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=3, drop_last=False)

    # 如果使用batch_sampler，则DataLoader的batch_size需要设置为1，shuffle需设置为False
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, batch_sampler=train_sampler)

    # 直接迭代train_sampler采样器，得到的是一个个索引值
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for data in train_sampler:
        print(data)
    print('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    actual_labels = []
    for epoch in range(max_epoches):
        print('epoch', epoch)
        for i, data in enumerate(train_loader):
            print(i, data[1])
        print('')

    print('0/1: {}'.format(actual_labels.count(0) / actual_labels.count(1)))



