# -*- coding:utf-8 -*-
"""
@file name  : lesson-02.py
@author     : tingsongyu
@date       : 2018-08-26
@brief      : 张量的创建
"""
import torch
import numpy as np
# torch.manual_seed(1)

# ===============================  exmaple 1 ===============================
# 通过torch.tensor创建张量
if False:
    arr = np.ones((3, 3))
    print(arr)
    print("ndarray的数据类型：", arr.dtype)

    t = torch.tensor(arr)  # 存储在cpu上
    print(t)

    # 将从arr创建tensor，并将其搬运到gpu上，注意这可能会耗费一些时间
    t = torch.tensor(arr, device='cuda')
    print(t)


# ===============================  exmaple 2 ===============================
# 通过torch.from_numpy创建张量
if False:
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    t = torch.from_numpy(arr)
    print("numpy array: ", arr)
    print("ndarray的数据类型：", arr.dtype)
    print("tensor : ", t)
    print("tensor的数据类型：", t.dtype)

    # print("\n修改arr")
    # arr[0, 0] = 0
    # print("numpy array: ", arr)
    # print("tensor : ", t)

    print("\n修改tensor")
    t[0, 0] = -1
    print("numpy array: ", arr)
    print("tensor : ", t)


# ===============================  exmaple 3 ===============================
# 通过torch.zeros创建张量
if False:
    out_t = torch.tensor([0])  # 先随便创建一个tensor

    t = torch.zeros((3, 3), out=out_t)

    print('t: ', t)
    print('out_t', out_t)
    print('t内存地址：', id(t))
    print('out_t内存地址：', id(out_t))
    print(id(t) == id(out_t))


# ===============================  exmaple 4 ===============================
# 通过torch.full创建全1张量
if False:
    t1 = torch.full((3, 3), 2)
    print(t1, '\n', t1.dtype)

    t2 = torch.full_like(t1, 3)
    print(t2, '\n', t2.dtype)


# ===============================  exmaple 5 ===============================
# 通过torch.arange创建等差数列张量
if False:
    t = torch.arange(2, 10, 2)
    print(t)


# ===============================  exmaple 6 ===============================
# 通过torch.linspace创建均分数列张量
if False:
    t = torch.linspace(2, 10, 5)
    print(t)


if False:
    t = torch.logspace(2, 10, 5, base=2)
    print(t)


if False:
    t = torch.eye(2)
    print(t)

    t = torch.eye(2, 2)
    print(t)

    t = torch.eye(2, 3)
    print(t)


# ===============================  exmaple 7 ===============================
# 通过torch.normal创建正态分布张量
# mean：张量 std: 张量
mean = torch.arange(1, 5, dtype=torch.float)
std = torch.arange(1, 5, dtype=torch.float)
t_normal = torch.normal(mean, std)
print("mean:{}\nstd:{}".format(mean, std))
print(t_normal)

# mean：标量 std: 标量
t_normal = torch.normal(0., 1., size=(4,))
print(t_normal)

# mean：张量 std: 标量
mean = torch.arange(1, 5, dtype=torch.float)
std = 1
t_normal = torch.normal(mean, std)
print("mean:{}\nstd:{}".format(mean, std))
print(t_normal)











