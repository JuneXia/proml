# config.py
import os
from libml.utils.config import SysConfig

cfg_net = {
    # win
    # 'apath': 'C:\\Users\\Administrator\\res\\mnist\\MNIST_IMG',
    # 'bpath': 'C:\\Users\\Administrator\\res\\mnist\\MNIST_BASE',

    # 250 ps
    'apath': '/home/tangni/res/mnist/MNIST_IMG',
    'bpath': '../../data/mnist/MNIST_BASE',
    'grid_template': '../../data/mnist/grid_template_6x6.jpg',

    'backbone': 'mobilenet_v1',  # 默认值： mobilenet_v2, 其他：mobilenet_v1
    'max_epoch': 250,
    'batch_size': 32,
    'device': 'cuda:1',  # cuda:0
    'epoch': 250,
    'grid_size': (4, 4),
}

if SysConfig['host_name'] in ['SC-202002221016']:
    # cfg_net['grid_template'] = 'C:\\Users\\Administrator\\res\\grid_template.png'
    cfg_net['apath'] = 'C:\\Users\\Administrator\\res\\mnist\\MNIST_IMG'
    # cfg_net['bpath'] = '../../data/mnist/MNIST_BASE'
elif SysConfig['host_name'] in ['mt']:
    # cfg_net['grid_template'] = '/home/tangni/res/grid_template.jpg'
    cfg_net['apath'] = '/home/tangni/res/mnist/MNIST_IMG'
    # cfg_net['bpath'] = '../../data/mnist/MNIST_BASE'
