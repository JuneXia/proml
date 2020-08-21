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

    'backbone': 'mobilenet_v1',  # 默认值： mobilenet_v2, 其他：mobilenet_v1
    'max_epoch': 250,
    'batch_size': 32,
    'device': 'cuda:1',  # cuda:0
    'epoch': 250,
    'grid_size': (3, 3),
}

if SysConfig['host_name'] == 'xjhomewin':
    cfg_net['data_path'] = 'F:\\xiajun\\res\\VOC\\VOC2007\\VOCtrainval'
    cfg_net['train_data_path'] = os.path.join(SysConfig["home_path"], SysConfig["proml_path"], "object_detection/fasterrcnn2/VOCdevkit")
elif SysConfig['host_name'] == 'DESKTOP-G2VDM3M':
    cfg_net['data_path'] = 'F:\\res\\VOC\\VOC2007\\VOCtrainval'
    cfg_net['train_data_path'] = os.path.join(SysConfig["home_path"], SysConfig["proml_path"], "object_detection/fasterrcnn2/VOCdevkit")