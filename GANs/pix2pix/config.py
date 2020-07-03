# config.py
import os
from libml.utils.config import SysConfig

cfg_net = {
    'backbone': 'mobilenet_v1',  # 默认值： mobilenet_v2, 其他：mobilenet_v1
    'max_epoch': 250,
    'batch_size': 32,
    'device': 'cuda:1',  # cuda:0
    'epoch': 250,
}

if SysConfig['host_name'] in ['mt1080', 'ps']:
    cfg_net['data_path'] = '/home/tangni/res/CMP_facade/CMP_facade_DB_base/base'
    cfg_net['valid_path'] = '/home/tangni/res/CMP_facade/CMP_facade_DB_extended/extended'
