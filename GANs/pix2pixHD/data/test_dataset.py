import sys
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader

torch.cuda.set_device(3)

sys.argv = ['train.py',
            '--nThreads', '1',
            '--instance_feat',
            ]
print(sys.argv)

opt = TrainOptions().parse()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

for data in dataset:
    print('debug')
