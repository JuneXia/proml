import os.path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random


DEBUG = False
if DEBUG:
    import numpy as np
    import cv2
    from libml.utils.tools import strcat


def arr2txt(arr, save_file):
    assert isinstance(arr, np.ndarray), "arr must be np.ndarray"

    with open(save_file, 'w') as f:
        max_size = len(str(arr.max()))
        tmp_arr = arr.tolist()
        for ar in tmp_arr:
            a = [str(a).rjust(max_size, ' ') for a in ar]
            a = strcat(a, ' ')
            f.write(a + '\n')


class MnistPairDataset(Dataset):
    def __init__(self, cfg, transforms=None):
        self.cfg = cfg
        self.transforms = transforms

        self.apaths = list()
        self.bpaths = list()
        apath = cfg['apath']
        bpath = cfg['bpath']
        bpath_list = os.listdir(bpath)
        for b in bpath_list:
            bpth = os.path.join(bpath, b)
            label, _ = b.rsplit('.', maxsplit=1)
            apth = os.path.join(apath, label)
            apath_list = os.listdir(apth)[0:100]
            apaths = [os.path.join(apth, a) for a in apath_list]
            bpaths = len(apath_list) * [bpth]

            self.apaths.extend(apaths)
            self.bpaths.extend(bpaths)

        index = np.arange(0, len(self.apaths))
        random.shuffle(index)
        self.apaths = np.array(self.apaths)
        self.bpaths = np.array(self.bpaths)
        self.apaths = self.apaths[index].tolist()
        self.bpaths = self.bpaths[index].tolist()

    def __getitem__(self, index):
        # index = 0
        apath = self.apaths[index]
        bpath = self.bpaths[index]
        imga = Image.open(apath)
        imgb = Image.open(bpath)

        if self.transforms is not None:
            imga = self.transforms(imga)
            imgb = self.transforms(imgb)

        input_dict = {'imagea': imga, 'imageb': imgb}

        return input_dict

    def __len__(self):
        return len(self.apaths)


import cv2
if __name__ == '__main__':
    grid_size = (5, 5)
    template = np.ones((28, 28), dtype=np.uint8) * 128
    interval = 28 // grid_size[0]
    for i in range(interval + 1):
        print(i)
        start_i = i * interval
        template[start_i:start_i+1, :] = 255
        template[:, start_i:start_i+1] = 255
        cv2.imshow('show', template)
        cv2.waitKey()

    cv2.imwrite('./grid_template_6x6.jpg', template)