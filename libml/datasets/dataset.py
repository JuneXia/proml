# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from libml.utils import tools
import jpeg4py as jpeg
import time

random.seed(1)

DEBUG = True

class Dataset1(Dataset):
    def __init__(self, data_dir, transform=None):
        if isinstance(data_dir, str):
            data_dir = [data_dir]

        self.time_dict = dict()
        self.time_dict['imread'] = 0
        self.time_dict['transform'] = 0
        self.time_dict['imopen'] = 0
        self.time_dict['jpeg'] = 0
        self.time_dict['fromarray'] = 0

        self.images_count = 0

        self.label_name = {"no": 0, "yes": 1}

        self.data_info = list()
        for dat_dir in data_dir:
            self.data_info.extend(self.get_img_info(dat_dir))

        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        t1 = time.time()
        # img = Image.open(path_img).convert('RGB')     # 0~255
        img = self.pil_loader(path_img)
        t2 = time.time()
        self.time_dict['imread'] += (t2 - t1)

        t1 = time.time()
        if self.transform is not None:
            img = self.transform(img)
        t2 = time.time()
        self.time_dict['transform'] += (t2 - t1)

        self.images_count += 1
        if self.images_count >= 128:
            for k, v in self.time_dict.items():
                print('{}: {:.4f}'.format(k, v), end=' ')
                self.time_dict[k] = 0
            print('')
            self.images_count = 0

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(root):
        data_info = list()
        print('Loading data dir: {}'.format(root))
        dirs = os.listdir(root)
        for label, sub_dir in enumerate(dirs):
            img_names = os.listdir(os.path.join(root, sub_dir))
            # img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))  # 保留后缀为.jpg的文件，其他的剔除

            for i in range(len(img_names)):
                img_name = img_names[i]
                path_img = os.path.join(root, sub_dir, img_name)
                data_info.append((path_img, int(label)))
            tools.view_bar('loading: ', label + 1, len(dirs))
            if label > 1000:
                break
        print('')

        return data_info

    def jpeg4py_loader(self, path):
        t1 = time.time()
        with open(path, 'rb') as f:
            t2 = time.time()
            self.time_dict['imopen'] += (t2 - t1)

            t1 = time.time()
            img = jpeg.JPEG(f).decode()
            t2 = time.time()
            self.time_dict['jpeg'] += (t2 - t1)

            t1 = time.time()
            img = Image.fromarray(img)
            t2 = time.time()
            self.time_dict['fromarray'] += (t2 - t1)
            return img

    def pil_loader(self, path):
        t1 = time.time()
        with open(path, 'rb') as f:
            t2 = time.time()
            self.time_dict['imopen'] += (t2 - t1)

            t1 = time.time()
            img = Image.open(f)
            t2 = time.time()
            self.time_dict['jpeg'] += (t2 - t1)

            t1 = time.time()
            img = img.convert('RGB')
            t2 = time.time()
            self.time_dict['fromarray'] += (t2 - t1)

            return img

