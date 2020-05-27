# -*- coding: utf-8 -*-
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
cls_label = {"no": 0, "yes": 1}


class MaskFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str or list 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        if isinstance(data_dir, str):
            data_dir = [data_dir]

        self.label_name = {"no": 0, "yes": 1}

        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = list()
        for dat_dir in data_dir:
            self.data_info.extend(self.get_img_info(dat_dir))

        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                if sub_dir not in cls_label.keys():
                    continue

                img_names = os.listdir(os.path.join(root, sub_dir))
                # img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))  # 保留后缀为.jpg的文件，其他的剔除

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = cls_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


class FaceidDatasetFormat(Dataset):
    def __init__(self, format_data, transform=None):
        if isinstance(format_data, tuple):
            image_list, label_list = format_data
        else:
            raise Exception('format_data unspported type!')

        self.data_info = list()
        for impath, imlable in zip(image_list, label_list):
            impath = impath.replace('/disk2/res', '/disk2/res/face')
            self.data_info.append((impath, int(imlable)))

        # self.label_name = {"no": 0, "yes": 1}
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)
