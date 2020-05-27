import os
import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
import voc_parse


class MaskFaceAkouDataset(Dataset):
    def __init__(self, data_dir, transforms=None):

        self.data_dir = data_dir
        self.transforms = transforms
        # self.img_dir = os.path.join(data_dir, "data")
        # self.txt_dir = os.path.join(data_dir, "label")

        self.images_label = voc_parse.parse_voc(data_dir)

        # self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".png"), os.listdir(self.img_dir)))]

    def __getitem__(self, index):
        """
        返回img和target
        :param idx:
        :return:
        """

        impath, label = self.images_label[index]
        label = np.array(label)

        img = Image.open(impath).convert("RGB")
        # img = cv2.imread(impath)
        # boxes = torch.tensor(label[:, 0:4], dtype=torch.float)
        # labels = torch.tensor(label[:, 4], dtype=torch.long)

        target = dict()
        target["boxes"] = label[:, 0:4]
        target["labels"] = label[:, 4]
        target["landmarks"] = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if len(self.images_label) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.images_label)