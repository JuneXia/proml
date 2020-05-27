# -*- coding: utf-8 -*-
import os
import sys
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from libml.utils import tools
import jpeg4py as jpeg
import multiprocessing
import threading
import lmdb
import six
import pyarrow as pa
import time

random.seed(1)

DEBUG = True


class ImageFolderLMDB(Dataset):
    """
    :reference: https://github.com/Lyken17/Efficient-PyTorch#data-loader
    """
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False, max_spare_txns=1)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

        self.time_dict = {'imload': 0, 'flow': 0, 'deserialize': 0, 'loadimage': 0, 'transform': 0,
                          'with.begin': 0, 'buf': 0, 'open.convert': 0, 'keys[index]': 0}
        self.count = 0
        self.time1 = time.time()

    def __getitem__(self, index):
        self.count += 1
        # proc = multiprocessing.current_process()
        # thread = threading.current_thread()
        # print('dataset: current process-name: {}, process-id: {}, thread-name: {}, thread-id: {}'.format(proc.name, proc.ident, thread.name, thread.ident))

        t1 = time.time()
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            self.time_dict['with.begin'] += time.time() - t1

            t1 = time.time()
            idx = self.keys[index]
            self.time_dict['keys[index]'] += time.time() - t1

            t1 = time.time()
            byteflow = txn.get(idx)
            self.time_dict['flow'] += time.time() - t1


        t1 = time.time()
        unpacked = pa.deserialize(byteflow)
        self.time_dict['deserialize'] += time.time() - t1

        # load image
        t1 = time.time()
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        self.time_dict['buf'] += time.time() - t1

        t1 = time.time()
        img = Image.open(buf).convert('RGB')
        self.time_dict['open.convert'] += time.time() - t1

        # load label
        t1 = time.time()
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        self.time_dict['transform'] += time.time() - t1

        if self.count >= 128:
            # for k, v in self.time_dict.items():
            #     print('{}: {:.4f}'.format(k, v), end=' ')
            #     self.time_dict[k] = 0
            # print('')

            # print('imload: {:.4f}'.format(time.time() - self.time1))
            self.count = 0
            self.time1 = time.time()

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        sys.argv = ['folder2lmdb.py',
                    '-f', '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44',
                    '-s', 'train']

    print(sys.argv)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
