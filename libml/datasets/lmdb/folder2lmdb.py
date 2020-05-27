# -*- coding: utf-8 -*-
import os
import sys
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import lmdb
import pyarrow as pa


def raw_reader(path):
    if not path.endswith('.png'):
        print('debug')

    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(folder_path, lmdb_path, map_size=None, write_frequency=5000, num_workers=16):
    print("Loading dataset from %s" % folder_path)
    dataset = ImageFolder(folder_path, loader=raw_reader)
    size_of_img = len(next(iter(dataset))[0])  # the size of per image
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)
    assert (len(dataset), len(data_loader))

    print("Generate LMDB to %s" % lmdb_path)
    isdir = os.path.isdir(lmdb_path)
    if map_size is None:
        map_size = (size_of_img + 1) * len(dataset)  # 加1表示的是用于存储label的存储空间。
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, help='the folder path.')
parser.add_argument('--lmdb_path', type=str, help='lmdb save path.', default='./')
parser.add_argument('--procs', type=int, help='the number of multiprocess.', default=8)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        sys.argv = ['folder2lmdb.py',
                    '--folder_path', '/disk1/home/xiaj/res/face/VGGFace2/Experiment/tmp',
                    '--lmdb_path', '/disk1/home/xiaj/res/face/example.lmdb',
                    ]

    print(sys.argv)
    args = parser.parse_args()

    folder2lmdb(args.folder_path, args.lmdb_path, num_workers=args.procs)
