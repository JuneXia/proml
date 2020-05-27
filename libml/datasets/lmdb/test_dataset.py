import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from libml.datasets.lmdb.dataset import ImageFolderLMDB
from libml.datasets.data_loader import DataLoaderX
import time


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
input_size = 160
margin_scale = 182/160
# DATA_DIR = '/disk2/res/face/Trillion Pairs/train_msra/msra'
DATA_DIR = '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44'
# DATA_DIR = '/disk2/res/face/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44'
BATCH_SIZE = 128


train_transform = transforms.Compose([
    # transforms.Resize((182, 182)),
    # transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale))),
    # transforms.RandomApply([transforms.Resize((30, 30)), transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale)))], p=0.1),
    # transforms.RandomCrop(160),
    # transforms.RandomCrop(input_size),
    # transforms.RandomRotation((10), expand=True),

    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), shear=(-10, 10, -10, 10)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 45, 0, 0)),

    # transforms.RandomGrayscale(p=0.1),

    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),

    # transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# local_rank = 0
# cudnn.benchmark = False
# cudnn.deterministic = True
# torch.manual_seed(local_rank)
# torch.set_printoptions(precision=10)
#
# torch.cuda.set_device(0)
#
# os.environ.setdefault("MASTER_ADDR", "10.10.2.199")
# os.environ.setdefault("MASTER_PORT", "10022")
# torch.distributed.init_process_group(backend='nccl', world_size=4, rank=local_rank)
# world_size = torch.distributed.get_world_size()


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    # targets_ = tuple()
    # for t in targets:
    #     target = dict()
    #     for k, v in t.items():
    #         if v is None:
    #             target[k] = v
    #         else:
    #             target[k] = torch.tensor(v)
    #     targets_ += (target, )
    return torch.stack(images, 0).to(device), torch.tensor(targets, device=device)


if __name__ == '__main__':
    db_path = '/disk2/res/face/VGGFace2-mtcnn-align182x182margin44.lmdb'
    train_data = ImageFolderLMDB(db_path, transform=train_transform, target_transform=None)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = DataLoaderX(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=30, pin_memory=False, collate_fn=None)

    time_dict = dict()
    t1 = time.time()
    for i, data in enumerate(train_loader):
        t2 = time.time()
        print('DataLoad {}: {:.4f}'.format(i, t2 - t1))
        inputs, labels = data
        time.sleep(0.4)

        # if i > 20 and t2 - t1 > 10:
        #     print('\tsleep 60')
        #     time.sleep(60)

        # 返回单个图片的可视化方法
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # img_tensor = inputs[0, ...]  # C H W
        # img = transform_invert(img_tensor, train_transform)
        # img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('show', img)
        # cv2.waitKey()
        t1 = time.time()


    # prefetcher = data_prefetcher(train_loader)
