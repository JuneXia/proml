import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image


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


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:  # TODO: what?
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)s
        A_path = self.A_paths[index]              
        A = Image.open(A_path)

        if DEBUG:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            arr = np.array(A)
            arr2txt(arr, 'src_A.txt')
            print("A.size: {}, arr.shape: {}".format(A.size, arr.shape))
            print("min,max,mean: ", arr.min(), arr.max(), arr.mean())
            tmp_A = A.resize((64, 32), Image.NEAREST)
            tmp_arr = np.array(tmp_A)
            arr2txt(tmp_arr, 'resized_A.txt')
            print("tmp_A.size: {}, tmp_arr.shape: {}".format(tmp_A.size, tmp_arr.shape))
            print("min,max,mean: ", tmp_arr.min(), tmp_arr.max(), tmp_arr.mean())
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # cv2.imshow('show', tmp_arr)
            # cv2.waitKey()
            # cv2.waitKey()

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')

            if DEBUG:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                arr = np.array(B)
                arr2txt(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 'src_B.txt')
                print("B.size: {}, arr.shape: {}".format(B.size, arr.shape))
                print("min,max,mean: ", arr.min(), arr.max(), arr.mean())
                tmp_B = B.resize((64, 32), Image.NEAREST)
                tmp_arr = np.array(tmp_B)
                arr2txt(cv2.cvtColor(tmp_arr, cv2.COLOR_RGB2GRAY), 'resized_B.txt')
                print("tmp_B.size: {}, tmp_arr.shape: {}".format(tmp_B.size, tmp_arr.shape))
                print("min,max,mean: ", tmp_arr.min(), tmp_arr.max(), tmp_arr.mean())
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                # cv2.imshow('show', tmp_arr)
                # cv2.waitKey()

            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:  # TODO: cityscapes 数据集的 train_inst 图片是使用 16位格式存的png图，Image、OpenCV 无法读这类图，可使用png读16位图
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)

            if DEBUG:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                arr = np.array(inst)
                arr2txt(arr, 'src_inst.txt')
                print("inst.size: {}, arr.shape: {}".format(inst.size, arr.shape))
                print("min,max,mean: ", arr.min(), arr.max(), arr.mean())
                tmp_inst = inst.resize((64, 32), Image.NEAREST)
                tmp_arr = np.array(tmp_inst)
                arr2txt(tmp_arr, 'resized_inst.txt')
                print("tmp_inst.size: {}, tmp_arr.shape: {}".format(tmp_inst.size, tmp_arr.shape))
                print("min,max,mean: ", tmp_arr.min(), tmp_arr.max(), tmp_arr.mean())
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                # cv2.imshow('show', tmp_arr)
                # cv2.waitKey()

            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'