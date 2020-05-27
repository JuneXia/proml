from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
import sys

FILE_ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_ROOT_PATH)


from face_detection.retinaface.data import cfg_mnet, cfg_re50
from face_detection.retinaface.layers.functions.prior_box import PriorBox
from face_detection.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
# from utils.nms.py_cpu_nms import py_cpu_nms

from face_detection.retinaface.models.retinaface import RetinaFace
from face_detection.retinaface.utils.box_utils import decode, decode_landm
# from utils.box_utils import decode, decode_landm\

from face_detection.base_detector import BaseDetector
from libml.utils import tools

# if 'flask' in sys.argv[0].rsplit('/', 1)[-1]:
#     from data import cfg_mnet, cfg_re50
#     from layers.functions.prior_box import PriorBox
#     from .utils.nms.py_cpu_nms import py_cpu_nms
#
#     from models.retinaface import RetinaFace
#     from .utils.box_utils import decode, decode_landm
# elif 'face_detection' in sys.argv[0].rsplit('/', 1)[-1]:
#     from data import cfg_mnet, cfg_re50
#     from layers.functions.prior_box import PriorBox
#     from .utils.nms.py_cpu_nms import py_cpu_nms
#
#     from models.retinaface import RetinaFace
#     from .utils.box_utils import decode, decode_landm


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Detector(BaseDetector):
    def __init__(self, device=None):
        super(Detector, self).__init__()
        self.cpu = False
        if device is not None:
            assert isinstance(device, str)
            if 'cpu' in device:
                self.cpu = True  # TODO： 为了适配历史逻辑
        else:
            device = 'cuda:0'

        self.network = 'resnet50'
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.keep_top_k = 750
        self.nms_threshold = 0.4
        self.vis_thres = 0.7  # 默认0.6

        self.cfg = None
        if self.network == 'mobile0.25':
            self.cfg = cfg_mnet
            model_path = os.path.join(FILE_ROOT_PATH, 'weights', 'mobilenet0.25_Final.pth')
        elif self.network == 'resnet50':
            self.cfg = cfg_re50
            model_path = os.path.join(FILE_ROOT_PATH, 'weights', 'Resnet50_Final.pth')
        else:
            raise Exception('目前只支持上述模型！')
        self.trained_model = model_path

        torch.set_grad_enabled(False)

        net = RetinaFace(cfg=self.cfg, phase='test')
        # net = load_model(net, self.trained_model, self.cpu)
        net = load_model(net, self.trained_model, True)
        net.eval()
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.cpu else device)
        self.net = net.to(self.device)

        self.resize = 1

    # def __name__(self):
    #     return 'RetinaFace.' + self.network

    def imread(self, impath):
        image = tools.imread(image_path, flags=1)
        return image

    def imwrite(self, image, impath):
        cv2.imwrite(impath, image)

    def detecting(self, image):
        """
        :param image: bgr image
        :return: return ndarray with shape[?, 15], 其中[?, 0:4]为人脸bbox, 依次是坐标(x1,y1),(x2,y2);
                                                   其中[?, 4:15]为关键点坐标，依次是坐标(x1,y1),(x2,y2),(x3,y4), ...
        """
        image = np.float32(image)

        im_height, im_width, _ = image.shape
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(image)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, self.nms_threshold,force_cpu=self.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        dets = dets[dets[:, 4] >= self.vis_thres]

        return dets


def show(image, dets):
    if type(image) == str:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elif type(image) == np.ndarray:
        pass
    else:
        raise Exception('aa')

    for b in dets:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        cv2.imshow('show', image)
        cv2.waitKey()
        print('debug')

    name = "test.jpg"
    cv2.imwrite(name, image)
    cv2.imshow('PyTorch-Retinaface', image)
    cv2.waitKey()


"""
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(1):  # TODO
        image_path = args.image_path
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)
            cv2.imshow('PyTorch-Retinaface', img_raw)
            cv2.waitKey()
"""


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(sys.argv)
# sys.argv = ['detect.py', '--trained_model', './weights/Resnet50_Final.pth']
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--image_path', type=str, default='./curve/test.jpg', help='Path for image to detect')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

'相对导入适用于你最终要放入包中的代码。如果你编写了很多相关性强的代码，' \
'那么应该采用这种导入方式。你会发现PyPI上有很多流行的包也是采用了相对导入。' \
'还要注意一点，如果你想要跨越多个文件层级进行导入，只需要使用多个句点即可。' \
'不过，PEP 328建议相对导入的层级不要超过两层。'


'当你在局部作用域中导入模块时，你执行的就是局部导入。' \
'如果你在Python脚本文件的顶部导入一个模块，那么你就是在将该模块导入至全局作用域，' \
'这意味着之后的任何函数或方法都可能访问该模块。例如：'


if __name__ == '__main__':
    detector = Detector()

    image_path = './data/test.jpg'
    image_path = '/disk1/home/xiaj/res/face/maskface/test4.jpg'
    if len(sys.argv) > 1:
        image_path = args.image_path
    image = detector.imread(image_path)
    dets = detector.detecting(image)
    show(image_path, dets)
