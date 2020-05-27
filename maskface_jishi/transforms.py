"""
reference: Pytorch_Retinaface/data/data_augment.py
"""
import cv2
import numpy as np
import random
import numbers
from PIL import Image
import torch
from torchvision.transforms import functional as F
# from torchvision.transforms import transforms
from utils.box_utils import matrix_iof

DEBUG_SHOW = False

class DetectionTransform(object):
    def __init__(self):
        pass

    def check_targets(self, targets):
        assert isinstance(targets, dict)


class Crop(object):
    """Crops the given PIL Image at the center.
    reference: torchvision.transforms.transforms.py > CenterCrop
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
        """
        img_ = np.array(img)
        height, width, _ = img_.shape
        pad_image_flag = True

        boxes = target['boxes']
        labels = target['labels']
        landmarks = target['landmarks']

        if DEBUG_SHOW:
            image_ = img_.copy()
            for i in range(len(boxes)):
                box = boxes[i].astype(np.int)
                cv2.rectangle(image_, (box[0], box[1]), (box[2], box[3]), (255, 255, 0))
            cv2.imshow('srcimg', image_)
            cv2.waitKey()

        for _ in range(250):
            """
            if random.uniform(0, 1) <= 0.2:
                scale = 1.0
            else:
                scale = random.uniform(0.3, 1.0)
            """
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                l = 0
            else:
                l = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            value = matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()

            landms_t = None
            if landmarks:
                landms_t = landmarks[mask_a].copy()
                landms_t = landms_t.reshape([-1, 5, 2])

            if boxes_t.shape[0] == 0:
                continue

            image_t = img_[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            # landmarks
            if landmarks:
                landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
                landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
                landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
                landms_t = landms_t.reshape([-1, 10])

            # make sure that the cropped image contains at least one face > 16 pixel at training image scale
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * self.size[1]
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * self.size[0]
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]

            if landmarks:
                landms_t = landms_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue

            pad_image_flag = False

            img = Image.fromarray(image_t)
            target['boxes'] = boxes_t
            target['labels'] = labels_t
            target['landmarks'] = landms_t
            return img, target

        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Distort(object):
    """
    reference: torchvision.transforms.transforms.py > RandomPerspective
    """

    def __init__(self):
        pass

    def __call__(self, image, target):
        """
        Args:
            img (PIL Image): Image to be distort transformed.

        Returns:
            PIL Image: distort transformed image.
        """
        if not F._is_pil_image(image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(image)))

        image = np.array(image)

        if random.randrange(2):

            # brightness distortion
            if random.randrange(2):
                self.convert(image, beta=random.uniform(-32, 32))

            # contrast distortion
            if random.randrange(2):
                self.convert(image, alpha=random.uniform(0.5, 1.5))

            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # saturation distortion
            if random.randrange(2):
                self.convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            # hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        else:

            # brightness distortion
            if random.randrange(2):
                self.convert(image, beta=random.uniform(-32, 32))

            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # saturation distortion
            if random.randrange(2):
                self.convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

            # hue distortion
            if random.randrange(2):
                tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                image[:, :, 0] = tmp

            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

            # contrast distortion
            if random.randrange(2):
                self.convert(image, alpha=random.uniform(0.5, 1.5))

        image = Image.fromarray(image)
        return image, target

    @staticmethod
    def convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Pad2Square(object):
    """
    reference: torchvision.transforms.transforms.py > Pad
    """
    def __init__(self, fill=0, padding_mode='constant'):
        # assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        # if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
        #     raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
        #                      "{} element tuple".format(len(padding)))

        # self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        width, height = img.size
        if width == height:
            return img, target

        img = np.array(img)
        long_side = max(width, height)
        img_ = np.empty((long_side, long_side, 3), dtype=img.dtype)
        img_[:, :] = self.fill
        img_[0:0 + height, 0:0 + width] = img
        img_ = Image.fromarray(img_)
        return img_, target

    # def __repr__(self):
    #     return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
    #         format(self.padding, self.fill, self.padding_mode)


class RandomMirror(object):
    """
    reference: torchvision.transforms.transforms.py > RandomHorizontalFlip
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            image = np.array(image)
            _, width, _ = image.shape
            image = image[:, ::-1]

            if 'boxes' in target.keys():
                boxes = target['boxes']

                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]

                target['boxes'] = boxes
            elif 'landmarks' in target.keys():
                landms = target['landmarks']

                landms = landms.copy()
                landms = landms.reshape([-1, 5, 2])
                landms[:, :, 0] = width - landms[:, :, 0]
                tmp = landms[:, 1, :].copy()
                landms[:, 1, :] = landms[:, 0, :]
                landms[:, 0, :] = tmp
                tmp1 = landms[:, 4, :].copy()
                landms[:, 4, :] = landms[:, 3, :]
                landms[:, 3, :] = tmp1
                landms = landms.reshape([-1, 10])

                target['landmarks'] = landms

            image = Image.fromarray(image)

        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Resize(object):
    """
    reference: https://www.zhihu.com/question/313095849
    """

    def __init__(self, min_size=600, max_size=1000, interpolation=Image.BILINEAR):
        assert isinstance(min_size, int) and isinstance(max_size, int)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        (H, W) = img.size
        scale1 = self.min_size / min(H, W)
        scale2 = self.max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img.resize((round(H * scale), round(W * scale)))  # 调整图片大小，长边小于1000，短边小于600

        if img.size[0] != self.min_size:
            print('debug')

        # 根据图片变化调整bbox顶点坐标
        if 'boxes' in target.keys():
            boxes = target['boxes']

            (o_H, o_W) = img.size
            y_scale = o_H / H
            x_scale = o_W / W
            boxes[:, 0] = y_scale * boxes[:, 0]
            boxes[:, 1] = x_scale * boxes[:, 1]
            boxes[:, 2] = y_scale * boxes[:, 2]
            boxes[:, 3] = x_scale * boxes[:, 3]

            target['boxes'] = boxes
        elif 'landmarks' in target.keys():
            pass

        return img, target


class ToTensor(object):
    def __call__(self, image, target):
        width, height = image.size
        image = F.to_tensor(image)  # 内部会做transpose((2, 0, 1))操作

        if 'boxes' in target.keys():
            target['boxes'][:, 0::2] /= width
            target['boxes'][:, 1::2] /= height
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float)

        # if 'landmarks' in target.keys():
        #     target['landmarks'][:, 0::2] /= width
        #     target['landmarks'][:, 1::2] /= height
        #     target['landmarks'] = torch.tensor(target['landmarks'], dtype=torch.float)

        if 'labels' in target.keys():
            target['labels'] = torch.tensor(target['labels'], dtype=torch.long)

        return image, target


def _crop(img, boxes, labels, landm, img_dim):
    img_ = np.array(img)
    height, width, _ = img_.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        landms_t = None
        if False:
            landms_t = landm[mask_a].copy()
            landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = img_[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        if False:
            landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
            landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
            landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
            landms_t = landms_t.reshape([-1, 10])

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        if False:
            landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        image_t = Image.fromarray(image_t)
        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return img, boxes, labels, landm, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = np.array(image)

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    image = Image.fromarray(image)
    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    if random.randrange(2):
        image = np.array(image)
        _, width, _ = image.shape
        image = image[:, ::-1]

        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        if landms:
            landms = landms.copy()
            landms = landms.reshape([-1, 5, 2])
            landms[:, :, 0] = width - landms[:, :, 0]
            tmp = landms[:, 1, :].copy()
            landms[:, 1, :] = landms[:, 0, :]
            landms[:, 0, :] = tmp
            tmp1 = landms[:, 4, :].copy()
            landms[:, 4, :] = landms[:, 3, :]
            landms[:, 3, :] = tmp1
            landms = landms.reshape([-1, 10])

        image = Image.fromarray(image)

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    image = np.array(image)
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    image_t = Image.fromarray(image_t)
    return image_t


def _resize_subtract_mean(image, boxes_t, insize, rgb_mean):
    image = np.array(image)

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]

    height, width, _ = image.shape
    y_scale = insize/height
    x_scale = insize/width
    boxes_t[:, 0] = y_scale * boxes_t[:, 0]
    boxes_t[:, 1] = x_scale * boxes_t[:, 1]
    boxes_t[:, 2] = y_scale * boxes_t[:, 2]
    boxes_t[:, 3] = x_scale * boxes_t[:, 3]

    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    # image = image.transpose(2, 0, 1)

    image = Image.fromarray(image.astype(np.uint8))
    return image, boxes_t


class Compose1(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Compose(object):
    """
    reference: Pytorch_Retinaface/data/data_augment.py > preproc
    """
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        global DEBUG_SHOW
        boxes = targets['boxes']
        labels = targets['labels']
        landm = None  # TODO: 临时借用下

        if DEBUG_SHOW:
            image_ = np.array(image).copy()

            # cv2.rectangle()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
            cv2.imshow('srcimage', image_)
            cv2.waitKey()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        if DEBUG_SHOW:
            image_ = np.array(image_t).copy()
            for i in range(len(boxes_t)):
                x1, y1, x2, y2 = boxes_t[i]
                cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

            cv2.imshow('cropimg', image_)
            cv2.waitKey()

        image_t = _distort(image_t)
        if DEBUG_SHOW:
            image_ = np.array(image_t).copy()
            cv2.imshow('distortimg', image_)
            cv2.waitKey()

        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        if DEBUG_SHOW:
            image_ = np.array(image_t).copy()
            for i in range(len(boxes_t)):
                x1, y1, x2, y2 = boxes_t[i]
                cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            cv2.imshow('padtosquareimg', image_)
            cv2.waitKey()

        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        if DEBUG_SHOW:
            image_ = np.array(image_t).copy()
            for i in range(len(boxes_t)):
                x1, y1, x2, y2 = boxes_t[i]
                cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.imshow('mirrorimg', image_)
            cv2.waitKey()

        width, height = image_t.size
        image_t, boxes_t = _resize_subtract_mean(image_t, boxes_t, self.img_dim, self.rgb_means)
        # boxes_t[:, 0::2] /= width
        # boxes_t[:, 1::2] /= height

        if DEBUG_SHOW:
            image_ = np.array(image_t).copy()
            for i in range(len(boxes_t)):
                x1, y1, x2, y2 = boxes_t[i]
                cv2.rectangle(image_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.imshow('resize', image_)
            cv2.waitKey()

        if landm:
            landm_t[:, 0::2] /= width
            landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)

        if landm:
            targets_t = np.hstack((boxes_t, landm_t, labels_t))
        else:
            targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t
