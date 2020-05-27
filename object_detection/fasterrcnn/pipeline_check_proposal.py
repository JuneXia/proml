import keras.layers as KL
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2

from utils import shapeData as dataSet
from config import Config

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


def building_block(filters, block):
    if block != 0:
        stride = 1
    else:
        stride = 2

    def f(x):
        y = KL.Conv2D(filters, (1, 1), strides=stride)(x)
        y = BatchNorm(axis=3)(y)
        y = KL.Activation("relu")(y)

        y = KL.Conv2D(filters, (3, 3), padding="same")(y)
        y = BatchNorm(axis=3)(y)
        y = KL.Activation("relu")(y)

        y = KL.Conv2D(4 * filters, (1, 1))(y)
        y = BatchNorm(axis=3)(y)

        if block == 0:
            shorcut = KL.Conv2D(4 * filters, (1, 1), strides=stride)(x)
            shorcut = BatchNorm(axis=3)(shorcut)
        else:
            shorcut = x
        y = KL.Add()([y, shorcut])
        y = KL.Activation("relu")(y)
        return y

    return f


def resNet_featureExtractor(inputs):
    x = KL.Conv2D(64, (3, 3), padding="same")(inputs)
    x = BatchNorm(axis=3)(x)
    x = KL.Activation("relu")(x)

    filters = 64
    blocks = [6, 6, 6]
    for i, block_num in enumerate(blocks):
        for block_id in range(block_num):
            x = building_block(filters, block_id)(x)
        filters *= 2
    return x


def rpn_net(inputs, k):
    shared_map = KL.Conv2D(256, (3, 3), padding="same")(inputs)
    shared_map = KL.Activation("linear")(shared_map)
    rpn_class = KL.Conv2D(2 * k, (1, 1))(shared_map)

    rpn_class = KL.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 2]))(rpn_class)
    # [tf.shape(x)[0], -1, 2], 第一个tf.shape(x)[0]对应的是batch_size, 第二个-1对应的是anchors数量，第3个2对应的是两个类别(前景或后景)，

    rpn_class = KL.Activation("linear")(rpn_class)
    rpn_prob = KL.Activation("softmax")(rpn_class)

    y = KL.Conv2D(4 * k, (1, 1))(shared_map)
    y = KL.Activation("linear")(y)

    rpn_bbox = KL.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]))(y)
    # [tf.shape(x)[0], -1, 4], 第一个tf.shape(x)[0]对应的是batch_size, 第二个-1对应的是anchors数量， 第3个4对应的bbox的4个坐标值

    return rpn_class, rpn_prob, rpn_bbox


def rpn_class_loss(rpn_match, rpn_class_logits):
    """
    :param rpn_match: rpn_match.shape = (b, num_anchors, 1)
           rpn_match 中存储的是每个anchor框是属于背景(-1)、前景(1)和无用anchor框(0)的标签，这些是根据 anchor 框和ground_truth的IoU出来的。
    :param rpn_class_logits: rpn_class_logits.shape = (b, num_anchors, 2)
    :return:
    """
    ## rpn_match (None, 576, 1)
    ## rpn_class_logits (None, 576, 2)

    # 挤压掉最后一个维度，现在rpn_match.shape = (b, num_anchors)
    rpn_match = tf.squeeze(rpn_match, -1)

    # 提取出rpn_match中所有的前后景anchor索引
    indices = tf.where(K.not_equal(rpn_match, 0))

    # 将rpn_match中所有非前景的标签都置为0
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)

    # 提取出indices索引所在的预测得分值
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)     ### prediction

    # 提取出indices索引所在的anchor框标签
    anchor_class = tf.gather_nd(anchor_class, indices)   ### target

    # 计算交叉熵损失
    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)

    # 如果loss长度大于0,则对loss取均值，否则loss值取为0
    loss = K.switch(tf.size(loss) > 0 , K.mean(loss), tf.constant(0.0))
    return loss


def batch_back(x, counts, num_rows):
    """
    对一个batch_size的数据进行打包：因为一个batch_size的数据中，每个batch中所包含的前景anchor数量是不一样的，所以这里要将它们打包到一起。
    :param x:
    :param counts:
    :param num_rows: TODO:这个应当是batch_size,
    :return:
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """
    :param target_bbox: input_rpn_bbox
    :param rpn_match: input_rpn_match
    :param rpn_bbox: rpn网络预测的bbox
    :return:
    """
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # 统计一个batch中的所有前景anchor框计数, batch_counts的shape是(batch_size, )
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)

    # 从target_bbox中选择前20个batch的数据
    batch_size = 20
    target_bbox = batch_back(target_bbox, batch_counts, batch_size)

    # 下面实现的实际上就是 smooth-L1 loss
    diff = K.abs(target_bbox - rpn_bbox)

    # 取出差值小于1的索引
    less_than_one = K.cast(K.less(diff, 1.0), "float32")

    # 如果差值小于1，则使用L2-loss；如果差值大于1，则使用L1-loss，TODO:但此时为什么要要减去0.5？
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    loss = K.switch(tf.size(loss) > 0 , K.mean(loss), tf.constant(0.0))
    return loss


def data_Gen(dataset, num_batch, batch_size, config):
    for _ in range(num_batch):
        images = []
        bboxes = []
        class_ids = []
        rpn_matchs = []
        rpn_bboxes = []
        for i in range(batch_size):
            image, bbox, class_id, rpn_match, rpn_bbox, _ = data = dataset.load_data()
            pad_num = config.max_gt_obj - bbox.shape[0]
            pad_box = np.zeros((pad_num, 4))
            pad_ids = np.zeros((pad_num, 1))
            bbox = np.concatenate([bbox, pad_box], axis=0)
            class_id = np.concatenate([class_id, pad_ids], axis=0)

            images.append(image)
            bboxes.append(bbox)
            class_ids.append(class_id)
            rpn_matchs.append(rpn_match)
            rpn_bboxes.append(rpn_bbox)
        images = np.concatenate(images, 0).reshape(batch_size, config.image_size[0], config.image_size[1], 3)
        bboxes = np.concatenate(bboxes, 0).reshape(batch_size, -1, 4)
        class_ids = np.concatenate(class_ids, 0).reshape(batch_size, -1)
        rpn_matchs = np.concatenate(rpn_matchs, 0).reshape(batch_size, -1, 1)
        rpn_bboxes = np.concatenate(rpn_bboxes, 0).reshape(batch_size, -1, 4)
        yield [images, bboxes, class_ids, rpn_matchs, rpn_bboxes], []



def anchor_refinement(boxes, deltas):
    """
    rpn target生成的逆操作
    :param boxes:
    :param deltas:
    :return:
    """
    boxes = tf.cast(boxes, tf.float32)
    h = boxes[:, 2] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + h / 2
    center_x = boxes[:, 1] + w / 2

    center_y += deltas[:, 0] * h
    center_x += deltas[:, 1] * w
    h *= tf.exp(deltas[:, 2])
    w *= tf.exp(deltas[:, 3])

    y1 = center_y - h / 2
    x1 = center_x - w / 2
    y2 = center_y + h / 2
    x2 = center_x + w / 2
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    return boxes


def boxes_clip(boxes, window):
    """
    boxes裁剪，确保boxes坐标值不超过边界
    :param boxes:
    :param window:
    :return:
    """
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    cliped = tf.concat([y1, x1, y2, x2], axis=1)
    cliped.set_shape((cliped.shape[0], 4))
    return cliped


def batch_slice(inputs, graph_fn, batch_size):
    if not isinstance(inputs, list):
        inputs = [inputs]
    output = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (list, tuple)):
            output_slice = [output_slice]
        output.append(output_slice)
    output = list(zip(*output))
    result = [tf.stack(o, axis=0) for o in output]
    if len(result) == 1:
        result = result[0]
    return result


import keras.engine as KE


class proposal(KE.Layer):
    def __init__(self, proposal_count, nms_thresh, anchors, batch_size, config=None, **kwargs):
        super(proposal, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.anchors = anchors
        self.nms_thresh = nms_thresh
        self.batch_size = batch_size
        self.config = config

    def call(self, inputs):
        probs = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, (1, 1, 4))
        prenms_num = min(100, self.anchors.shape[0])
        idxs = tf.nn.top_k(probs, prenms_num).indices

        probs = batch_slice([probs, idxs], lambda x, y: tf.gather(x, y), self.batch_size)
        deltas = batch_slice([deltas, idxs], lambda x, y: tf.gather(x, y), self.batch_size)
        anchors = batch_slice([idxs], lambda x: tf.gather(self.anchors, x), self.batch_size)
        refined_boxes = batch_slice([anchors, deltas], lambda x, y: anchor_refinement(x, y), self.batch_size)
        H, W = self.config.image_size[:2]
        windows = np.array([0, 0, H, W]).astype(np.float32)
        cliped_boxes = batch_slice([refined_boxes], lambda x: boxes_clip(x, windows), self.batch_size)
        normalized_boxes = cliped_boxes / np.array([H, W, H, W])

        def nms(normalized_boxes, scores):
            idxs_ = tf.image.non_max_suppression(normalized_boxes, scores, self.proposal_count, self.nms_thresh)
            box = tf.gather(normalized_boxes, idxs_)
            pad_num = tf.maximum(self.proposal_count - tf.shape(normalized_boxes)[0], 0)
            box = tf.pad(box, [(0, pad_num), (0, 0)])
            return box

        proposal_ = batch_slice([normalized_boxes, probs], nms, self.batch_size)
        return proposal_

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


if __name__ == '__main__1':  # 测试RPN网络
    input_image = KL.Input(shape=[64, 64, 3], dtype=tf.float32)
    feature_map = resNet_featureExtractor(input_image)
    rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map, 9)
    model = Model([input_image], [rpn_class, rpn_prob, rpn_bbox])

    print(model.outputs)

    model.summary()


if __name__ == '__main__2':  # 训练数据测试
    config = Config()
    dataset = dataSet([64, 64], config=config)

    dataGen = data_Gen(dataset, 35000, 20, config)

    for data in dataGen:
        [images, bboxes, class_ids, rpn_matchs, rpn_bboxes], xxx = data

        print('debug')


if __name__ == "__main__3":  # 训练RPN
    input_image = KL.Input(shape=[64, 64, 3], dtype=tf.float32)
    input_bboxes = KL.Input(shape=[None, 4], dtype=tf.float32)
    input_class_ids = KL.Input(shape=[None], dtype=tf.int32)
    input_rpn_match = KL.Input(shape=[None, 1], dtype=tf.int32)
    input_rpn_bbox = KL.Input(shape=[None, 4], dtype=tf.float32)

    feature_map = resNet_featureExtractor(input_image)
    rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map, 9)

    loss_rpn_match = KL.Lambda(lambda x: rpn_class_loss(*x), name="loss_rpn_match")([input_rpn_match, rpn_class])

    loss_rpn_bbox = KL.Lambda(lambda x: rpn_bbox_loss(*x), name="loss_rpn_bbox")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])

    model = Model([input_image, input_bboxes, input_class_ids, input_rpn_match, input_rpn_bbox],
                  [rpn_class, rpn_prob, rpn_bbox, loss_rpn_match, loss_rpn_bbox])
    model.summary()

    loss_lay1 = model.get_layer("loss_rpn_match").output
    loss_lay2 = model.get_layer("loss_rpn_bbox").output

    model.add_loss(tf.reduce_mean(loss_lay1))
    model.add_loss(tf.reduce_mean(loss_lay2))

    model.compile(loss=[None] * len(model.output), optimizer=keras.optimizers.SGD(lr=0.00005, momentum=0.9))

    model.metrics_names.append("loss_rpn_match")
    model.metrics_tensors.append(tf.reduce_mean(loss_lay1, keep_dims=True))

    model.metrics_names.append("loss_rpn_bbox")
    model.metrics_tensors.append(tf.reduce_mean(loss_lay2, keep_dims=True))

    config = Config()
    dataset = dataSet([64, 64], config=config)

    dataGen = data_Gen(dataset, 35000, 20, config)
    # his = model.fit_generator(dataGen, steps_per_epoch=20, epochs=1200)
    his = model.fit_generator(dataGen, steps_per_epoch=20, epochs=2)
    model.save_weights("model_material.h5")

    exit(0)


if __name__ == '__main__':
    input_image = KL.Input(shape=[64, 64, 3], dtype=tf.float32)
    input_bboxes = KL.Input(shape=[None, 4], dtype=tf.float32)
    input_class_ids = KL.Input(shape=[None], dtype=tf.int32)
    input_rpn_match = KL.Input(shape=[None, 1], dtype=tf.int32)
    input_rpn_bbox = KL.Input(shape=[None, 4], dtype=tf.float32)

    feature_map = resNet_featureExtractor(input_image)
    rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map, 9)

    loss_rpn_match = KL.Lambda(lambda x: rpn_class_loss(*x), name="loss_rpn_match")([input_rpn_match, rpn_class])

    loss_rpn_bbox = KL.Lambda(lambda x: rpn_bbox_loss(*x), name="loss_rpn_bbox")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])

    model = Model([input_image, input_bboxes, input_class_ids, input_rpn_match, input_rpn_bbox],
                  [rpn_class, rpn_prob, rpn_bbox, loss_rpn_match, loss_rpn_bbox])
    model.summary()

    loss_lay1 = model.get_layer("loss_rpn_match").output
    loss_lay2 = model.get_layer("loss_rpn_bbox").output

    model.add_loss(tf.reduce_mean(loss_lay1))
    model.add_loss(tf.reduce_mean(loss_lay2))

    model.compile(loss=[None] * len(model.output), optimizer=keras.optimizers.SGD(lr=0.00005, momentum=0.9))

    model.metrics_names.append("loss_rpn_match")
    model.metrics_tensors.append(tf.reduce_mean(loss_lay1, keep_dims=True))

    model.metrics_names.append("loss_rpn_bbox")
    model.metrics_tensors.append(tf.reduce_mean(loss_lay2, keep_dims=True))

    config = Config()
    dataset = dataSet([64, 64], config=config)

    dataGen = data_Gen(dataset, 35000, 20, config)

    model.load_weights("model_material.h5")



    # config = Config()
    # dataset = dataSet([64, 64], config=config)
    #
    # dataGen = data_Gen(dataset, 35000, 20, config)

    test_data = next(dataGen)[0]

    images = test_data[0]
    bboxes = test_data[1]
    class_ids = test_data[2]
    rpn_matchs = test_data[3]
    rpn_bboxes = test_data[4]

    rpn_class, rpn_prob, rpn_bbox, _, _ = \
        model.predict([images, bboxes, class_ids, rpn_matchs, rpn_bboxes])
    rpn_class.argmax()

    rpn_class = tf.convert_to_tensor(rpn_class)
    rpn_prob = tf.convert_to_tensor(rpn_prob)
    rpn_bbox = tf.convert_to_tensor(rpn_bbox)

    import utils

    anchors = utils.anchor_gen([8, 8], ratios=config.ratios, scales=config.scales, rpn_stride=config.rpn_stride,
                               anchor_stride=config.anchor_stride)

    proposals = proposal(proposal_count=16, nms_thresh=0.7, anchors=anchors, batch_size=20, config=config)(
        [rpn_prob, rpn_bbox])

    sess = tf.Session()
    proposals_ = sess.run(proposals) * 64

    import random

    ix = random.sample(range(20), 1)[0]
    proposal_ = proposals_[ix]
    img = images[ix]

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # %matplotlib inline

    # plt.imshow(img)
    # axs = plt.gca()

    if True:
        img_copy = img.astype(np.uint8)
        for i in range(proposal_.shape[0]):
            box = proposal_[i]
            # rec = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], facecolor='none', edgecolor='r')
            cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
            cv2.imshow('show', img_copy)
            cv2.waitKey()




input_image = KL.Input(shape=[64,64,3], dtype=tf.float32)
input_bboxes = KL.Input(shape=[None,4], dtype=tf.float32)
input_class_ids = KL.Input(shape=[None],dtype=tf.int32)
input_rpn_match = KL.Input(shape=[None, 1], dtype=tf.int32)
input_rpn_bbox = KL.Input(shape=[None, 4], dtype=tf.float32)


feature_map = resNet_featureExtractor(input_image)
rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map, 9)

loss_rpn_match = KL.Lambda(lambda x: rpn_class_loss(*x), name="loss_rpn_match")([input_rpn_match, rpn_class])

loss_rpn_bbox = KL.Lambda(lambda x: rpn_bbox_loss(*x), name="loss_rpn_bbox")([input_rpn_bbox, input_rpn_match, rpn_bbox])

model = Model([input_image, input_bboxes, input_class_ids, input_rpn_match, input_rpn_bbox],
              [rpn_class, rpn_prob, rpn_bbox, loss_rpn_match, loss_rpn_bbox])


import keras
loss_lay1 = model.get_layer("loss_rpn_match").output
loss_lay2 = model.get_layer("loss_rpn_bbox").output

model.add_loss(tf.reduce_mean(loss_lay1))
model.add_loss(tf.reduce_mean(loss_lay2))

model.compile(loss=[None]*len(model.output), optimizer=keras.optimizers.SGD(lr=0.00005, momentum=0.9))

model.metrics_names.append("loss_rpn_match")
model.metrics_tensors.append(tf.reduce_mean(loss_lay1, keep_dims=True))

model.metrics_names.append("loss_rpn_bbox")
model.metrics_tensors.append(tf.reduce_mean(loss_lay2, keep_dims=True))



from utils import shapeData as dataSet
from config import Config

config = Config()
dataset = dataSet([64,64], config=config)


def data_Gen(dataset, num_batch, batch_size, config):
    for _ in range(num_batch):
        images = []
        bboxes = []
        class_ids = []
        rpn_matchs = []
        rpn_bboxes = []
        for i in range(batch_size):
            image, bbox, class_id, rpn_match, rpn_bbox, _ = data = dataset.load_data()
            pad_num = config.max_gt_obj - bbox.shape[0]
            pad_box = np.zeros((pad_num, 4))
            pad_ids = np.zeros((pad_num, 1))
            bbox = np.concatenate([bbox, pad_box], axis=0)
            class_id = np.concatenate([class_id, pad_ids], axis=0)

            images.append(image)
            bboxes.append(bbox)
            class_ids.append(class_id)
            rpn_matchs.append(rpn_match)
            rpn_bboxes.append(rpn_bbox)
        images = np.concatenate(images, 0).reshape(batch_size, config.image_size[0], config.image_size[1], 3)
        bboxes = np.concatenate(bboxes, 0).reshape(batch_size, -1, 4)
        class_ids = np.concatenate(class_ids, 0).reshape(batch_size, -1)
        rpn_matchs = np.concatenate(rpn_matchs, 0).reshape(batch_size, -1, 1)
        rpn_bboxes = np.concatenate(rpn_bboxes, 0).reshape(batch_size, -1, 4)
        yield [images, bboxes, class_ids, rpn_matchs, rpn_bboxes], []

dataGen = data_Gen(dataset, 35000, 20, config)
#his = model.fit_generator(dataGen, steps_per_epoch=20, epochs=1200)
#model.save_weights("model_material.h5")
model.load_weights("model_material.h5")
















