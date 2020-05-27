import torch
from torch import nn
from torch.nn import functional as F


class ClassifyLoss(nn.Module):
    def __init__(self):
        super(ClassifyLoss, self).__init__()

    def forward(self, input, target):
        """
        :param input: Predict result
        :param target: Ground-Truth
        :return:
        """

        # 挤压掉最后一个维度，现在rpn_match.shape = (b, num_anchors)
        # rpn_match = tf.squeeze(rpn_match, -1)

        # 提取出rpn_match中所有的前后景anchor索引
        indices = tf.where(K.not_equal(rpn_match, 0))

        # 将rpn_match中所有非前景的标签都置为0
        anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)

        # 提取出indices索引所在的预测得分值
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)  ### prediction

        # 提取出indices索引所在的anchor框标签
        anchor_class = tf.gather_nd(anchor_class, indices)  ### target

        # 计算交叉熵损失
        loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_logits, from_logits=True)

        # 如果loss长度大于0,则对loss取均值，否则loss值取为0
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss



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

