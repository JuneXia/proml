import tensorflow as tf
import cv2
import numpy as np
from libml.utils.config import SysConfig


def non_max_suppression(bounding_boxes, confidence_score, threshold):
    """
    reference: https://hellozhaozheng.github.io/z_post/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89-NMS-Implementation/
    :param bounding_boxes:
    :param confidence_score:
    :param threshold: IoU 阈值
    :return:
    """
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)
    score = np.array(confidence_score)

    # 计算 n 个候选框的面积大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas =(x2 - x1 + 1) * (y2 - y1 + 1)

    # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
    order = np.argsort(score)

    picked_index = [] # 返回值
    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_index.append(index)

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < threshold)
        order = order[left]

    return picked_index


# testing
if __name__ == "__main__":
    img = cv2.imread(SysConfig['home_path'] + "/dev/proml/tutorial/lena.png")

    boxes = [[50, 150, 200, 400], [80, 200, 260, 420], [300, 50, 460, 240], [320, 150, 480, 350]]
    scores = [0.3, 0.6, 0.8, 0.2]
    colors = [(0, 0, 100), (0, 0, 200), (0, 100, 0), (0, 200, 0)]
    normalized_boxes = np.array(boxes) / 512
    proposal_count = len(normalized_boxes)
    nms_thresh = 0.1  # IoU 阈值

    img_copy = img.copy()
    for i in range(len(boxes)):
        box = boxes[i]
        color = colors[i]
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), color, thickness=2)
    cv2.imwrite('proposal_bboxes.jpg', img_copy)
    cv2.imshow('proposal_bboxes', img_copy)
    cv2.waitKey()

    pick_index = non_max_suppression(boxes, scores, nms_thresh)

    idxs = tf.image.non_max_suppression(normalized_boxes, scores, proposal_count, nms_thresh)
    sess = tf.Session()
    rslt = sess.run(idxs)

    img_copy = img.copy()
    for i, idx in enumerate(rslt):
        box = boxes[idx]
        color = colors[idx]
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), color, thickness=2)
    cv2.imwrite('nms_bboxes.jpg', img_copy)
    cv2.imshow('nms_bboxes', img_copy)
    cv2.waitKey()
