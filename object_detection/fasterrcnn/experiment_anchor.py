import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 常规输入以及anchor框参数
# *************************************
# input_shape = (128, 128, 3)
# size_X = 16  # 每行生成多少个anchor框
# size_Y = 16  # 每列生成多少个anchor框
# rpn_stride = 8  # 相邻两个anchor框之间的跨度
#
# scales = [2, 4, 8]  # anchor框的缩放比例
# ratios = [0.5, 1, 2]  # anchor框的长宽比
# *************************************


# 为了方便了解anchor框的生成机理，这里使用较小的参数来生成anchor框，方便理解
# *************************************
input_shape = (24, 24, 3)
size_X = 3  # 每行生成多少个anchor框
size_Y = 3  # 每列生成多少个anchor框
rpn_stride = 8  # 相邻两个anchor框之间的跨度

scales = [1, 2, 4]  # anchor框的缩放比例
ratios = [0.5, 1, 2]  # anchor框的长宽比
# *************************************


def anchor_gen(size_X, size_Y, rpn_stride, scales, ratios):
    scales, ratios = np.meshgrid(scales, ratios)
    scales, ratios = scales.flatten(), ratios.flatten()
    scalesY = scales * np.sqrt(ratios)
    scalesX = scales / np.sqrt(ratios)

    shiftX = np.arange(0, size_X) * rpn_stride
    shiftY = np.arange(0, size_Y) * rpn_stride
    shiftX = shiftX + rpn_stride/2
    shiftY = shiftY + rpn_stride/2
    shiftX, shiftY = np.meshgrid(shiftX, shiftY)

    centerX, anchorX = np.meshgrid(shiftX, scalesX)
    centerY, anchorY = np.meshgrid(shiftY, scalesY)

    anchor_center = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    anchor_size = np.stack([anchorY, anchorX], axis=2).reshape(-1, 2)

    boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)

    return boxes


if __name__ == '__main__1':
    anchors = anchor_gen(size_X, size_Y, rpn_stride, scales, ratios)

    plt.figure(figsize=(10, 10))
    img = np.ones(input_shape)
    plt.imshow(img)

    axs = plt.gca()  # get current axs

    for i in range(anchors.shape[0]):
        box = anchors[i]
        rec = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor="r", facecolor="none")
        axs.add_patch(rec)

    plt.show()


    print('debug')

