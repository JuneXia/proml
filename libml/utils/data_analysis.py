"""
from FlaskFace/utils/data_analysis.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号


def hist_show(data, bins=10):
    """
    绘制直方图
    data:必选参数，绘图数据
    bins:直方图的长条形数目，可选项，默认为10
    normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    facecolor:长条形的颜色
    edgecolor:长条形边框的颜色
    alpha:透明度
    """
    plt.hist(data, bins=bins, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("range")
    # 显示纵轴标签
    plt.ylabel("frequency")
    # 显示图标题
    plt.title("frequency distribution histogram")
    plt.show()


def youtube_data_analysis(data_path):
    cls_list = os.listdir(data_path)
    print('')
    cls_statistic = []
    for cls in cls_list:
        cls_path = os.path.join(data_path, cls)
        folder_list = os.listdir(cls_path)

        img_count = 0
        for folder in folder_list:
            folder_path = os.path.join(cls_path, folder)
            img_list = os.listdir(folder_path)
            img_count += len(img_list)

        if img_count < 20:
            print(cls_path, ' have {} images'.format(img_count))
        cls_statistic.append([cls, len(folder_list), img_count])

    cls_statistic.sort(key=lambda x: x[2])

    return cls_statistic


def celeb_data_analysis(data_path):
    tmp_count = 0
    cls_list = os.listdir(data_path)
    cls_statistic = []
    for cls in cls_list:
        cls_path = os.path.join(data_path, cls)
        if os.path.isfile(cls_path):
            print(cls_path, ' is not folder, continue')
            continue
        img_list = os.listdir(cls_path)

        if len(img_list) <= 1:
            tmp_count += 1
            print(cls_path, ' have {} images'.format(len(img_list)))

        cls_statistic.append([cls, len(img_list)])

    cls_statistic.sort(key=lambda x: x[1])

    print(tmp_count)
    return cls_statistic


def vggface2_data_analysis(data_path):
    return celeb_data_analysis(data_path)

def gcwebface_data_analysis(data_path):
    return celeb_data_analysis(data_path)

if __name__ == '__main__':
    # 随机生成（10000,）服从正态分布的数据
    data = np.random.randn(10000)
    hist_show(data, bins=40)

    """
    data_path = '/home/xiajun/res/face/YouTubeFaces/Experiment/frame_mtcnn_align55x47_margin16'
    cls_statistic = youtube_data_analysis(data_path)
    cls_statistic_array = np.array(cls_statistic)
    print('class num: {}, total iamges: {}'.format(cls_statistic_array.shape[0], cls_statistic_array[:, 2].astype(int).sum()))
    hist_show(cls_statistic_array[:, 2].astype(int), bins=500)
    """

    """
    data_path = '/home/xiajun/res/face/CelebA/Experiment/img_align_celeba_identity_crop'
    cls_statistic = celeb_data_analysis(data_path)
    cls_statistic_array = np.array(cls_statistic)
    print('class num: {}, total iamges: {}'.format(cls_statistic_array.shape[0], cls_statistic_array[:, 1].astype(int).sum()))
    hist_show(cls_statistic_array[:, 1].astype(int), bins=500)
    """

    """
    data_path = '/home/xiajun/res/face/VGGFace2/Experiment/train_mtcnn_align160x160_margin32'
    cls_statistic = vggface2_data_analysis(data_path)
    cls_statistic_array = np.array(cls_statistic)
    print('class num: {}, total iamges: {}'.format(cls_statistic_array.shape[0], cls_statistic_array[:, 1].astype(int).sum()))
    hist_show(cls_statistic_array[:, 1].astype(int), bins=500)
    """


    data_path = '/disk1/home/xiaj/res/face/Trillion Pairs/train_msra/Experiment/retinaface_align182x182_margin44'
    cls_statistic = gcwebface_data_analysis(data_path)
    cls_statistic_array = np.array(cls_statistic)
    print('class num: {}, total iamges: {}'.format(cls_statistic_array.shape[0], cls_statistic_array[:, 1].astype(int).sum()))
    hist_show(cls_statistic_array[:, 1].astype(int), bins=500)

    print('debug')
