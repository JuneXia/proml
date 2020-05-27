"""
metrics/evaluate.py中的roc评估接口使用说明：
step1: 先将你的预测结果按 [真实标签， 每个类别的预测分数]的顺序存于csv文件；
    以二分类问题为例：0, 0.6, 0.3, 0.1
                   2, 0.1, 0.2, 0.8
                   1, 0.2, 0.6, 0.2
                   ...
    TODO: 预测标签可以去掉。

step2: 然后就可以按照本测试示例调用roc接口绘制roc曲线了。
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from libml.utils import tools
from libml.utils import plot
from libml.metrics import evaluate


if __name__ == '__main__1':
    pred_file = 'data/test_bin_classify_predict_result1.csv'
    preds = tools.load_csv(pred_file)
    preds = preds.astype(np.float32)
    y_true, y_score = preds[:, 0], preds[:, 1:]
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.stack((1 - y_true, y_true)).transpose()
    # y_score = evaluate.prob_encode(y_pred, y_score)

    fpr, tpr, thresholds, roc_auc = evaluate.roc(y_true, y_score)

    # 绘制所有ROC曲线
    n_classes = y_true.shape[-1]
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for j, color in zip(range(n_classes), colors):
        plt.plot(fpr[j], tpr[j], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(j, roc_auc[j]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__2':
    pred_file = [{'csv_file': 'data/test_bin_classify_predict_result1.csv',
                  'name': 'paddle'},
                 {'csv_file': '/disk1/home/xiaj/dev/flaskmask/paddle_mask_detection/predicts_pyramidbox_lite_server_mask_1.1.0.csv',
                  'name': 'paddle1.1.0'},
                 {# 'csv_file': 'data/test_bin_classify_predict_result2.csv',
                  'csv_file': '/disk1/home/xiaj/dev/FlaskFace/predicts_retinaface_inceptionresnetv1.csv',
                  'name': 'our'}
                 ]

    plt.figure()
    for pfile in pred_file:
        preds = tools.load_csv(pfile['csv_file'])
        preds = preds.astype(np.float32)
        y_true, y_score = preds[:, 0], preds[:, 1:]
        y_pred = np.argmax(y_score, axis=1)
        y_true = np.stack((1 - y_true, y_true)).transpose()

        fpr, tpr, thresholds, roc_auc = evaluate.roc(y_true, y_score)

        # 绘制所有ROC曲线
        n_classes = y_true.shape[-1]
        lw = 2
        color = plot.get_color()
        if True:
            plt.plot(fpr["micro"], tpr["micro"],
                     label='{}: micro-average ROC (auc={:0.2f})'.format(pfile['name'], roc_auc["micro"]),
                     color=color, linestyle=':', linewidth=2)

        if False:
            plt.plot(fpr["macro"], tpr["macro"],
                     label='{}: macro-average ROC (auc={:0.2f})'.format(pfile['name'], roc_auc["macro"]),
                     color=color, linestyle=':', linewidth=2)

        if False:
            for j in range(n_classes):
                linestyle = plot.get_linestyle(j)
                plt.plot(fpr[j], tpr[j], lw=lw,
                         label='{}: ROC curve of class {} (area={:0.2f})'.format(pfile['name'], j, roc_auc[j]),
                         color=color, linestyle=linestyle)

        if False:
            for j, (fp, tp, thr) in enumerate(zip(fpr['micro'], tpr['micro'], thresholds['micro'])):
                plt.text(fp, tp, '%.4f' % thr, color='navy', ha='center', va='bottom', fontsize=9)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

