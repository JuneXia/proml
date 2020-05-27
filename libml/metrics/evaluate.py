import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp


def prob_encode(y_pred, prob):
    """
    输入参数示例：y_pred=array([0, 1, 0, ...])
                 prob=array([0.6, 0.9, 0.8, ...])
                 prob[0]表示预测为y_pred[0]的概率，
                 prob[i]表示预测为y_pred[i]的概率.
    函数功能： 将prob转化为 array([[0.6, 0.4]
                                [0.1, 0.9]
                                [0.8, 0.2]
                                ...       ])
    :param y_pred:
    :param prob:
    :return:
    """
    if not (y_pred.shape == prob.shape and len(y_pred.shape) == 1):
        raise Exception('y_pred.shape must be equal to prob.shape!')
    if set(y_pred.astype(np.int32)) != {0, 1}:
        raise Exception('y_pred must be binary classes!')

    y_score = (1 - y_pred) * (1 - prob) + y_pred * prob
    y_score = np.stack((1 - y_score, y_score)).transpose()

    return y_score


def roc(y_true, y_score):
    """
    输入参数示例： y_true: array([[1, 0, 0]
                                [0, 0, 1]
                                [0, 1, 0]
                                ...      ])
                 y_score: array([[0.6, 0.3, 0.1]
                                 [0.1, 0.2, 0.8]
                                 [0.2, 0.6, 0.2]
                                  ...           ])
    :param y_true:
    :param y_score:
    :return: for example，对于三分类问题，返回值应当是 “fpr, tpr, thresholds, roc_auc”这个4个字典，
            每个字典都含有4个key: "0,1,2,mirco,macro"
    """
    if not (y_true.shape == y_score.shape):
        raise Exception('y_true.shape must be equal to y_score.shape!')

    n_classes = y_true.shape[-1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # 计算微平均ROC曲线和AUC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # 计算宏平均ROC曲线和AUC
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 首先汇总所有FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 然后再用这些点对ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # 最后求平均并计算AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return fpr, tpr, thresholds, roc_auc


def confusion_matrix(y_true, y_pred):
    tp = np.sum(np.logical_and(y_pred, y_true))
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

    return tp, fp, tn, fn


def plt_roc(fpr, tpr, roc_auc):
    """
    输入参数示例： fpr: array([0.1, 0.5, 0.6, ... ])
                 tpr: array([0.0, 0.4, 0.7, ... ])
                 roc_auc: array([0.86])
    :param fpr:
    :param tpr:
    :param roc_auc:
    :return:
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
