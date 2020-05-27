import numpy as np
from libml.utils import tools
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as pr


if __name__ == '__main__':
    pred_file = [{# 'csv_file': 'data/test_bin_classify_predict_result1.csv',
                  'csv_file': '/disk1/home/xiaj/dev/flaskmask/paddle_mask_detection/predicts_pyramidbox_lite_server_mask_1.1.0.csv',
                  'name': 'paddle'},
                 {# 'csv_file': 'data/test_bin_classify_predict_result2.csv',
                  'csv_file': '/disk1/home/xiaj/dev/FlaskFace/predicts_retinaface_inceptionresnetv1.csv',
                  'name': 'our'}
                 ]

    plt.figure()
    for pfile in pred_file:
        preds = tools.load_csv(pfile['csv_file'])
        preds = preds.astype(np.float32)
        y_true, y_score = preds[:, 0], preds[:, 1:]
        pos_score = y_score[:, 1]

        P, R, thresholds = pr(y_true, pos_score)
        n_classes = len(set(y_true))

        plt.plot(R, P,
                 label='{}'.format(pfile['name']),
                 color='deeppink', linestyle='-', linewidth=2)

    lw = 2
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R binary classification')
    plt.legend(loc="lower right")
    plt.show()
