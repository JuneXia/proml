import numpy as np
import matplotlib.pyplot as plt
from libml.utils import tools
from libml.metrics import evaluate


if __name__ == '__main__':
    pred_file = [{'csv_file': 'data/test_bin_classify_predict_result2.csv',
                  'name': 'paddle'},
                 {'csv_file': '/disk1/home/xiaj/dev/flaskmask/paddle_mask_detection/predicts_pyramidbox_lite_server_mask_1.1.0.csv',
                  'name': 'paddle1.1.0'},
                 {# 'csv_file': 'data/test_bin_classify_predict_result2.csv',
                  'csv_file': '/disk1/home/xiaj/dev/FlaskFace/predicts_retinaface_inceptionresnetv1.csv',
                  'name': 'our'}
                 ]

    plt.figure()
    result = []
    for pfile in pred_file:
        preds = tools.load_csv(pfile['csv_file'])
        preds = preds.astype(np.float32)
        y_true, y_score = preds[:, 0], preds[:, 1:]
        y_pred = np.argmax(y_score, axis=1)

        tp, fp, tn, fn = evaluate.confusion_matrix(y_true, y_pred)

        P = tp/(tp+fp)
        R = tp/(tp+fn)
        fpr = fp/(tn+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)

        result.append({'P': P, 'R': R, 'fpr': fpr, 'acc': acc, 'name': pfile['name']})

    print('print result:')
    print('**********************************************')
    for rslt in result:
        print(rslt)
    print('**********************************************')

