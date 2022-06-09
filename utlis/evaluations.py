import numpy as np
from sklearn.metrics import roc_curve, auc

def evaluate(confusion_matrix, scores, targets):
    cm_value = confusion_matrix.value()
    scores = np.array(scores)
    targets = np.array(targets)

    # ROC and AUC
    fpr, tpr, threshold = roc_curve(targets, scores[:, 1])
    roc_auc = auc(fpr, tpr)

    # ACC
    accuracy = 100. * ((cm_value[0][0] + cm_value[1][1]) /
                       (cm_value.sum()))

    # precision (Benign = 0, Malignant = 1)
    precision = 100. * ((cm_value[1][1]) / ((cm_value[0][1]) + (cm_value[1][1])))

    # recall
    recall = 100.*((cm_value[1][1]) / ((cm_value[1][1]) + (cm_value[1][0])))

    # specificity
    specificity = 100.*((cm_value[0][0]) / ((cm_value[0][0]) + (cm_value[0][1])))

    # F1-value
    F1 = 100.*((2 * (cm_value[1][1])) / (2 * (cm_value[1][1]) +
                                         (cm_value[1][0]) + (cm_value[0][1])))

    return fpr, tpr, roc_auc, accuracy, precision, recall, specificity, F1
