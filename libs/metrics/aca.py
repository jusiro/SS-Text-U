import numpy as np

from sklearn.metrics import confusion_matrix


def average_class_wise_accuracy(output, target):

    # Confusion matrix
    cm = confusion_matrix(target, np.argmax(output, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    # Accuracy per class - and average
    aca = np.round(np.mean(np.diag(cm_norm) * 100), 2)

    return aca