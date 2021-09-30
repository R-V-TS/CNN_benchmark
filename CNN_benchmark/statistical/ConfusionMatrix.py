import numpy as np


def ConfusionMatrix(y_true, y_predict, normalize=False, class_length=None):
    """
    Calculate a confusion matrix
    :param y_true: True array (class_id for all blocks (images))
    :param y_predict: Predicted array (class_id for all blocks (images))
    :param normalize: is normalize matrix
    :return: confusion matrix
    """
    if class_length is None:
        class_length = max(set(y_true)) + 1

    conflusion_matrix = np.zeros((class_length, class_length), dtype=np.float32 if normalize else np.uint8)

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    for true_i, predict_i in zip(y_true.flatten(), y_predict.flatten()):
        conflusion_matrix[predict_i][true_i] += 1

    return conflusion_matrix