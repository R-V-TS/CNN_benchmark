import matplotlib.pyplot as plt
import numpy as np
from . import utils


def ConfusionMatrix(confusion_matrix, labels=None, show=True, save=False):
    if labels is None:
        labels = np.arange(np.shape(confusion_matrix)[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion_matrix)

    for i in range(np.shape(confusion_matrix)[0]):
        for j in range(np.shape(confusion_matrix)[1]):
            text = ax.text(j, i, f"{int(confusion_matrix[i, j])}",
                           ha="center", va="center", color="w")

    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.xticks(range(np.shape(confusion_matrix)[0]), labels)
    plt.yticks(range(np.shape(confusion_matrix)[0]), labels)

    if save:
        utils.__result_dir_create("results")
        plt.savefig('results/conflusion_matrix.jpg')
    if show:
        plt.show()