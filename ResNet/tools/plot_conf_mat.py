import os
from matplotlib import pyplot as plt
import seaborn as sns


def plot_confusion_matrix(folder_path, classes, title='Confusion matrix', cmap='Blues'):
    plt.figure(figsize=(10,7))
    plt.rcParams.update({'font.size': 16})
    cm = [[0.88, 0.12], [0.11, 0.89]]
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(folder_path, 'confusion_matrix.png'))
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix("../", ['no error', 'S 96'])
