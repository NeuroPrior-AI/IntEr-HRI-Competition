import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns


def standardize(X, X_val, X_test):
    # Standardize data
    # Note that X has shape (num_samples, num_features, sequence_length)
    mean = np.mean(X, axis=(0, 2), keepdims=True)  # Calculate mean across samples and sequence length
    std = np.std(X, axis=(0, 2), keepdims=True)  # Calculate standard deviation across samples and sequence length
    # Perform standardization (Z-score normalization)
    X = (X - mean) / (std + 1e-7)
    # Standardize validation data
    X_val = (X_val - mean) / (std + 1e-7)
    # Standardize validation data
    X_test = (X_test - mean) / (std + 1e-7)

    return X, X_val, X_test, mean, std


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 16})
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['no error', 'S 96'],
                yticklabels=['no error', 'S 96'])
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()


def calculate_f1_score(confusion_matrix):
    # Extract true positive, false positive, true negative, false negative
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]

    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def predict(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            pred = model(inputs)
            pred = torch.max(pred.data, 1)[1]   # get class indices of each prediction
            pred_np = pred.cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            all_preds.extend(pred_np)
            all_labels.extend(labels_np)

    return np.array(all_labels), np.array(all_preds)


if __name__ == "__main__":
    ...
