import torch
import numpy as np
from datasets.dataset import EEGDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def predict(model, loader):
    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()

    all_preds = []
    all_labels = []

    for i, (inputs, labels) in enumerate(loader):
        with torch.no_grad():
            pred = model(inputs)

        pred = torch.max(pred.data, 1)[1]
        labels = torch.max(labels.data, 1)[1]
        pred_np = pred.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        all_preds.append(pred_np)
        all_labels.append(labels_np)

    return np.concatenate(all_labels), np.concatenate(all_preds)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}.')

    MODEL_PATH = "./tmp/train_logs/resnet_dropout_136batch_120epochs/latest_model.pt"

    # Load model
    model = torch.load(MODEL_PATH)
    model.to(device)

    # Load data
    # We have 3258 training examples and 1087 test examples.
    X = np.load('./tmp/data/X.npy')
    y = np.load('./tmp/data/y.npy')
    X_val = np.load('./tmp/data/X_val.npy')
    y_val = np.load('./tmp/data/y_val.npy')
    print(f'Loaded data.')

    # Standardize data
    # Note that X has shape (num_samples, num_features, sequence_length)
    mean = np.mean(X, axis=(0, 2), keepdims=True)  # Calculate mean across samples and sequence length
    std = np.std(X, axis=(0, 2), keepdims=True)  # Calculate standard deviation across samples and sequence length
    # Perform standardization (Z-score normalization)
    X = (X - mean) / (std + 1e-7)
    # Standardize validation data
    X_val = (X_val - mean) / (std + 1e-7)
    print(f'Standardized data.')

    # Initialize dataloaders
    batch_size = 136
    train_dataset = EEGDataset(X, y, device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # each set has 68 trials
    val_dataset = EEGDataset(X_val, y_val, device)
    val_dataloader = DataLoader(dataset=val_dataset)

    labels, preds = predict(model, val_dataloader)
    conf_mat = confusion_matrix(labels, preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot()
    plt.show()
