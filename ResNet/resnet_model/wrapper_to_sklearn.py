import torch
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, mean, std, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.mean = mean
        self.std = std

    def fit(self, X, y):
        # PyTorch models typically need a separate training function that handles mini-batching
        # Here, you could write your PyTorch training loop, or call another function that does it
        pass

    def predict(self, X):
        with torch.no_grad():
            self.model.eval()
            X = (X - mean) / (std + 1e-7)       # standardize data
            X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            self.model.eval()
            X = (X - mean) / (std + 1e-7)       # standardize data
            X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
            outputs = torch.softmax(self.model(X_tensor), dim=1)
        return outputs.cpu().numpy()


if __name__ == "__main__":
    # Load model:
    model = torch.load('./best_model.pt')
    # Load standardization parameters:
    mean = np.load('./mean.npy')
    std = np.load('./std.npy')
    # Wrap model to scikit-learn (with predict function)
    pytorch_classifier = PyTorchClassifier(model, mean, std)

    # TODO: Load data:
    # X needs to be 3 dimensional and have shape (batch size, 64, 501)
    X = np.random.rand(1, 64, 501)

    pred = pytorch_classifier.predict(X)

    print(pred)