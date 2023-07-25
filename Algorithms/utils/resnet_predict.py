import torch
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from Models.resnet.resnet import ResNet

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
            X = (X - self.mean) / (self.std + 1e-7)       # standardize data
            X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)      # predictions has shape (batch,)
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            self.model.eval()
            X = (X - self.mean) / (self.std + 1e-7)       # standardize data
            X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
            outputs = torch.softmax(self.model(X_tensor), dim=1)    # output has shape (batch, num_classes)
        return outputs.cpu().numpy()

def resnet_predict(X):
    '''
    X needs to be 3 dimensional and have shape (batch size, 64, 501)
    '''
    # Load model:
    model = ResNet(num_classes=2)
    checkpoint = torch.load('../Models/pre-trained/resnet/best_model.pt')
    model.load_state_dict(checkpoint)
    # Load standardization parameters:
    mean = np.load('../Models/pre-trained/resnet/mean.npy')
    std = np.load('../Models/pre-trained/resnet/std.npy')
    # Wrap model to scikit-learn (with predict function)
    pytorch_classifier = PyTorchClassifier(model, mean, std)

    return pytorch_classifier.predict_proba(X)
