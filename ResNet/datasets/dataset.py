import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):

    def __init__(self, data, labels, device):
        self.data = torch.tensor(data, dtype=torch.float32).to(device)       # num trials in single recording (68) x 64 eeg channels x 1 second (501)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)   # num trials in single recording x 1
        self.n_samples = self.data.shape[0]             # # num trials in single recording
        self.lens = [d.shape[1] for d in data]          # used for padding sequences (i.e. pack_padded_sequence) when using RNN

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index, :, :], self.labels[index]
