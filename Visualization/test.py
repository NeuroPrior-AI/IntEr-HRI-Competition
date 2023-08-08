from Algorithms.utils.resnet_predict import resnet_predict
from mne.io import RawArray
import mne
import numpy as np
from mne import create_info

raw_fname = "Dataset/training data/AA56D/data/20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr"
marker_fname = "Dataset/training data/AA56D/data/20230427_AA56D_orthosisErrorIjcai_multi_set1.vmrk"

# Load the EEG data
raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)

# Get the data and transpose it to the correct shape (channels x time)
data = raw.get_data().T

sfreq = 500  # Adjust based on your data's sampling rate
ch_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ',
            'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4', 'P8',
            'PO9', 'O1', 'OZ', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1',
            'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2',
            'C6', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
            'PO3', 'POZ', 'PO4', 'PO8']  # List of channel names in the order of your data

ch_types = ['eeg'] * len(ch_names)
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create Raw object from the data buffer
raw = RawArray(data[:, 0:64].T, info)

# Pick channels by type (in this case, EEG channels)
picks = mne.pick_types(info, meg=False, eeg=True)
print("info:", info)

# Apply filtering to the data (only to the channels specified by picks)
raw.filter(l_freq=1, h_freq=30, picks=picks)

data_get = raw.get_data(picks=picks)

# Split the data into time intervals
start = 0
X = []
print("data_get.shape: ", data_get.shape[1]/500)
while start <= data_get.shape[1]/500 - 3:
    Xi = raw.copy().crop(tmin=start, tmax=start + 1).get_data(picks=picks)
    X.append(Xi)
    start += 0.01

X = np.array(X)
print("X.shape: ", X.shape)
pred_prob_i = resnet_predict(X)[:, 1]
print("pred_prob_i: ", pred_prob_i)

print("number greate than 0.9: ", np.sum(pred_prob_i > 0.9))