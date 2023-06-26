# mne imports
import os
print(os.getcwd())
import mne
import numpy as np
from mne.preprocessing import Xdawn
import warnings
from scipy.signal import butter, cheby1, ellip, sosfilt
import pywt


event_id={'no error': 1, 'error': 2, 'reaction': 3}
mapping = {64: 1, 32: 1, 48: 1, 96: 2, 80: 3, 1: 1, 2: 1, 16: 1}
tmin=-0.1
tmax=0.9

# Other import statements ...



def process_file(raw_fname):
    warnings.filterwarnings("ignore")
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    
    # Get the data and the sampling frequency
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    # Define your lowcut and highcut frequencies
    lowcut = 0.1
    highcut = 30
    
    #d Apply butterworth filter
    #data_filtered = butter_bandpass_filter(data, lowcut, highcut, sfreq, order=5)
    #raw_filtered = raw.copy()
    #raw_filtered._data = data_filtered

    # Apply chevyshev filter
    data_filtered = cheby_bandpass_filter(data, lowcut, highcut, sfreq, order=5, rp=5)
    raw_filtered = raw.copy()
    raw_filtered._data = data_filtered

    # Apply elliptic (Cauer) filter
    #data_filtered = ellip_bandpass_filter(data, lowcut, highcut, sfreq, order=5, rp=5, rs=40)
    #raw_filtered = raw.copy()
    #raw_filtered._data = data_filtered

    # Apply Discrete Wavelet Transform
    # Note: Wavelet transform returns two sets of coefficients, not a filtered signal.
    # Hence, it may not be suitable to replace the signal in the Raw object.
    # (cA, cD) = wavelet_transform(data, wavelet='db4')

    # Update 'raw' to be the filtered data
    raw = raw_filtered

    # Apply the filter
    #raw.filter(l_freq=0.1, h_freq=30)
    events = mne.events_from_annotations(raw)[0]
    # Apply the mapping to the second column of the events array
    events[:, 2] = np.vectorize(lambda x: mapping.get(x, -1))(events[:, 2])
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                        proj=False, picks=picks, baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    X = epochs.get_data()*1000 # format is in (trials, channels=64, samples=501)
    y = labels

    return X, y


def split_eeg(raw_fname, duration, offset):
    length = tmax - tmin
    # Get the interval to split
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    raw.filter(l_freq=0.1, h_freq=30, method='iir')
    events = mne.events_from_annotations(raw)[0]
    interval = events[events[:, 2] == 1][:, 0]/raw.info['sfreq']

    # Split the interval
    start = interval[0] + offset
    X = []
    while start <= interval[1] - length:
        # Extract the slice
        Xi = raw.copy().crop(tmin=start, tmax=start + length).get_data(picks=picks)*1000
        # Add the slice to the list
        X.append(Xi)
        start += duration
        
    return np.array(X)

def get_96_timepts(raw_fname):
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    events = mne.events_from_annotations(raw)[0]
    start = events[events[:, 2] == 1][:, 0][0]
    return (events[events[:, 2] == 96][:, 0] - start)/raw.info['sfreq']

def get_80_timepts(raw_fname):
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    events = mne.events_from_annotations(raw)[0]
    start = events[events[:, 2] == 1][:, 0][0]
    return (events[events[:, 2] == 80][:, 0] - start)/raw.info['sfreq']

def get_64_32_timepts(raw_fname):
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    events = mne.events_from_annotations(raw)[0]
    start = events[events[:, 2] == 1][:, 0][0]
    fr = events[(events[:, 2] == 64) | (events[:, 2] == 32)][:, 0]
    return (fr)/raw.info['sfreq']

# Butterworth Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

# Chebyshev Filter
def cheby_bandpass(lowcut, highcut, fs, order=5, rp=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = cheby1(order, rp, [low, high], btype='band', output='sos')
    return sos

def cheby_bandpass_filter(data, lowcut, highcut, fs, order=5, rp=5):
    sos = cheby_bandpass(lowcut, highcut, fs, order=order, rp=rp)
    y = sosfilt(sos, data)
    return y

# Elliptic (Cauer) Filter
def ellip_bandpass(lowcut, highcut, fs, order=5, rp=5, rs=40):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = ellip(order, rp, rs, [low, high], btype='band', output='sos')
    return sos

def ellip_bandpass_filter(data, lowcut, highcut, fs, order=5, rp=5, rs=40):
    sos = ellip_bandpass(lowcut, highcut, fs, order=order, rp=rp, rs=rs)
    y = sosfilt(sos, data)
    return y

# Discrete Wavelet Transform
def wavelet_transform(data, wavelet='db4'):
    (cA, cD) = pywt.dwt(data, wavelet)
    return cA, cD


