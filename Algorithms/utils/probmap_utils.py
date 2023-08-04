# Import necessary packages
import mne
import numpy as np
import itertools
from joblib import load
from Algorithms.utils.resnet_predict import resnet_predict

from Preprocess.utils.filter import Filter

def split_eeg(raw_fname, duration, offset, tmin, tmax, filter_type='cheby'):
    """
    Splits EEG data into several time intervals.

    Returns:
    np.array: Array containing the split EEG data.
    """

    length = tmax - tmin

    # Load raw data
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)

    # Apply filtering to the data
    if filter_type is not None:
        filter = Filter(raw=raw)
        raw = filter.filter_data(filter_type=filter_type)
    else:
        print("No filter applied")
        raw.filter(l_freq=0.1, h_freq=50)

    # Extract time intervals
    events = mne.events_from_annotations(raw)[0]
    interval = events[events[:, 2] == 1][:, 0] / raw.info['sfreq']

    # Split the data into time intervals
    start = interval[0] + offset
    X = []
    while start <= interval[1] - length:
        Xi = raw.copy().crop(tmin=start, tmax=start + length).get_data(picks=picks)
        X.append(Xi)
        start += duration

    return np.array(X)

def generate_prob_map(raw_fname, duration, precision, model_name, cla, tmin, tmax):
    """
    Generates a probability map using a trained model.

    Parameters:
    raw_fname (str): Path to the file containing raw EEG data.
    duration (float): Duration of each interval.
    precision (int): The precision level for splitting the EEG data.
    model_name (str): Name of the model to use for prediction.
    cla (int): Class label.
    tmin (float): Start time.
    tmax (float): End time.

    Returns:
    list: The generated probability map.
    """

    prob_map = []
    pretrined_model_path = "Models/pre-trained/"
    # Split EEG data and get predictions
    for i in range(precision):
        # Load the model and get predictions
        if model_name == "ensemble":
            X = split_eeg(raw_fname, duration, i / precision, tmin, tmax)
            clf = load(pretrined_model_path + 'Ensemble.joblib')
            print("X.shape: ", X.shape)
            pred_prob_i = clf.predict_proba(X.reshape(X.shape[0], X.shape[1], X.shape[2]))[:, cla]
        elif model_name == "ensemble_80":
            X = split_eeg(raw_fname, duration, i / precision, tmin, tmax)
            clf = load(pretrined_model_path + 'Ensemble_80_2.joblib')
            pred_prob_i = clf.predict_proba(X.reshape(X.shape[0], X.shape[1], X.shape[2]))[:, cla]
        elif model_name == "ensemble_96":
            X = split_eeg(raw_fname, duration, i / precision, tmin, tmax)
            clf = load(pretrined_model_path + 'Ensemble_96.joblib')
            pred_prob_i = clf.predict_proba(X.reshape(X.shape[0], X.shape[1], X.shape[2]))[:, cla]
        elif model_name == "resnet":
            X = split_eeg(raw_fname, duration, i / precision, tmin, tmax, filter_type='bandpass')
            pred_prob_i = resnet_predict(X)[:, cla]
            print("pred_prob_i: ", pred_prob_i)
        else:
            raise ValueError("Model not found")

        # Update probability map
        # pred_prob_i = list(itertools.chain.from_iterable((x,) * precision for x in pred_prob_i))
        pred_prob_i = [x for x in pred_prob_i for _ in range(precision)]
        pred_prob_i = [0] * i + pred_prob_i
        if not prob_map:
            prob_map = pred_prob_i
        else:
            prob_map = [max(a, b) for a, b in zip(prob_map, pred_prob_i)]
    return prob_map

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

def get_start_time(raw_fname):
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    events = mne.events_from_annotations(raw)[0]
    start = events[events[:, 2] == 1][:, 0][0]
    return start