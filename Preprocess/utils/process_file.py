import mne
import numpy as np
import warnings
import warnings
from utils.filter import Filter


def process_file(raw_fname, tmin, tmax, mapping, event_id, filter_type):
    # Ignore warnings and set log level
    warnings.filterwarnings("ignore")
    mne.set_log_level('WARNING')

    # Read the raw data and select the EEG channels
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg=False,
                           eeg=True, stim=False, eog=False)

    # Apply filtering to the data
    filter = Filter(raw=raw)
    raw = filter.filter_data(filter_type=filter_type)

    # Extract events and remap the event ids
    events = mne.events_from_annotations(raw)[0]
    events[:, 2] = np.vectorize(lambda x: mapping.get(x, -1))(events[:, 2])

    # Create epochs from the raw data and return data and labels
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        proj=False, picks=picks, baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    X = epochs.get_data()  # format is in (trials, channels=64, samples=501)
    y = labels

    return X, y


# Function to process files in seconds
def process_file_sec(raw_fname, filter_type, length):
    # Ignore warnings and set log level
    warnings.filterwarnings("ignore")
    mne.set_log_level('WARNING')

    # Read the raw data and select the EEG channels
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg=False,
                           eeg=True, stim=False, eog=False)
    
    # Apply filtering to the data
    filter = Filter(raw=raw)
    raw = filter.filter_data(filter_type=filter_type)


    # Prepare for extraction of one-second epochs
    # print("Length of data in seconds: ", raw.times[-1])
    if length == None:
        length_of_data_in_sec = raw.times[-1]
    else:
        length_of_data_in_sec = length

    X = []
    y = []
    for start_time in range(int(length_of_data_in_sec)):
        one_sec_raw = raw.copy().crop(tmin=start_time, tmax=start_time + 1)
        events, event_id = mne.events_from_annotations(one_sec_raw)

        one_sec_raw_matrix = one_sec_raw.get_data(picks=picks)

        X.append(one_sec_raw_matrix)
        label = 1
        if len(events) > 0 and events[0][2] == 96:
            label = 2
        # elif len(events) > 0 and events[0][2] == 80:
        #     label = 3
        y.append(label)

    # X = np.array(X)
    # y = np.array(y)

    return X, y

# Function to process non-error files


def process_file_nonerr(raw_fname):
    # Ignore warnings and set log level
    warnings.filterwarnings("ignore")
    mne.set_log_level('WARNING')

    # Read the raw data, apply filtering and extract events
    raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)
    raw.filter(l_freq=0.1, h_freq=30)
    events, event_id = mne.events_from_annotations(raw)

    # Define the interval for cropping the raw data
    interval = [events[2][0]/raw.info['sfreq'],
                events[-2][0]/raw.info['sfreq']]

    # Check if the interval contains exactly two events, otherwise raise an error
    if len(interval) != 2:
        return ValueError("The number of events is not 2")

    # Crop the raw data according to the defined interval
    raw.crop(tmin=interval[0], tmax=interval[1], include_tmax=False)

    # Select the EEG channels
    picks = mne.pick_types(raw.info, meg=False,
                           eeg=True, stim=False, eog=False)

    # Get omit_intervals
    omit_intervals = []
    event_id = {'80': 80, '96': 96}
    samples_96 = events[events[:, 2] == event_id['96'], 0]/raw.info['sfreq']

    # Populate the omit_intervals list
    for time_pt in samples_96:
        omit_intervals.append([time_pt - 3, time_pt + 5])

    # Start splitting
    length_of_data_in_sec = raw.times[-1]
    X = []
    y = []
    filter = 0
    for start_time in range(int(length_of_data_in_sec)):
        # Skip every third window
        if filter % 3 == 0:
            filter += 1
            continue

        # Check if the current window overlaps any of the omit_intervals
        if any([omit_start <= start_time < omit_end for omit_start, omit_end in omit_intervals]):
            continue  # Skip this window if it overlaps

        one_sec_raw = raw.copy().crop(tmin=start_time, tmax=start_time + 1)
        events, event_id = mne.events_from_annotations(one_sec_raw)

        one_sec_raw_matrix = one_sec_raw.get_data(picks=picks)
        X.append(one_sec_raw_matrix)
        label = 1
        y.append(label)
        filter += 1
    X = np.array(X)
    y = np.array(y)
    return X, y
