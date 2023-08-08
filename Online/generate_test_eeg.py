# from pylsl import StreamInfo, StreamOutlet
# import mne
# import time

# raw_fname = "Dataset/training data/AA56D/data/20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr"
# # Load the EEG data
# raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)

# # Get the data and transpose it to the correct shape (channels x time)
# data = raw.get_data().T

# # Create info about the stream
# info = StreamInfo('MyEEG', 'EEG', raw.info['nchan'], raw.info['sfreq'], 'float32', 'myuid1234')

# # Create a StreamOutlet
# outlet = StreamOutlet(info)

# # Send each sample in the data one at a time
# for sample in data:
#     # Send the sample
#     print(sample)
#     outlet.push_sample(sample)
#     # Wait for 1/sfreq of a second
#     time.sleep(1.0/raw.info['sfreq'])
    
from pylsl import StreamInfo, StreamOutlet
import mne
import numpy as np
import time

def load_markers(marker_fname, sfreq):
    markers = {}
    with open(marker_fname, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Mk"):
                parts = line.strip().split(',')
                description = parts[1]
                position = int(parts[2])
                code = int(description.split('S')[-1].strip()) if 'S' in description else 0
                markers[position] = code
    return markers

# training dataset for AQ59D
#raw_fname = "Dataset/training data/AQ59D/data/20230421_AQ59D_orthosisErrorIjcai_multi_set2.vhdr"
#marker_fname = "Dataset/training data/AQ59D/data/20230421_AQ59D_orthosisErrorIjcai_multi_set2.vmrk"

# test dataset AQ59D test set 6
#raw_fname = "Dataset/test data/AQ59D/data/20230421_AQ59D_orthosisErrorIjcai_multi_set6.vhdr"
#marker_fname = "Dataset/test data/AQ59D/data/20230421_AQ59D_orthosisErrorIjcai_multi_set6.vmrk"

# test dataset AQ59D test set 7
raw_fname = "Dataset/test data/AQ59D/data/20230421_AQ59D_orthosisErrorIjcai_multi_set7.vhdr"
marker_fname = "Dataset/test data/AQ59D/data/20230421_AQ59D_orthosisErrorIjcai_multi_set7.vmrk"

# Load the EEG data
raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)

# Get the data and transpose it to the correct shape (channels x time)
data = raw.get_data().T

# Get the sampling frequency
sfreq = raw.info['sfreq']

# Load the markers from the VMRK file
markers = load_markers(marker_fname, sfreq)

# Delete the third, fourth, and fifth channels from the last in the EEG channel names
ch_names = raw.ch_names[:-3] + ['Label', 'Time']
info = StreamInfo('MyEEG', 'EEG', len(ch_names), raw.info['sfreq'], 'float32', 'myuid1234')

# Create a StreamOutlet
outlet = StreamOutlet(info)

# Get time information for each sample
times = raw.times

# List to collect samples with non-zero labels
non_zero_labels = []

# Iterate over samples
for i, sample in enumerate(data):
    # Determine the label based on the marker value
    label = markers.get(i, 0)

    # Get the time for this sample
    time_sample = times[i]

    # Delete the third, fourth, and fifth columns from the last
    sample = np.delete(sample, [-1, -2, -3])

    # Create a new sample array with original data, label, and time
    new_sample = np.concatenate((sample, [label, time_sample]))

    # Print all the details for every sample
    print(new_sample)

    # If label is not 0, add to list for later printing
    if label != 0:
        non_zero_labels.append((i, new_sample))

    # Send the sample
    outlet.push_sample(new_sample)
    
    # print("[INFO] sample shape", new_sample.shape)

    # Wait for 1/sfreq of a second
    time.sleep(1.0 / raw.info['sfreq'])
    
    # time.sleep(1)

# Print the samples with non-zero labels at the end
print("\nSamples with non-zero labels:")
for idx, sample_with_label in non_zero_labels:
    print(f"Sample index {idx}:")
    print(sample_with_label)

