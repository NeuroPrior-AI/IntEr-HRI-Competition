from pylsl import StreamInfo, StreamOutlet
import mne
import time

raw_fname = "Dataset/training data/AA56D/data/20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr"
# Load the EEG data
raw = mne.io.read_raw_brainvision(raw_fname, preload=True, verbose=False)

# Get the data and transpose it to the correct shape (channels x time)
data = raw.get_data().T

# Create info about the stream
info = StreamInfo('MyEEG', 'EEG', raw.info['nchan'], raw.info['sfreq'], 'float32', 'myuid1234')

# Create a StreamOutlet
outlet = StreamOutlet(info)

# Send each sample in the data one at a time
for sample in data:
    # Send the sample
    outlet.push_sample(sample)
    # Wait for 1/sfreq of a second
    time.sleep(1.0/raw.info['sfreq'])
