import glob
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from mne.viz import plot_epochs_image

path = "../Dataset/training data"
file_type = ".vhdr"
vhdr_files = glob.glob(f"{path}/**/*{file_type}", recursive=True)


for file in vhdr_files:
    mapping = {1: 1, 2: 1, 32: 1, 64: 1, 48: 1, 80: 1, 96: 2}
    event_id={'non-P300': 1, 'P300': 2}
    raw = mne.io.read_raw_brainvision(file, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg=False,
                            eeg=True, stim=False, eog=False)
    raw.filter(1, 30, fir_design="firwin")

    # Extract events and remap the event ids
    events = mne.events_from_annotations(raw)[0]
    events[:, 2] = np.vectorize(lambda x: mapping.get(x, -1))(events[:, 2])

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=2,
                            proj=False, picks=picks, baseline=None, preload=True, verbose=False)

    # Plot image epoch before xdawn
    fig = plot_epochs_image(epochs["P300"], picks=[34], vmin=-30, vmax=30)

    for i in fig:
        # save the figure to /figures
        i.savefig(f"figures/{file.split('/')[-1].split('.')[0]}_eeg.png")