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
    print(raw.info["bads"])
