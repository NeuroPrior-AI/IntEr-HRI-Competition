#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:44:00 2023

@author: zhezhengren
"""

import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folders = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']
base_path = 'C:/Users/PaulS/Desktop/IntErHRI_data/EEG/training data/'
output_base_path = 'C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch/'
event_codes_labels = {  # Name:                                         # Occurence:
    1: "S 1",           # start/end of the EEG recording                2/68
    32: "S 32",         # start of extension movement                   15/68
    48: "S 48",         # marker for trial with no errors               24/68
    64: "S 64",         # start of flexion movement                     15/68
    80: "S 80",         # subject squeezed the ball                     6/68
    96: "S 96"          # error introduced                              6/68
}

os.makedirs(output_base_path, exist_ok=True)

for folder in folders:
    print("Processing folder:", folder)
    folder_path = os.path.join(base_path, folder, 'data')
    output_folder_path = os.path.join(output_base_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith('.vhdr'):
            file_path = os.path.join(folder_path, filename)
            print('Analyzing file:', file_path)
            raw = mne.io.read_raw_brainvision(file_path, scale=1, preload=True)
            
            raw.filter(l_freq=0.1, h_freq=50)
            raw.set_montage('standard_1020', on_missing='ignore')

            epochs = mne.Epochs(raw, mne.events_from_annotations(raw)[0], tmin=-0.1, tmax=0.9, preload=True)

            epochs.drop_bad()

            # Pick only EEG channels
            epochs.pick_types(eeg=True)

            epochs_data = epochs.get_data()
            n_trials, n_channels, n_time_points = epochs_data.shape

            epochs_data_2d = epochs_data.reshape(n_trials * n_time_points, n_channels)
            df = pd.DataFrame(epochs_data_2d, columns=epochs.ch_names)

            trial_index = np.repeat(np.arange(n_trials)+1, n_time_points)

            event_names_1d = [""]*n_trials*n_time_points
            for i in range(n_trials):
                event_code = epochs.events[i, 2]
                if event_code in event_codes_labels.keys():
                    event_names_1d[i * n_time_points:(i + 1) * n_time_points] = [event_codes_labels[event_code]] * n_time_points
                else:
                    print(f"Unrecognized event code at trial {i}: {event_code}")

            df.insert(0, 'Trial Index', trial_index)
            df.insert(1, 'Time Point', np.tile(np.arange(n_time_points), n_trials) + 1)
            df.insert(len(df.columns), 'Event Name', event_names_1d)

            df.to_csv(os.path.join(output_folder_path, f'{filename}_combined.csv'))

# Concatenate all dataframes and save as final CSV
final_df = pd.concat([pd.read_csv(os.path.join(output_base_path, folder, filename)) for folder in folders for filename in os.listdir(os.path.join(output_base_path, folder)) if filename.endswith('_combined.csv')])
final_df.to_csv(os.path.join(output_base_path, 'EEG_Epoch.csv'))
