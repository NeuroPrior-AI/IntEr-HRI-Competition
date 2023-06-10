import os
import mne
import pandas as pd
import numpy as np

folders = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']
base_path = '/Users/zhezhengren/Desktop/NeuroPrior_AI/Model_Competion/EEG/training data/'
output_base_path = '/Users/zhezhengren/Desktop/NeuroPrior_AI/Model_Competion/EEG/Data_Processing/Data_Epoch_Range/'
event_codes_labels = {
    1: "S 1",
    32: "S 32",
    48: "S 48",
    64: "S 64",
    80: "S 80",
    96: "S 96"
}

os.makedirs(output_base_path, exist_ok=True)

for folder in folders:
    print("Processing folder:", folder)
    folder_path = os.path.join(base_path, folder, 'data')
    output_folder_path = os.path.join(output_base_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)

    df_folder = pd.DataFrame()  # Initialize a new DataFrame for each folder

    for filename in os.listdir(folder_path):
        if filename.endswith('.vhdr'):
            file_path = os.path.join(folder_path, filename)
            print('Analyzing file:', file_path)
            raw = mne.io.read_raw_brainvision(file_path, scale=1, preload=True)
            raw.filter(l_freq=0.1, h_freq=50)
            raw.set_montage('standard_1020', on_missing='ignore')

            raw.pick_types(eeg=True)

            data, times = raw.get_data(return_times=True)

            epochs = mne.Epochs(raw, mne.events_from_annotations(raw)[0], tmin=-0.1, tmax=0.9, preload=True)
            epochs.drop_bad()

            event_onset_samples = epochs.events[:, 0]
            event_codes = epochs.events[:, 2]

            event_names = [""] * raw.n_times
            for onset, code in zip(event_onset_samples, event_codes):
                if code in event_codes_labels.keys():
                    # prevent index error if the epoch exceeds the data length
                    start, end = int(onset + epochs.tmin*raw.info['sfreq']), int(onset + epochs.tmax*raw.info['sfreq'])
                    if end <= len(event_names):
                        event_names[start:end] = [event_codes_labels[code]] * (end - start)

            df_file = pd.DataFrame(data.T, columns=raw.ch_names)
            df_file.insert(0, 'Time_Point', np.arange(len(df_file.index)))
            df_file.insert(0, 'Event Name', event_names)
            df_file.insert(0, 'FileName', filename)

            # Save individual processed file
            df_file.to_csv(os.path.join(output_folder_path, f'{filename}_processed.csv'))

            df_folder = pd.concat([df_folder, df_file])  # Concatenate the new file's data to the folder's DataFrame

    # Save the combined file for the folder
    df_folder.to_csv(os.path.join(output_folder_path, f'{folder}_combined.csv'))

# Concatenate all dataframes and save as final CSV
final_df = pd.concat([pd.read_csv(os.path.join(output_base_path, folder, filename)) for folder in folders for filename in os.listdir(os.path.join(output_base_path, folder)) if filename.endswith('_combined.csv')])
final_df.to_csv(os.path.join(output_base_path, 'Data_Epoch.csv'))