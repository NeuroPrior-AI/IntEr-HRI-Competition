import os
import mne
import pandas as pd
import numpy as np

folders = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']
base_path = 'C:/Users/PaulS/Desktop/IntErHRI_data/EEG/training data/'
event_codes_labels = {  # Name:                                         # Occurence:
    1: "S 1",           # start/end of the EEG recording                2/68
    32: "S 32",         # start of extension movement                   15/68
    48: "S 48",         # marker for trial with no errors               24/68
    64: "S 64",         # start of flexion movement                     15/68
    80: "S 80",         # subject squeezed the ball                     6/68
    96: "S 96"          # error introduced                              6/68
}

for folder in folders:
    print("Processing folder:", folder)
    folder_path = os.path.join(base_path, folder, 'data')
    df_folder = pd.DataFrame()  # Initialize a new DataFrame for each folder

    for filename in os.listdir(folder_path):
        if filename.endswith('.vhdr'):
            file_path = os.path.join(folder_path, filename)
            print('Analyzing file:', file_path)
            raw = mne.io.read_raw_brainvision(file_path, scale=1, preload=True)

            # Print sampling rate
            print('The sampling frequency is: ', raw.info['sfreq'], 'Hz')

            raw.filter(l_freq=0.1, h_freq=50)
            raw.set_montage('standard_1020',on_missing='ignore')

            events = mne.events_from_annotations(raw)[0]

            raw.pick_types(eeg=True)

            times = np.arange(raw.n_times) / raw.info['sfreq']  # time in seconds
            data, times = raw.get_data(return_times=True)
            events_from_annot, event_dict = mne.events_from_annotations(raw)

            event_names = np.empty(raw.n_times, dtype='<U5')  # create empty array of strings
            for sample, _, code in events:
                if code in event_codes_labels.keys():
                    event_names[sample] = event_codes_labels[code]

            # Transpose the data to match your format
            df_file = pd.DataFrame(data.T, columns=raw.ch_names)
            df_file.insert(0, 'Time_Point', times)
            df_file.insert(0, 'Event Name', event_names)
            df_file.insert(0, 'FileName', filename)

            df_folder = pd.concat([df_folder, df_file])  # Concatenate the new file's data to the folder's DataFrame

    # Save individual CSV for each folder
    df_folder.to_csv(f'C:/Users/PaulS/Desktop/IntErHRI_data/csv data/EEG_{folder}.csv')

# Concatenate all dataframes and save as final CSV
final_df = pd.concat([df_folder for folder in folders])
final_df.to_csv('C:/Users/PaulS/Desktop/IntErHRI_data/csv data/EEG_FINAL.csv')
