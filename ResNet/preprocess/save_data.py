import os
import numpy as np
import pandas as pd

val_list = [  # selected using select_val_data.py
    '20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr_combined.csv',
    '20230427_AA56D_orthosisErrorIjcai_multi_set9.vhdr_combined.csv',
    '20230424_AC17D_orthosisErrorIjcai_multi_set9.vhdr_combined.csv',
    '20230424_AC17D_orthosisErrorIjcai_multi_set10.vhdr_combined.csv',
    '20230426_AJ05D_orthosisErrorIjcai_multi_set3.vhdr_combined.csv',
    '20230426_AJ05D_orthosisErrorIjcai_multi_set10.vhdr_combined.csv',
    '20230421_AQ59D_orthosisErrorIjcai_multi_set4.vhdr_combined.csv',
    '20230421_AQ59D_orthosisErrorIjcai_multi_set5.vhdr_combined.csv',
    '20230425_AW59D_orthosisErrorIjcai_multi_set8.vhdr_combined.csv',
    '20230425_AW59D_orthosisErrorIjcai_multi_set10.vhdr_combined.csv',
    '20230425_AY63D_orthosisErrorIjcai_multi_set7.vhdr_combined.csv',
    '20230425_AY63D_orthosisErrorIjcai_multi_set8.vhdr_combined.csv',
    '20230426_BS34D_orthosisErrorIjcai_multi_set4.vhdr_combined.csv',
    '20230426_BS34D_orthosisErrorIjcai_multi_set11.vhdr_combined.csv',
    '20230424_BY74D_orthosisErrorIjcai_multi_set1.vhdr_combined.csv',
    '20230424_BY74D_orthosisErrorIjcai_multi_set10.vhdr_combined.csv'
]

event_code_to_index = {         # 6 classes
    "S 1": 0,
    "S 32": 1,
    "S 48": 2,
    "S 64": 3,
    "S 80": 4,
    "S 96": 5
}

event_code_to_index2 = {        # 3 classes
    "S 1": 0,
    "S 32": 0,
    "S 48": 0,
    "S 64": 0,
    "S 80": 1,
    "S 96": 2
}

event_code_to_index3 = {        # 2 classes
    "S 1": 0,
    "S 32": 0,
    "S 48": 0,
    "S 64": 0,
    "S 80": 0,
    "S 96": 1
}


def load_csv(csv, code_to_index_map):
    """
    Load the eeg data from csv file to numpy arrays. data is a 3D np array and label is a 2D one-hot array.
    :param csv: path of csv file
    :param code_to_index_map: number of classes for classification
    :return: the data and labels of the eeg data
    """
    # Read the CSV file using pandas
    data_frame = pd.read_csv(csv)

    # Extract the values from the DataFrame
    data_values = data_frame.values

    num_trials = data_values[-1, 1]
    num_classes = len(set(code_to_index_map.values()))
    samples_per_second = 501
    eeg_channels = 64

    data = np.empty((num_trials, eeg_channels, samples_per_second))     # default type: float64
    label = np.empty((num_trials, num_classes), dtype=int)

    for i in range(num_trials):
        data[i, :, :] = data_values[i * samples_per_second:(i + 1) * samples_per_second, 3:3 + eeg_channels].T
        class_index = code_to_index_map[data_values[i * samples_per_second, -1]]
        label[i] = np.eye(num_classes)[class_index]

    return data, label


def save_data(root_directory, code_to_index_map, val_list):
    """
    Returns X of size: num trials x 64 eeg channels x 1 second (501), y of size: num trials x 1. Similar for X_val and y_val.
    :param code_to_index_map: number of classes for classification
    :param root_directory: root dir of all patient folders.
    :param val_list: list of validation csvs.
    :return: X, y, X_val, y_val
    """
    X = None
    y = None
    X_val = None
    y_val = None

    # Iterate through subdirectories in the root directory
    for directory in os.listdir(root_directory):
        directory_path = os.path.join(root_directory, directory)

        # Check if the item in the root directory is a subdirectory
        if os.path.isdir(directory_path):
            # Get the list of CSV files in the subdirectory
            files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

            for file in files:
                file_path = os.path.join(directory_path, file)

                data, label = load_csv(file_path, code_to_index_map)

                if file in val_list:
                    X_val = np.concatenate((X_val, data), axis=0) if X_val is not None else data
                    y_val = np.concatenate((y_val, label), axis=0) if y_val is not None else label
                else:
                    X = np.concatenate((X, data), axis=0) if X is not None else data
                    y = np.concatenate((y, label), axis=0) if y is not None else label

    return X, y, X_val, y_val


if __name__ == "__main__":
    # Debug:
    # csv = "C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch/AA56D/20230427_AA56D_orthosisErrorIjcai_multi_set1.vhdr_combined.csv"
    # load_csv(csv, event_code_to_index2)

    X, y, X_val, y_val = save_data("C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch", event_code_to_index3, val_list)
    np.save('../tmp/data/X.npy', X)
    np.save('../tmp/data/y.npy', y)
    np.save('../tmp/data/X_val.npy', X_val)
    np.save('../tmp/data/y_val.npy', y_val)
    print(f'Saved data.')


