import os
import numpy as np
from save_data import load_csv, event_code_to_index3


def save_data(root_directory, code_to_index_map):
    # Iterate through subdirectories in the root directory
    for directory in os.listdir(root_directory):
        directory_path = os.path.join(root_directory, directory)
        X = None
        y = None

        # Check if the item in the root directory is a subdirectory
        if os.path.isdir(directory_path):
            # Get the list of CSV files in the subdirectory
            files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

            for file in files:
                file_path = os.path.join(directory_path, file)

                data, label = load_csv(file_path, code_to_index_map)

                X = np.concatenate((X, data), axis=0) if X is not None else data
                y = np.concatenate((y, label), axis=0) if y is not None else label

            np.save(f'../tmp/cross_validation_data/X_{directory}.npy', X)
            np.save(f'../tmp/cross_validation_data/y_{directory}.npy', y)


if __name__ == "__main__":
    root_directory = "C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch"
    # Save data (only need to execute on first run)
    save_data("C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch", event_code_to_index3)
    print("Done.")
