import os
import random

root_directory = "C:/Users/PaulS/Desktop/IntErHRI_data/csv_epoch"


if __name__ == "__main__":
    csv_files = []

    # Iterate through subdirectories in the root directory
    for directory in os.listdir(root_directory):
        directory_path = os.path.join(root_directory, directory)

        # Check if the item in the root directory is a subdirectory
        if os.path.isdir(directory_path):
            # Get the list of CSV files in the subdirectory
            files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

            # Select two random CSV file names from the list, if available
            if len(files) >= 2:
                selected_files = random.sample(files, 2)
                csv_files.extend(selected_files)
            elif len(files) == 1:
                csv_files.extend(files)

    # Print the list of randomly selected CSV file names
    print(csv_files)