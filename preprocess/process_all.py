# Import necessary libraries
import numpy as np
import glob
import pickle
from process_file import process_file

# Define the mapping for event ids
mapping = {64: 1, 32: 1, 48: 1, 96: 2, 80: 3, 1: 1, 2: 1}

# Define the time range for epochs
tmin = -0.1
tmax = 0.9

# Define the baseline period
baseline = -0.3, 0.0

# Define the directory path for training data and file type
path = "../training data"
file_type = ".vhdr"

# Get a list of all .vhdr files in all subdirectories of the defined path
vhdr_files = glob.glob(f"{path}/**/*{file_type}", recursive=True)

# Process all .vhdr files and store the data and labels
all_data_labels = [process_file(filename, tmin, tmax, event_id={
    'no error': 1, 'error': 2, 'reaction': 3}) for filename in vhdr_files]

# Concatenate all processed data and labels along the first axis
X = np.concatenate([data for data, labels in all_data_labels], axis=0)
y = np.concatenate([labels for data, labels in all_data_labels], axis=0)

# Print the dimensions of the data and labels
print("The shape of X is: " + str(X.shape) +
      "and the shape of y is: " + str(y.shape))
print("Number of distinct elements in y:", len(set(y)))
print("Max value in y:", max(y))
print("Min value in y:", min(y))

# Save all data and labels
with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('y.pkl', 'wb') as f:
    pickle.dump(y, f)

# List of subjects
subjects = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D',
            'AW59D', 'AY63D', 'BS34D', 'BY74D']

# Uncomment the following lines if you want to process and save data by subject

# for subject in subjects:
#     subject_path = "../" + subject + "/training data"
#     subject_files = glob.glob(f"{path}/**/*{file_type}", recursive=True)

#     subject_data = [process_file(filename) for filename in subject_files]

#     X = np.concatenate([data for data, labels in subject_data], axis=0)
#     y = np.concatenate([labels for data, labels in subject_data], axis=0)

#     # Save all data
#     with open('data_by_subject/' + subject + '/X.pkl', 'wb') as f:
#         pickle.dump(X, f)
#     with open('data_by_subject/' + subject + '/y.pkl', 'wb') as f:
#         pickle.dump(y, f)
