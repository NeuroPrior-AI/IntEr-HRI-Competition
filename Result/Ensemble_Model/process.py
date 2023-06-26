# mne imports
# from mne.datasets import sample

import numpy as np
import glob
import pickle
from process_file import process_file
##################### Process all data ######################
# data_path = sample.data_path()

# Process all .vhdr files in all subdirectories of data_folder
path = "/Users/zhezhengren/Desktop/NeuroPrior_AI/Model_Competion/EEG/training data/"
file_type = ".vhdr"
vhdr_files = glob.glob(f"{path}/**/*{file_type}", recursive=True)

all_data_labels = [process_file(filename) for filename in vhdr_files]

##################### Process, filter and epoch the data ######################
# After all files processed, concatenate all data and labels along the first axis
X = np.concatenate([data for data, labels in all_data_labels], axis=0)
y = np.concatenate([labels for data, labels in all_data_labels], axis=0)

# Check the dimension of the data
print("the shape of X is: " + str(X.shape) + "and the shape of y is: " + str(y.shape))
print("Number of distinct elements in y:", len(set(y)))
print("Max value in y:", max(y))
print("Min value in y:", min(y))

# Save all data
with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('y.pkl', 'wb') as f:
    pickle.dump(y, f)

##################### Process data by subject ######################
subjects = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']

for subject in subjects:
     subject_path = "../" + subject + "/training data"
     subject_files = glob.glob(f"{path}/**/*{file_type}", recursive=True)

     subject_data = [process_file(filename) for filename in subject_files]

     X = np.concatenate([data for data, labels in subject_data], axis=0)

     y = np.concatenate([labels for data, labels in subject_data], axis=0)

     # Save all data
     with open('data_by_subject/' + subject + '/X.pkl', 'wb') as f:
         pickle.dump(X, f)
     with open('data_by_subject/' + subject + '/y.pkl', 'wb') as f:
         pickle.dump(y, f)