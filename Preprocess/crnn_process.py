# Import necessary libraries
import argparse
import os
import numpy as np
import glob
import pickle
from utils.process_file import process_file, process_file_sec


def main(args):
    # mapping: mapping of event ids to classes
    # (1, 2, 32, 64, 48, 80) -> no error, (96) -> error
    mapping = {1: 1, 2: 1, 32: 1, 64: 1, 48: 1, 80: 1, 96: 2}

    # Get a list of all .vhdr files in all subdirectories of the defined path
    vhdr_files = glob.glob(f"{args.path}/**/*{args.file_type}", recursive=True)

    # Process all .vhdr files and store the data and labels
    # X_list = [process_file_sec(filename, args.filter_type, length=args.length)[0] for filename in vhdr_files]
    # y_list = [process_file_sec(filename, args.filter_type, length=args.length)[1] for filename in vhdr_files]
    # max_size = max(x.shape[0] for x in X_list)
    # print("Max size of data in seconds: ", max_size)
    # # Align all data to the same size
    # X = np.zeros((len(X_list), max_size, 64, 501))
    # y = np.ones((len(y_list), max_size))
    # # Update X and y
    # for i in range(len(X_list)):
    #     X[i, :X_list[i].shape[0], :, :] = X_list[i]
    #     y[i, :y_list[i].shape[0]] = y_list[i]
    
    # # Print the dimensions of the data and labels
    # print("The shape of X is: " + str(X.shape) +
    #       "and the shape of y is: " + str(y.shape))
    # np.savetxt('y_values.txt', y, fmt='%d')

    
    X = [process_file_sec(filename, args.filter_type, length=args.length)[0] for filename in vhdr_files]
    y = [process_file_sec(filename, args.filter_type, length=args.length)[1] for filename in vhdr_files]

    # Print the dimensions of the data and labels
    # print("The shape of X is: " + str([len(Xi) for Xi in X]))
    # np.savetxt('y_values.txt', y, fmt='%d')

    # Save all data and labels
    with open('X_crnn.pkl', 'wb') as f:
        pickle.dump(X, f)
    with open('y_crnn.pkl', 'wb') as f:
        pickle.dump(y, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--path', type=str, default='../Dataset/training data',
                        help='path to the training data')
    parser.add_argument('--file_type', type=str, default='.vhdr',
                        help='file type of the training data')
    parser.add_argument('--filter_type', type=str, default='bandpass',
                        help='filter type, choose from [bandpass, butter, cheby, ellip]')
    parser.add_argument('--length', type=int, default=None,
                        help='length of data in seconds')
    main(parser.parse_args())

    print("Done!")