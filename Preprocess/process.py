# Import necessary libraries
import argparse
import numpy as np
import glob
import pickle
from utils.process_file import process_file


def main(args):
    # mapping: mapping of event ids to classes
    # (1, 2, 32, 64, 48, 80) -> no error, (96) -> error
    mapping = {1: 1, 2: 1, 32: 1, 64: 1, 48: 1, 80: 1, 96: 2}

    # Get a list of all .vhdr files in all subdirectories of the defined path
    vhdr_files = glob.glob(f"{args.path}/**/*{args.file_type}", recursive=True)

    # Process all .vhdr files and store the data and labels
    all_data_labels = [process_file(filename, args.tmin, args.tmax, mapping, event_id={
        'non-P300': 1, 'P300': 2}, filter_type=args.filter_type) for filename in vhdr_files]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--path', type=str, default='../Dataset/training data',
                        help='path to the training data')
    parser.add_argument('--file_type', type=str, default='.vhdr',
                        help='file type of the training data')
    parser.add_argument('--tmin', type=float, default=-0.1,
                        help='start time of epochs')
    parser.add_argument('--tmax', type=float, default=0.9,
                        help='end time of epochs')
    parser.add_argument('--baseline', type=tuple, default=(-0.3, 0.0),
                        help='baseline period')
    parser.add_argument('--filter_type', type=str, default='bandpass',
                        help='filter type, choose from [bandpass, butter, cheby, ellip]')
    main(parser.parse_args())
