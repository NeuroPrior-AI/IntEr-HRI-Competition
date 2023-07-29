# Import necessary packages
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Algorithms.utils.probmap_utils import generate_prob_map, get_96_timepts, get_start_time, get_80_timepts


# Specify the raw data file, precision, duration, and time range
raw_fname = "Dataset/test data/BY74D/data/20230424_BY74D_orthosisErrorIjcai_multi_set6.vhdr"
precision = 10
duration = 1
tmin = -0.1
tmax = 0.9


def generate_top6(raw_fname, duration, precision, model_name, cla=1, tmin=-0.1, tmax=0.9, doPlot=True):
    # Generate probability maps for events 96 and 80 using the ensemble model
    if model_name == 'ensemble':
        prob_map_96 = generate_prob_map(
            raw_fname, duration, precision, "ensemble", cla, tmin, tmax)
    elif model_name == 'resnet':
        prob_map_96 = generate_prob_map(
            raw_fname, duration, precision, "resnet", cla, tmin, tmax)
    else:
        raise ValueError("Model not found")

    # -----------------------------Find top windows (in sec)-----------------------------#
    gap = precision * 5

    # Compute rolling sums
    rolling_sums = np.array([np.sum(prob_map_96[i:i+precision])
                            for i in range(len(prob_map_96) - precision + 1)])

    # Sort indices by rolling sum
    indices = np.argsort(rolling_sums)[::-1]

    # Keep track of top windows and blacklist of excluded timepoints
    top_windows = []
    blacklist = set()

    for idx in indices:
        # Check if this window overlaps with a blacklisted timepoint
        if any(i in blacklist for i in range(idx, idx + precision)):
            continue

        # Add this window to our top windows
        top_windows.append(idx)

        # Add all points within gap after this window to blacklist
        blacklist.update(range(idx - gap, idx + precision + gap))

        if len(top_windows) >= 6:
            break

    # --------------------------------Plot probability map-------------------------------#
    if doPlot == True:
        # Get the time points for events 96 and 80
        s96_positions = get_96_timepts(raw_fname)
        s80_positions = get_80_timepts(raw_fname)

        # Generate time array for x-axis of the plot
        time = np.arange(0, len(prob_map_96)/precision * duration, duration/precision)
        # Plot the probability maps
        plt.figure(figsize=(20, 5))
        plt.plot(time, prob_map_96, label='96', color='black')
        # plt.plot(time, prob_map_80, label='80', color = 'r')
        # plt.plot(time, prob_map_96, label='sum', color='black')
        # Set x-ticks every 1 unit
        plt.xticks(np.arange(min(time), max(time)+1, 5))
        plt.xlabel('Time (sec)')
        plt.ylabel('Probability')

        # Add vertical lines and labels for events 96 and 80 (for training data)
        for xc in s96_positions:
            plt.axvline(x=xc, color=(0.6, 0.6, 1),
                        linestyle='--', linewidth=0.5)
            plt.text(xc, 1, '96', color=(0.6, 0.6, 1),
                     rotation=90, verticalalignment='center')

        for xc in s80_positions:
            plt.axvline(x=xc, color=(1, 0.6, 0.6),
                        linestyle='--', linewidth=0.5)
            plt.text(xc, 1, '80', color=(1, 0.6, 0.6),
                     rotation=90, verticalalignment='center')

        # Add vertical lines and labels for predicted timepoints
        for xc in top_windows:
            plt.axvline(x=xc/precision + 3/precision, color="black",
                        linestyle='--', linewidth=0.5)
            plt.text(xc/precision + 3/precision, 2, 'p', color="black",
                     rotation=90, verticalalignment='center')
        
        # Add legend and save the plot
        plt.legend()
        plt.savefig('Algorithms/figures/probmap.png')

    # ---------------------------Convert time into sample indices---------------------------#
    start = get_start_time(raw_fname)
    offset = 500 / precision * 3
    top_windows = [start + x * 500/precision + offset for x in top_windows]

    # sort the top windows in ascending order
    top_windows.sort()
    # convert top_windows to int from float
    top_windows = [int(x) for x in top_windows]

    print("Top windows: ", top_windows)

    return top_windows


def main(args):
    generate_top6(raw_fname, duration, precision, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate probability map')
    parser.add_argument('--model', type=str, default='ensemble',
                        help='choose model from [ensemble, resnet]')
    main(parser.parse_args())
