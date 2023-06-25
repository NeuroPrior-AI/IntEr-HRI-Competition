# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from probmap_utils import generate_prob_map, get_96_timepts, get_64_32_timepts, get_80_timepts

# Specify the raw data file, precision, duration, and time range
raw_fname = "test data/AA56D/data/20230427_AA56D_orthosisErrorIjcai_multi_set5.vhdr"
precision = 20
duration = 1
tmin=-0.1
tmax=0.9

# Generate probability maps for events 96 and 80 using the ensemble model
# prob_map_96 = generate_prob_map(raw_fname, duration, precision, "ensemble", 1, tmin, tmax)
# prob_map_80 = generate_prob_map(raw_fname, duration, precision, "ensemble", 2, tmin, tmax)

# Generate probability maps for events 96 and 80 using the ResNet model
prob_map_96 = generate_prob_map(raw_fname, duration, precision, "resnet", 1, tmin, tmax)
# prob_map_80 = generate_prob_map(raw_fname, duration, precision, "resnet", 2, tmin, tmax)

# Get the time points for events 96 and 80
s96_positions = get_96_timepts(raw_fname)
s80_positions = get_80_timepts(raw_fname)

# Generate time array for x-axis of the plot
time = np.arange(0, len(prob_map_96)/precision*duration, duration/precision)

# Plot the probability maps
plt.figure(figsize=(100, 5))
plt.plot(time, prob_map_96, label='96', color = 'b')
# plt.plot(time, prob_map_80, label='80', color = 'r')
plt.xticks(np.arange(min(time), max(time)+1, 1))  # Set x-ticks every 1 unit
plt.xlabel('Time (sec)')
plt.ylabel('Probability')

# Add vertical lines and labels for events 96 and 80
for xc in s96_positions:
    plt.axvline(x=xc, color=(0.6, 0.6, 1), linestyle='--', linewidth=0.5)
    plt.text(xc, 1, '96', color=(0.6, 0.6, 1) , rotation=90, verticalalignment='center')

for xc in s80_positions:
    plt.axvline(x=xc, color=(1, 0.6, 0.6), linestyle='--', linewidth=0.5)
    plt.text(xc, 1, '80', color=(1, 0.6, 0.6), rotation=90, verticalalignment='center')

# Add legend and save the plot
plt.legend()
plt.savefig('probmap.png')