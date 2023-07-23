import re

import numpy as np

if __name__ == "__main__":
    log_dir = "../tmp/crossval_logs/resnet_cycliclr_80epochs_68batch"

    participants = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']

    for p in participants:
        print(f'Started with participant {p}.')
        p_dir = log_dir + f"/{p}"

        accs = []

        for trial in range(10):
            t_dir = p_dir + f"/trial{trial}"

            # Open the file
            with open(t_dir + '/training_log.txt', 'r') as file:
                # Get the line we want to extract a number from
                lines = file.readlines()

                saved_model_line_index = None

                # Look for the index of last occurrence of "Saved model at"
                for i, line in enumerate(lines):
                    if "Saved model at" in line:
                        saved_model_line_index = i

                line = lines[saved_model_line_index + 1]

                # Find and extract the floating point number using regex
                score = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                # If a score was found, add it to the array
                if score:
                    accs.append(float(score[-3]))
                else:
                    print(f'Error parsing score.')

        mean = np.mean(accs) / 100
        std_dev = np.std(accs) / 100

        # print(f'Mean: {mean:.2f}')
        # print(f'Standard Deviation: {std_dev:.2f}')
        print(f'{mean:.4f} ± {std_dev:.4f}')

