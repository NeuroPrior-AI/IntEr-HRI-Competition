import argparse
import glob
import re
from Algorithms.probmap import generate_top6

# Specify the test data path, precision, duration, and time range

ground_truths = {
    "AA56D_set6": [35438, 45679, 56132, 76437, 92538, 112236],
    "AA56D_set5": [38339,  52386,  80472,  93949, 110389, 127213],
    "AC17D_set5": [24574,  44832,  65498,  75827,  85830, 105950],
    "AC17D_set6": [32985,  42838,  59319,  85930,  99094, 115967],
    "AJ05D_set5": [29859,  44867,  59376,  73436,  94344, 111985],
    "AJ05D_set6": [34472,  46150,  57921,  98744, 111208, 130666],
    "AQ59D_set6": [30989,  45383,  80984,  94935, 105786, 116483],
    "AQ59D_set7": [20550,  32360,  51188,  62390,  73316, 119470],
    "AW59D_set6": [80025,  89766,  99353, 115699, 126243, 151322],
    "AW59D_set5": [28346,  53529,  72696,  82370,  94985, 104723],
    "AY63D_set5": [30827,  44573,  58691,  76108, 104608, 116030],
    "AY63D_set6": [33830,  48181,  68574,  88862, 106121, 123453],
    "BS34D_set6": [39528,  56616,  66819,  83440,  93522, 113575],
    "BS34D_set5": [29753,  39928,  50250,  83927,  94285, 110885],
    "BY74D_set5": [35875,  45760,  55458,  71575,  84799, 107807],
    "BY74D_set6": [63161,  79681,  99674, 116457, 126649, 143375]
}


def extract_pattern(filepath):
    # Use regex to directly find the pattern "[Alphabets][Numbers][Optional Alphabets]_set[Numbers]"
    match = re.search(r"([A-Za-z]+\d+[A-Za-z]?).*set(\d+)", filepath)

    if match:
        # Combine the matched groups with an underscore and 'set' prefix
        result = f"{match.group(1)}_set{match.group(2)}"
        return result
    else:
        return None


def main(args):
    vhdr_files = glob.glob(f"{args.path}/**/*{args.file_type}", recursive=True)

    # Open a text file for writing
    with open('Testing/output.txt', 'w') as file:
        score = 0
        for raw_fname in vhdr_files:
            temp_score = 0
            true_pos = 0
            key = extract_pattern(raw_fname)
            top6 = generate_top6(raw_fname, duration=1, precision=10,
                                 model_name=args.model, cla=1, tmin=-0.1, tmax=0.9, doPlot=False)
            
            # Write the key, top6, and ground_truths[key] to the file
            file.write(f"Set: {key}\n")
            file.write(f"Predicted: {top6}\n")
            file.write(f"Ground Truth: {ground_truths[key]}\n")
            
            diff = [x - y for x, y in zip(top6, ground_truths[key])]
            print("diff: ", diff)
            for i in diff:
                if i < 0 or i > 1000:
                    temp_score += 1000
                else:
                    temp_score += i
                    true_pos += 1
            file.write(f"Set Score: {temp_score}\n")
            file.write(f"No. of True Positives: {true_pos}\n")
            file.write(f"-----------------------------------\n")
            score += temp_score
        print("Final score: ", score)

    # score = 0
    # for raw_fname in vhdr_files:
    #     key = extract_pattern(raw_fname)
    #     top6 = generate_top6(raw_fname, duration=1, precision=10,
    #                          model_name=args.model, cla=1, tmin=-0.1, tmax=0.9, doPlot=False)
    #     diff = [x - y for x, y in zip(top6, ground_truths[key])]
    #     print("diff: ", diff)
    #     for i in diff:
    #         if i < 0 or i > 1000:
    #             score += 1000
    #         else:
    #             score += i
    # print("Final score: ", score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate probability map')
    parser.add_argument('--path', type=str, default='Dataset/test data/',
                        help='path to the test data')
    parser.add_argument('--file_type', type=str, default='.vhdr',
                        help='file type of the training data')
    parser.add_argument('--model', type=str, default='ensemble',
                        help='choose model from [ensemble, resnet]')
    main(parser.parse_args())
