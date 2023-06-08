# IntEr-HRI-Competition

# Attention
- Need to revise the File_Path (Including training data and EEG_Final.csv) in EEG_Final.py and EEG_Epoch.py when run the scripts

# Task
- Begin to implement the time series transformer model
- Have a meeting with Bo Zhao to discuss whether the dataset is good and how to use the dataset
- Communicate with Xuzhe and Paul Tang to solve the problem together

# 1. Data_Processing
- In EEG/Data_Processing, there are file for the all data including individual (EEG_AA56D, EEG_AC17D.....) and combined file (EEG_Final.csv).
- The sampling frequency is 500 Hz, the time point will increase 0.002 for each.
- If there is a event, it will label the name of label. If there is no event, it will directly empty.

# 2. Epoch
- When you run EEG_epoch.py scripts, you will get the file struture as below
- Epoch is from -0.1s to 0.9s

```plaintext
output_base_path
│
└───EEG_Epoch.csv
│
└───AA56D
│   │   set1_combined.csv
│   │   set2_combined.csv
│   │   set3_combined.csv
│   │   ...
│
└───AC17D
│   │   set1_combined.csv
│   │   set2_combined.csv
│   │   set3_combined.csv
│   │   ...
```
- EEG-Epoch.csv contain all of the set_combined.csv in all folder
- set1_combined.csv will contain the data for set1.vhdr


# 3. Transformer_Model
1. source:
- https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
- https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/mobilepresent?slide=id.g31364026ad_3_2
