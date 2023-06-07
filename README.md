# IntEr-HRI-Competition

# Attention
- Need to revise the File_Path (Including training data and EEG_Final.csv) in EEG_Final.py when run the scripts

# Task
- Begin to implement the time series transformer model
- Have a meeting with Bo Zhao to discuss how to use the dataset
- Communicate with Xuzhe and Paul Tang to solve the problem together

# 1. Data Processing
- In EEG/Data_Processing, there are file for the all data including individual (EEG_AA56D, EEG_AC17D.....) and combined file (EEG_Final.csv).
- The sampling frequency is 500 Hz, the time point will increase 0.002 for each.
- If there is a event, it will label the name of label. If there is no event, it will directly empty.

# 2. Transformer Model
1. source:
- https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
- https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/mobilepresent?slide=id.g31364026ad_3_2
