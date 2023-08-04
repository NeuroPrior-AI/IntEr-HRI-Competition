"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

import time
from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np 
from time import perf_counter 
import requests
import mne
import warnings
import warnings
from Preprocess.utils.filter import Filter
import argparse
import os
import glob
import pickle
from scipy.signal import butter, cheby1, ellip, sosfilt
import pywt
import mne

from Algorithms.utils.resnet_predict import resnet_predict

def printMeta(stream_info_obj):
    """
    This function prints some basic meta data of the stream
    """
    print("") 
    print("Meta data")
    print("Name:", stream_info_obj.name())
    print("Type:", stream_info_obj.type())
    print("Number of channels:", stream_info_obj.channel_count())
    print("Nominal sampling rate:", stream_info_obj.nominal_srate())
    print("Channel format:",stream_info_obj.channel_format())
    print("Source_id:",stream_info_obj.source_id())
    print("Version:",stream_info_obj.version())
    print("")

def getRingbufferValues(chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer): 
    """
    This function provides the most recent data samples and timestamps in a ringbuffer 
    (first val is oldest, last the newest) 

    Attributes:
        chunk               : current data chunk
        timestamps          : LSL local host timestamp for the data chunk 
        current_local_time  : LSL local client timestamp when the chunk is received
        timestamp offset    : correction factor that needs to be added to the timestamps to map it into the client's local LSL time
        data_buffer         : data buffer array of shape (buffer_size, n_channels)
        timestamp_buffer    : timestamps buffer of shape (buffer_size, 3). The 3 columns correspond to the host timestamp, the client local time and time correction offset resp.

    Returns:
        data_buffer         : data buffer array of shape (buffer_size, n_channels)
        timestamp_buffer    : timestamp buffer of shape (buffer_size, 3)
    """
    #data 
    current_chunk = np.array(chunk)
    n_samples = current_chunk.shape[0] # shape (samples, channels)

    temp_data = data_buffer[n_samples:, :]
    data_buffer[0:temp_data.shape[0], :] = temp_data
    data_buffer[temp_data.shape[0]:, :] = current_chunk

    # timestamps 
    current_timestamp_buffer = np.array(timestamps)

    temp_time = timestamp_buffer[n_samples:, 0]
    timestamp_buffer[0:temp_time.shape[0], 0] = temp_time
    timestamp_buffer[temp_time.shape[0]:, 0] = current_timestamp_buffer

    # current local time and offset correction 
    temp_local_time = timestamp_buffer[n_samples:, 1]
    timestamp_buffer[0:temp_local_time.shape[0], 1] = temp_local_time
    timestamp_buffer[temp_local_time.shape[0]:, 1] = current_local_time

    temp_offset_time = timestamp_buffer[n_samples:, 2]
    timestamp_buffer[0:temp_offset_time.shape[0], 2] = temp_offset_time
    timestamp_buffer[temp_offset_time.shape[0]:, 2] = timestamp_offset

    return data_buffer, timestamp_buffer

def sendDetectedError(team_name, secret_id, timestamp_buffer_vals, local_clock_time):
    """
    This function gathers all the relevant results and sends it to the host.
    This function should be called everytime an error is detected.

    Attributes:
        team_name (str)         : each team will be assigned a team name which 
        secret_id (str)         : each team will be provided with a secret code
        timestamp_buffer_vals   : subset of the timestamp_buffer array at the instant when you have predicted an error and want to send the current result. Basically the i-th element of the timestamp_buffer array
        local_clock_time        : current LSL local clock time when you have run your classifier and predicted an error. This can be determined with the help of "local_clock()" call.
    """
    
    # calculate the final values for the timings 
    comm_delay = timestamp_buffer_vals[1] -timestamp_buffer_vals[0] -timestamp_buffer_vals[2]
    computation_time = local_clock_time - timestamp_buffer_vals[1]

    # connection to API for sending the results online 
    url = 'http://10.250.223.221:5000/results'
    myobj = {'team': team_name,
            'secret': secret_id,
            'host_timestamp': timestamp_buffer_vals[0], 
            'comp_time': computation_time, 
            'comm_delay': comm_delay}

    x = requests.post(url, json = myobj)


import mne
import numpy as np
import warnings
from mne import create_info
from mne.io import RawArray

def process_data_buffer(data_buffer, sfreq, ch_names, filter_type='bandpass'):

    # Determine the number of channels
    print(data_buffer.shape)

    # Create a minimal Info object
    # info = create_info(ch_names=ch_names, sfreq=sfreq)
    ch_types = ['eeg'] * len(ch_names) # Assuming all channels are EEG channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


    # Create Raw object from the data buffer
    raw = RawArray(data_buffer[:,0:64].T, info)

    # Pick channels by type (in this case, EEG channels)
    picks = mne.pick_types(info, meg=False, eeg=True)
    print("info:", info)

    # Apply filtering to the data (only to the channels specified by picks)
    if filter_type == 'bandpass':
        raw.filter(l_freq=1, h_freq=30, picks=picks) # Adjust frequency bands as needed

    # Other preprocessing steps can be applied here if necessary

    return raw.get_data(picks=picks)


def main():

    #************************************************************
    # ********************** user params ************************
    #************************************************************

    #params 
    buffer_size = 501  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
    dt_read_buffer= 0.04 # time in seconds how often the buffer is read  (updated with new incoming chunks)

    # team info
    team_name= 'NeuroPrior AI'
    
    #secret id
    secret_id = 'NeuroPrior AI' # Needed revised the sercet id here
    
    #************************************************************
    #************************************************************
    #************************************************************

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG') # create data stream 

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0]) 
    stream_info = inlet.info()
    printMeta(stream_info) # print stream info 

    # run continiously 
    running = True


    # uncomment if data should be recorded (not necessary for online prediction, use buffer for that)
    #data_arr = []
    #time_stamp_arr = []

    #inits 
    data_buffer = np.zeros((buffer_size, stream_info.channel_count())) # data buffer has shape (buffer_size, n_channels) last 2 channels are only markers and indices ! 
    timestamp_buffer = np.zeros((buffer_size, 3)) # buffer for different kinds of timestamps or time values 

    # get timestamp offset 
    timestamp_offset = inlet.time_correction()

    detected_time = []
    while running:
       
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk

        if(chunk): # if list not empty (new data)
            
            # get timing info 
            current_local_time = local_clock()
            timestamp_offset = inlet.time_correction()
            print("chunk shape:", len(chunk))
            
            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            data_buffer, timestamp_buffer = getRingbufferValues(chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer) 
            #************************************************************
            #*************************** Data to use ********************
            #************************************************************

            # data_buffer can be used for further processing and classification (maybe use parallel processing to not slow down the data access for longer model prediction times)
            
            # Inside the while loop after updating data_buffer

            #************************************************************
            #************************* Data Preprocessing ****************
            #************************************************************
            # Apply necessary preprocessing to the data in data_buffer.
            # E.g., apply specific filters, normalization, feature extraction, etc.

            # Assuming that raw_fname is the path to the raw EEG file in BrainVision format.
            # You may need to adjust the parameters and function calls according to your use case.

            # Inside the main while loop after updating data_buffer

            # Set the sampling frequency and channel names according to your data
            sfreq = 500  # Adjust based on your data's sampling rate
            ch_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 
                        'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8',
                        'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4', 'P8',
                        'PO9', 'O1', 'OZ', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1',
                        'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2',
                        'C6', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
                        'PO3', 'POZ', 'PO4', 'PO8'] # List of channel names in the order of your data

            processed_data = process_data_buffer(data_buffer, sfreq, ch_names)

            # Continue with error detection and other steps

            #************************************************************
            #************************* Error Detection *******************
            #************************************************************

            # Utilize a pre-trained machine learning model, statistical methods,
            # or other techniques to predict errors based on the processed data.

            # Assuming that you have a pre-trained model for error detection
            # You must load the trained model and use it to predict errors in the processed_data

            # from sklearn.externals import joblib
            
            from joblib import load

            # model_path = 'Models/pre-trained/Ensemble.joblib' # TODO: Update the path to your trained model
            # model = load(model_path)

            # print("shape is:", processed_data)
            # predictions = model.predict(np.expand_dims(processed_data, axis=0))
            # print(predictions)

            pred_prob_i = resnet_predict(np.expand_dims(processed_data, axis=0))[:, 1]
            print("pred_prob_i: ", pred_prob_i)
            print("detected_time: ", detected_time)

            # error_detected = 2 in predictions # 2 is assumed to represent error
            # error_index_in_buffer = np.where(predictions == 2)[0][0] if error_detected else None
            threshold = 0.9
            error_detected = pred_prob_i[0] >= threshold
            
            #************************************************************
            #*********************** Send Detected Error *****************
            #*************************************************************
            
            if error_detected:
                # if an error was detected, use the following lines to send the timepoint (timestamp) of detection
                local_clock_time = local_clock()
                error_index_in_buffer = 500 # calculate the error index in the current data buffer. Arrays timestamp_buffer and data_buffer are related to each other (same data points) 
                # sendDetectedError(team_name, secret_id, timestamp_buffer[error_index_in_buffer, :], local_clock_time)
            

                # wait for some time to ensure a "fixed" frequency to read new data from buffer 
                while((perf_counter()-old_time) < dt_read_buffer): 
                    pass

                # just for checking the loop frequency 
                print("time: ", (perf_counter()-old_time)*1000)
                
                # uncomment to record ALL data received (not required for participants)
                # data_arr = data_arr+chunk
                # time_stamp_arr = time_stamp_arr + timestamps

        old_time = perf_counter()
    print("detected_time: ", detected_time)
    


if __name__ == '__main__':
    main()
