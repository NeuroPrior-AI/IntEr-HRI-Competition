"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

from mne.io import RawArray
from mne import create_info
import numpy as np
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
    print("Channel format:", stream_info_obj.channel_format())
    print("Source_id:", stream_info_obj.source_id())
    print("Version:", stream_info_obj.version())
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
    # data
    current_chunk = np.array(chunk)
    n_samples = current_chunk.shape[0]  # shape (samples, channels)
    print("[DEBUG] n_samples:", n_samples)

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
    comm_delay = timestamp_buffer_vals[1] - \
        timestamp_buffer_vals[0] - timestamp_buffer_vals[2]
    computation_time = local_clock_time - timestamp_buffer_vals[1]

    # connection to API for sending the results online
    url = 'http://10.250.223.221:5000/results'
    myobj = {'team': team_name,
             'secret': secret_id,
             'host_timestamp': timestamp_buffer_vals[0],
             'comp_time': computation_time,
             'comm_delay': comm_delay}

    x = requests.post(url, json=myobj)


def process_data_buffer(data_buffer, sfreq, ch_names, filter_type='bandpass'):
    # Create a minimal Info object
    # info = create_info(ch_names=ch_names, sfreq=sfreq)
    # Assuming all channels are EEG channels
    data_to_process = np.copy(data_buffer[:, 0:64].T)
    ch_types = ['eeg'] * len(ch_names)
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object from the data buffer
    raw = RawArray(data_to_process, info)

    # Pick channels by type (in this case, EEG channels)
    picks = mne.pick_types(info, meg=False, eeg=True)
    print("info:", info)

    # Apply filtering to the data (only to the channels specified by picks)
    if filter_type == 'bandpass':
        # Adjust frequency bands as needed
        raw = raw.filter(l_freq=1, h_freq=30, picks=picks)
    # Other preprocessing steps can be applied here if necessary

    return raw.get_data(picks=picks)

# def process_data_buffer(data_buffer, sfreq, ch_names, filter_type='bandpass'):
#     # Assuming all channels are EEG channels
#     ch_types = ['eeg'] * len(ch_names)
#     info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

#     # Result container to store processed segments
#     result = []

#     # Split data_buffer into segments of size 500
#     segment_size = 500
#     for start_idx in range(0, data_buffer.shape[0], segment_size):
#         segment = data_buffer[start_idx:start_idx + segment_size, 0:64].T

#         # Create Raw object from the segment
#         raw = RawArray(segment, info)

#         # Pick channels by type (in this case, EEG channels)
#         picks = mne.pick_types(info, meg=False, eeg=True)

#         # Apply filtering to the data (only to the channels specified by picks)
#         if filter_type == 'bandpass':
#             # Adjust frequency bands as needed
#             raw.filter(l_freq=0.1, h_freq=30, picks=picks, method='fir', fir_design='firwin')

#         # Get the filtered data and add it to the result container
#         result.append(raw.get_data(picks=picks))

#     # Concatenate the processed segments
#     processed_data = np.concatenate(result, axis=1)

#     return processed_data

def main():

    # ************************************************************
    # ********************** user params ************************
    # ************************************************************

    # params
    # size of ringbuffer in samples for ensemble model, currently set to 2505 (5 sec data times 500 Hz sampling rate)
    buffer_size = 2505
    
    # TO DO : Now 2500 is not working for resNet model, only 2505 is working. Need to figure out why.
    # size of ringbuffer in samples for resNet model, currently set to 2500 (5 sec data times 500 Hz sampling rate)
    #buffer_size=2500
    
    # time in seconds how often the buffer is read  (updated with new incoming chunks)
    dt_read_buffer = 0.1 # dt for read buffer in ensemble model
    #dt_read_buffer = 1.0  # dt for read buffer in resNet model

    # team info
    team_name = 'NeuroPrior AI'

    # secret id
    secret_id = 'NeuroPrior AI'  # Needed revised the sercet id here

    # ************************************************************
    # ************************************************************
    # ************************************************************

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')  # create data stream

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    stream_info = inlet.info()
    printMeta(stream_info)  # print stream info

    # run continiously
    running = True

    # uncomment if data should be recorded (not necessary for online prediction, use buffer for that)
    # data_arr = []
    # time_stamp_arr = []

    # inits
    # data buffer has shape (buffer_size, n_channels) last 2 channels are only markers and indices !
    data_buffer = np.zeros((buffer_size, stream_info.channel_count()))
    # buffer for different kinds of timestamps or time values
    timestamp_buffer = np.zeros((buffer_size, 3))

    # get timestamp offset
    timestamp_offset = inlet.time_correction()

    detected_time = []
    prob_record = []
    last_detected_time = 0
    while running:        

        chunk, timestamps = inlet.pull_chunk()  # get a new data chunk

        if (chunk):  # if list not empty (new data)
            # print("[INFO] New chunk received has shape:" + str(np.array(chunk).shape))
            # print("[INFO] chunk:", chunk)
            # get timing info
            current_local_time = local_clock()
            timestamp_offset = inlet.time_correction()

            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            data_buffer, timestamp_buffer = getRingbufferValues(
                chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer)
            # ************************************************************
            # *************************** Data to use ********************
            # ************************************************************
            # data_buffer can be used for further processing and classification (maybe use parallel processing to not slow down the data access for longer model prediction times)
            # print("[INFO] Data buffer shape:", data_buffer.shape)
            # print("[INFO] Data buffer:", data_buffer)
            
            # Inside the while loop after updating data_buffer

            # If the first column of data_buffer is zero, then skip this iteration
            if data_buffer[0, 0] == 0:
                print("Skipping iteration...")
                continue
            # ************************************************************
            # ************************* Data Preprocessing ****************
            # ************************************************************
            markers = data_buffer[:, -2]  # get marker values
            indices = data_buffer[:, -1]  # get indices
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
                        'PO3', 'POZ', 'PO4', 'PO8']  # List of channel names in the order of your data

            processed_data = process_data_buffer(data_buffer, sfreq, ch_names)
            
            # Continue with error detection and other steps
            
            # Inside the main while loop after updating data_buffer

            # Debugging: Print processed data shape and some sample values
            print("[DEBUG] Processed data:", processed_data[0])

            # Split the processed data into chunks of size 500
            # processed_data_chunks = np.array_split(processed_data, processed_data.shape[0] // 500)
            processed_data_chunks = np.stack(np.split(processed_data, 5, axis=1), axis=0)
            print("[DEBUG] Processed data chunks shape:", processed_data_chunks.shape)

            # ************************************************************
            # ************************* Error Detection *******************
            # ************************************************************

            from joblib import load
            
            threshold = 0.80 # threshold for ensemble model
            #threshold =0.95 # threshold for resnet model
            error_detected = 0
            error_index_in_buffer = 0
            
            # Load the model of resnet
            #pred_prob_i = resnet_predict(processed_data_chunks)[:, 1]
            
            # Load the model of ensemble
            model = load('Models/pre-trained/Ensemble.joblib') # ensemble model for training data
            #model = load('Models/pre-trained/Ensemble_AQ59D.joblib') # ensemble model for only AQ59D data
            pred_prob_i = model.predict_proba(processed_data_chunks)[:, 1]
            
            print("[LOG] predicted probability: ", pred_prob_i)
            
            
            for index, value in enumerate(pred_prob_i):
                temp_index = index + 1
                if value > threshold and indices[temp_index * 500] - last_detected_time > 5:
                    last_detected_time = indices[temp_index * 500]
                    detected_time.append(indices[temp_index * 500] * 500)
                    prob_record.append(value)
                    error_detected = 1
                    error_index_in_buffer = temp_index * 500
                    break
                
            print("[LOG] detected err time so far: ", detected_time)
            print("[LOG] detected err prob so far: ", prob_record)

            # ************************************************************
            # *********************** Send Detected Error *****************
            # *************************************************************

            if error_detected:
                # if an error was detected, use the following lines to send the timepoint (timestamp) of detection
                local_clock_time = local_clock()
                #sendDetectedError(team_name, secret_id, timestamp_buffer[error_index_in_buffer, :], local_clock_time)

                print("[LOG] detected err at: ", indices[error_index_in_buffer],
                      " with probability: ", pred_prob_i,
                      " and sent it at: ", local_clock_time)
                
            # wait for some time to ensure a "fixed" frequency to read new data from buffer
            while ((perf_counter()-old_time) < dt_read_buffer):
                pass

            # just for checking the loop frequency
            #print("time: ", (perf_counter()-old_time)*1000)

            # uncomment to record ALL data received (not required for participants)
            # data_arr = data_arr+chunk
            # time_stamp_arr = time_stamp_arr + timestamps

        old_time = perf_counter()
        
    print("[INFO] Closing stream")


if __name__ == '__main__':
    main()
