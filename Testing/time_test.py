# Import necessary packages
import time
from Algorithms.utils.probmap_utils import split_eeg
import mne
import numpy as np
import itertools
from joblib import load
from Algorithms.utils.resnet_predict import resnet_predict

raw_fname = "Dataset/test data/BY74D/data/20230424_BY74D_orthosisErrorIjcai_multi_set6.vhdr"
precision = 10
duration = 1
tmin = -0.1
tmax = 0.9

pretrined_model_path = "Models/pre-trained/"
ensemble = load(pretrined_model_path + 'Ensemble.joblib')
X = split_eeg(raw_fname, duration, 0, tmin, tmax)
print("X.shape: ", X.shape)
pred_prob_ensemble = ensemble.predict_proba(X[1:2])[:, 1]

start = time.time()
pred_prob_ensemble = ensemble.predict_proba(X)[:, 1]
end = time.time()
print("pred_prob_ensemble: ", pred_prob_ensemble)
print("Time elapsed for ensemble: ", (end - start)/X.shape[0])


pred_prob_resnet = resnet_predict(X[1:2])[:, 1]

start = time.time()
pred_prob_resnet = resnet_predict(X)[:, 1]
end = time.time()
print("pred_prob_resnet: ", pred_prob_resnet)
print("Time elapsed for resnet: ", (end - start)/X.shape[0])
