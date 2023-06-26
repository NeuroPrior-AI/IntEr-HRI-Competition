#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:55:29 2023

@author: zhezhengren
"""

import numpy as np
import pickle
from tensorflow.keras import utils as np_utils
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
from joblib import dump
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import LSTM

n_class = 3
def create_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(chans, samples)))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class KerasClassifierWithProba(KerasClassifier):
    def predict_proba(self, X):
        preds = self.model.predict(X)
        return preds  # Assuming softmax activation function in the output layer

    _estimator_type = "classifier"

def create_cnn():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=n_class, activation='relu', input_shape=(chans, samples)))
    model.add(Flatten())
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

all_data_path = '../preprocess'
with open(all_data_path + '/X.pkl', 'rb') as f:
    X = pickle.load(f)
with open(all_data_path + '/y.pkl', 'rb') as f:
    y = pickle.load(f)
y = np_utils.to_categorical(y-1)

# X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.5)
# X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.125)

# X_train = X
# Y_train = y

chans, samples = 64, 501
n_components = 5

clfs = [('mlp', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), MLPClassifier((150,), learning_rate='adaptive', batch_size=64))),
        ('logreg', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), LogisticRegression(C=1.0))),
        ('svm', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), SVC(kernel='rbf', C=10, gamma='scale', probability=True))),
        ('rf', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), RandomForestClassifier(n_estimators=150))),
        ('xgb', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), XGBClassifier(n_estimators=150, learning_rate=0.1))),
        ('knn', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), KNeighborsClassifier(n_neighbors=5)))]

cnn = KerasClassifierWithProba(build_fn=create_cnn, epochs=10, batch_size=10, verbose=0)

lstm = KerasClassifierWithProba(build_fn=create_lstm, epochs=10, batch_size=10, verbose=0)
clfs.append(('lstm', lstm))

# Grid search for hyperparameter tuning
parameters = {'n_estimators': [100, 150, 200]}
clf_xgb = GridSearchCV(XGBClassifier(), parameters)
clfs.append(('xgb_gs', make_pipeline(XdawnCovariances(n_components), TangentSpace(metric='riemann'), clf_xgb)))

clfs.append(('cnn', cnn))

clf = VotingClassifier(estimators=clfs, voting='soft') 

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_flattened, Y_train.argmax(axis=-1))

X_res = X_res.reshape(X_res.shape[0], chans, samples)

clf.fit(X_res, y_res)
dump(clf, 'Ensemble.joblib')
preds_rg = clf.predict(X_test)

acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

# names = ['no error', 'S 80']
names = ['no error', 'S 96', 'S 80']
plt.figure(0)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis=-1), names, title='Ensemble Method')
plt.savefig('cm_rg.png')

# cv_results = cross_validate(clf, X, y.argmax(axis=-1), cv=10)
# print("10-fold cross validation accuracy: ", np.mean(cv_results['test_score']))

# F1 score, 10-fold cross validation, confusion matrix for each subject and save it in the folders
# 10-fold cross validation

